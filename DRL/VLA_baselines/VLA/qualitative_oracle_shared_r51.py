"""CPU preflight: existing feedback Oracle on the FROZEN VLA physical world.

An explicitly adapted Oracle baseline, NOT a trained RL/VLA policy. Preserve
the existing feedback trajectory code while bridging its executor API to the
VLA's unchanged assets, body-contact gripper, initial pose and physics guard.
Never run this controller under a VLA label.
"""
import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.qualitative_recorder_r51 import COHORT, StateRecorder, canonical_hash, physical_state
from env_eval.feedback_environment import FeedbackEnvironment
from env_eval.feedback_motion import JointReference
from env_eval.oracle_feedback import FeedbackRollout


class SharedFeedbackRollout(FeedbackRollout):
    """Only capture changes; inherited feedback curves still generate control."""
    def capture(self):
        if getattr(self, 'initializing', False):
            return
        env = self.env
        self.samples.append({'time': self.tick_count/env.physics_hz, 'phase': self.phase,
            'tcp': env.gripper.grasp_center().tolist(),
            'object': list(p.getBasePositionAndOrientation(env.pick_id)[0]),
            'joints': [p.getJointState(env.robot_id, j)[0] for j in env.robot_spec.arm_joint_ids],
            'gripper_width': env.gripper.width(), 'bilateral': env.gripper._bilateral(env.pick_id),
            'held': env.gripper.grasped_object == env.pick_id})


def bind_executor(env):
    """No load/reset/changeDynamics/collision edits. Python API adaptation only."""
    def pose(self):
        state = p.getLinkState(self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])
    env.pose = MethodType(pose, env)
    env.q = np.array([p.getJointState(env.robot_id, j)[0] for j in env.robot_spec.arm_joint_ids])
    env.q_velocity = np.zeros(6)
    env.reference = JointReference(env.q, 1/env.physics_hz, 35.)
    env.reference_acceleration = np.zeros(6)
    env.offset_filter = None
    env.last_ik = env.q.copy()
    env.physics_commands = []
    env.preview_sampler = None
    env.touch_reference = None
    env.pressure_constraint = None
    env.self_pairs, env.arm_gripper_pairs = [], []
    env.demo_renderer = p.ER_TINY_RENDERER  # no image rendering in this preflight
    env.max_mount_translation_error, env.max_mount_rotation_error = 0., 0.
    env.initial_objects = {name: np.array(p.getBasePositionAndOrientation(body)[0]) for name, body in env.object_ids.items()}
    origin, orientation = env.pose()
    inverse = p.invertTransform(origin, orientation)
    offset, rotation = p.multiplyTransforms(*inverse, *p.getBasePositionAndOrientation(env.gripper.body))
    env.gripper.mount_offset = np.array(offset)
    env.gripper.mount_rotation = rotation
    # Native gripper exposes neither metric. Recompute after every real tick;
    # these are audit fields, never controller or policy inputs.
    env.gripper.max_mount_translation_error = 0.
    env.gripper.max_mount_rotation_error = 0.
    def tick(self, center, quaternion, command):
        FeedbackEnvironment.tick(self, center, quaternion, command)
        mount, quat = self.pose()
        inverse = p.invertTransform(mount, quat)
        offset, rotation = p.multiplyTransforms(*inverse, *p.getBasePositionAndOrientation(self.gripper.body))
        self.gripper.max_mount_translation_error = max(self.gripper.max_mount_translation_error,
            float(np.linalg.norm(np.array(offset)-self.gripper.mount_offset)))
        dot = np.clip(abs(np.dot(rotation, self.gripper.mount_rotation)), 0, 1)
        self.gripper.max_mount_rotation_error = max(self.gripper.max_mount_rotation_error, float(np.degrees(2*np.arccos(dot))))
        self.max_mount_translation_error = self.gripper.max_mount_translation_error
        self.max_mount_rotation_error = self.gripper.max_mount_rotation_error
    env.tick = MethodType(tick, env)


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((COHORT/'plan.json').read_text())
    profile = json.loads(Path(plan['observation_profile_file']).read_text())
    expected = json.loads((args.paired_capture/'initial_scene.json').read_text())
    env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e', image_size=384,
        control_hz=10, assist_grasp=True, seed=args.seed, required_success_steps=1, use_semantic_success=True,
        success_threshold_scale=plan['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=plan['success']['semantic_thresholds'])
    rollout = None
    try:
        for spec in plan['trials']:
            if spec['seed'] != args.seed:
                continue
            task = TaskSpec.from_dict(spec['task'])
            env.reset(task)
            if task.name == args.task:
                break
        initial = physical_state(env)
        if (canonical_hash(initial) != expected['measured_state_sha256'] or
                canonical_hash(task.to_dict()) != canonical_hash(expected['task'])):
            raise ValueError('Oracle and SmolVLA measured initial scene NOT equal')
        atomic_json(out/'initial_scene.json', {'task': task.to_dict(), 'measured_state': initial,
            'measured_state_sha256': canonical_hash(initial), 'matches_SmolVLA_exactly': True})
        bind_executor(env)
        if canonical_hash(physical_state(env)) != expected['measured_state_sha256']:
            raise ValueError('API adapter altered physical world')
        action = 'put' if args.task == 'pick_place' else args.task
        scene = SimpleNamespace(action=action, config=args.seed, pick=task.pick_name, place=task.place_name)
        rollout = SharedFeedbackRollout(env, scene, out, video=False)
        rollout.state_capture = StateRecorder(env, out)
        physics_ticks = 0
        benchmark_samples = []
        original_step, original_post = p.stepSimulation, env.gripper.post_step
        def counted_step(*a, **kw):
            nonlocal physics_ticks
            value = original_step(*a, **kw)
            physics_ticks += 1
            return value
        def sampled_post():
            safe_post_step = False
            try:
                value = original_post()
                safe_post_step = True
                return value
            finally:
                if safe_post_step and physics_ticks % 24 == 0:
                    benchmark_samples.append({'physics_tick': physics_ticks,
                        'semantic_success': bool(env.is_semantic_success())})
                if physics_ticks % 8 == 0:
                    rollout.state_capture.capture(physics_ticks, physics_ticks-1)
                    rollout.state_capture.frames[-1]['oracle_phase'] = rollout.phase
        atomic_json(out/'protocol.json', {
            'controller': 'ADAPTED_existing_env_eval_feedback_Oracle_NOT_trained_RL_or_VLA',
            'source': 'env_eval.oracle_feedback.FeedbackRollout',
            'source_sha256': sha256_file(Path('env_eval/oracle_feedback.py')),
            'executor_source': 'env_eval.feedback_environment.FeedbackEnvironment.tick',
            'initial_state_and_task_equal_to_SmolVLA': True, 'asset_home_dynamics_gripper_unchanged': True,
            'API_adaptations': ['mount transform measured from shared body-contact gripper',
                'native VLA body-contact motor/assist/guard retained; legacy pregrasp-force attributes are not implemented',
                'shared VLA collision filters retained; no Oracle-only collision proxies/compliance',
                'existing Oracle reference servo250N/gain0.16/jerk35 retained as controller, not plant change'],
            'not_identical_to_original_reference_Oracle_environment': True,
            'wrapper_sha256': sha256_file(Path(__file__)),
            'not_SR_comparison': True, 'staged_faults': False})
        p.stepSimulation, env.gripper.post_step = counted_step, sampled_post
        try:
            result = rollout.run()
            status = 'controller_completed'
        except Exception as error:
            result = {'error': f'{type(error).__name__}: {error}', 'success': False}
            status = 'controller_failed_preserved'
        finally:
            p.stepSimulation, env.gripper.post_step = original_step, original_post
        atomic_json(out/'outcome.json', {'status': status, 'result': result,
            'recorded_physics_ticks': physics_ticks,
            'completed_executor_command_count': len(env.physics_commands),
            'benchmark_region_success_at_terminal': bool(env.is_semantic_success()),
            'benchmark_region_success_at_any_matched_10Hz_sample': any(s['semantic_success'] for s in benchmark_samples),
            'benchmark_first_hit_10Hz_tick': next((s['physics_tick'] for s in benchmark_samples if s['semantic_success']), None),
            'benchmark_10Hz_samples': benchmark_samples,
            'observed_correct_attachment': any(s['held'] for s in rollout.samples),
            'scene_initial_parity_pass': True, 'qualitative_controller_variant_NOT_original_reference': True})
        atomic_json(out/'trajectory.json', rollout.samples)
        atomic_json(out/'physics_commands.json', env.physics_commands)
        rollout.state_capture.close()
        print(json.dumps({'output': str(out), 'status': status, 'success': result.get('success'),
                          'error': result.get('error')}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--task', choices=['touch', 'pick', 'move', 'pick_place'], required=True)
    parser.add_argument('--paired-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
