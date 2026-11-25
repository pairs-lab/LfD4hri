"""CPU-only replay of recorded NEURAL controls, never a new policy rollout.

Recover the actual offending linkage/contact state before safety rollback.
All64 pre-action observations must match the recorded rollout; divergence
is reported and never silently treated as proof of its failure cause.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.scene64_conditioning import model_input


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.progress.read_text())
    selected = [r for r in original['rows'] if r.get('safety_violation')]
    protocol = json.loads(PROFILE_PROTOCOL.read_text())
    profile = json.loads(Path(protocol['observation_profile_file']).read_text())
    if profile_hash(profile) != protocol['observation_profile_sha256']:
        raise ValueError('Frozen observation profile changed')
    atomic_json(args.output/'plan.json', {
        'scope': 'REPLAY_of_saved_neural_controls_NOT_new_task_SR',
        'source_progress': str(args.progress.resolve()),
        'source_sha256': sha256_file(args.progress),
        'cases': [{'seed': r['seed'], 'task': r['task']} for r in selected],
        'GPU_model_loaded': False, 'controls_modified': False,
        'physics_camera_success_and_guards_modified': False,
        'teacher_or_geometric_policy_used': False,
    })
    rows = []
    for row in selected:
        rng = np.random.default_rng(row['seed'])
        tasks = [randomize_rl_scene(t, rng) for t in select_tasks('all')]
        task = next(t for t in tasks if t.name == row['task'])
        trajectory = Path(row['trajectory'])
        trace = json.loads(trajectory.read_text())[task.name]
        env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e',
            image_size=384, control_hz=10, assist_grasp=True, seed=row['seed'],
            required_success_steps=1, use_semantic_success=True,
            success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
            semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        result = {'seed': row['seed'], 'task': task.name, 'recorded_steps': len(trace),
            'source_trajectory_sha256': sha256_file(trajectory)}
        try:
            obs = env.reset(task)
            maximum = 0.
            for index, frame in enumerate(trace):
                actual = model_input(env, obs)['scene64']
                expected = np.asarray(frame['pre_action_scene64'])
                delta = float(np.max(np.abs(actual-expected)))
                maximum = max(maximum, delta)
                if delta > 1e-5:
                    raise ValueError(f'Physical replay differs at step{index}: max64 error{delta}')
                obs, _, done, info = env.step(np.asarray(frame['action'], np.float32))
                if done and index != len(trace)-1:
                    raise ValueError('Physical replay terminated at a different step')
            diagnostics = info.get('safety_diagnostics_before_rollback')
            if not diagnostics or info.get('safety_violation') != row['safety_violation']:
                raise ValueError('Original exact safety abort not reproduced')
            q0 = diagnostics['joint_positions_rad'][0]
            result.update(status='exact_abort_reproduced', max_pre_action64_error=maximum,
                diagnostics_before_rollback=diagnostics,
                exceeded_checks={
                    'penetration_over_0p8mm': diagnostics['surface_clearance_m'] < -.0008,
                    'follower_error_over_0p15rad': max(abs(v) for v in diagnostics['follower_errors_rad']) > .15,
                    'joint_speed_over15rad_s': max(abs(v) for v in diagnostics['joint_velocities_rad_s']) > 15,
                    'driver_outside_safe_interval': not -.015 <= q0 <= .849,
                },
                contact_body_names={str(c['other_body']): next((name for name, body in env.object_ids.items()
                    if body == c['other_body']), str(c['other_body']))
                    for c in diagnostics['contacts']})
        except Exception as error:
            result.update(status='replay_diverged_NO_causal_claim', error=f'{type(error).__name__}: {error}')
        finally:
            env.close()
        rows.append(result)
        atomic_json(args.output/'audit.json', {'status': 'running', 'rows': rows,
            'new_task_SR_established': False})
        print(json.dumps({'seed': row['seed'], 'task': task.name, 'status': result['status']}), flush=True)
    atomic_json(args.output/'audit.json', {'status': 'completed', 'rows': rows,
        'exact_reproductions': sum(r['status'] == 'exact_abort_reproduced' for r in rows),
        'new_task_SR_established': False, 'controls_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
