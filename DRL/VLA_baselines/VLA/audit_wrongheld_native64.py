"""Fresh causal/native observations for the four isolated TRAIN wrong-held cases.

Offline data engineering only: no VLA fallback, evaluated SR, live-source
mutation, automatic admission or fitting. Stored physical acquisition controls
are replayed, not regenerated, and never become supervised labels. Every case
is retained, including failures. The new native execution path is independently
replayed with the existing camera-rounding/state/safety gates.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from VLA.benchmark import _interaction_audit
from VLA.collect_demos import collect_episode
from VLA.collect_wrong_held_recovery_pilot import PhysicalWrongHeldReset
from VLA.common.artifact_io import atomic_json
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.evaluate_smolvla_v9_source import reset_observation_errors, validate_reset_observation
from VLA.pi0.scene32_inputs import RAW_SLOTS, state32
from VLA.scene64_conditioning import augment_prompt, encode_numeric, parse_prompt
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection
from VLA.simstate_scene64 import scene_state

TASKS = ("touch", "pick", "move", "pick_place")


def feedback_checks(states, actions, held_after):
    states, actions = np.asarray(states), np.asarray(actions)
    if states.ndim != 2 or states.shape[1] != 64 or actions.shape != (len(states), 7):
        raise ValueError("Aligned actual before-action64 and seven command labels required")
    if not (states[0, 30] == 1 and states[0, 31] == 1 and states[0, 17] == 0):
        raise ValueError("First actual observation must identify a wrong-held body")
    if np.max(np.abs(actions[0, :6])) > 1e-5 or actions[0, 6] != -1 or held_after[0] != "":
        raise ValueError("First label must release the distractor without dragging it")
    for raw in states:
        numeric = encode_numeric(raw)
        transported = parse_prompt(augment_prompt("Pick the requested object up", raw))
        if numeric[30] != raw[30] or state32(raw)[RAW_SLOTS.index(30)] != raw[30]:
            raise ValueError("Wrong-body feedback disappeared from numeric conditioning")
        if np.max(np.abs(transported - raw)) > .000051:
            raise ValueError("Complete language64 exceeds its four-decimal transport bound")
    return {"wrong_held_observations": int(np.count_nonzero(states[:, 30])),
            "first_label_stationary_OPEN_and_wrong_body_released": True,
            "actual67_native32_and_full64_language_preserve_wrong_hold": True}


class NativeWrongHeldCapture:
    def __init__(self, env, row, native, closures):
        self.env, self.row, self.native, self.closures = env, row, native, closures
        self.states, self.history, self.requests, self.servo = [], [], [], []
        self.before, self.after, self.cursor = [], [], 0
        self.quality = NearSurfaceTouchCollection(env, None, None) if row['task']['name'] == 'touch' else env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        if task != TaskSpec.from_dict(self.row['task']):
            raise ValueError("Original instruction/scene changed")
        obs = self.env.reset(task)
        saved = {k: copy.deepcopy(getattr(self.env, k)) for k in
                 ('initial_pick_z', 'initial_pick_position', 'initial_pick_top', 'initial_pick_bottom')}
        retarget = PhysicalWrongHeldReset(self.env)
        retarget.retarget(TaskSpec.from_dict(self.row['startup_task_not_training_instruction']))
        self.previous_rgb = (obs.front_rgb.copy(), obs.wrist_rgb.copy())
        for action in self.row['recovery_start_actions_not_training_labels']:
            self.previous_rgb = (obs.front_rgb.copy(), obs.wrist_rgb.copy())
            obs, _, done, info = self.env.step(np.asarray(action, np.float32))
            if done or info.get('safety_violation'):
                raise ValueError("Stored physical acquisition controls terminated/unsafe")
        retarget.retarget(task)
        for key, value in saved.items():
            setattr(self.env, key, value)
        evidence = _interaction_audit(self.env)
        if (evidence['held_object'] != self.row['recovery_start_evidence']['held_object']
                or evidence['correct_object_held']
                or not self.env.gripper._bilateral(self.env.gripper.grasped_object)):
            raise ValueError("Physical wrong-body identity/bilateral acquisition differs")
        if np.linalg.norm(np.asarray(evidence['object_base_xyz_m']) - saved['initial_pick_position']) > .001:
            raise ValueError("Requested target moved during distractor acquisition")
        self.env.step_count, self.env.success_streak, self.env._last_success = 0, 0, False
        if self.env.is_strict_success():
            raise ValueError("Initially completed correction is not a recovery example")
        return self.env.observe()

    def step(self, action):
        current = self.env.observe()
        self.history.append(tuple(v.copy() for v in self.previous_rgb))
        self.previous_rgb = (current.front_rgb.copy(), current.wrist_rgb.copy())
        self.states.append(scene_state(self.env, current))
        self.before.append(_interaction_audit(self.env)['held_object'] or '')
        self.requests.append(np.asarray(action, np.float32).copy())
        obs, reward, done, info = self.quality.step(action)
        self.servo.append(self.env.requested_arm_targets.copy())
        self.after.append(_interaction_audit(self.env)['held_object'] or '')
        self.cursor += 1
        if info.get('safety_violation') or info.get('expert_quality_failure'):
            raise ValueError("Unchanged safety/gentle-contact guard failed")
        if self.cursor < len(self.native):
            done = False  # retain the whole predefined expert-label sequence
        elif self.cursor == len(self.native):
            if not info['strict_raw_success']:
                raise ValueError("Full native sequence misses terminal strict predicate")
            if self.row['task']['name'] == 'touch':
                if not self.quality._last_success:
                    raise ValueError("Terminal touch exceeds existing 0.2mm/5N expert-quality gate")
            elif not (self.env.gripper.grasped_object == self.env.pick_id
                      and self.env.gripper._bilateral(self.env.pick_id)):
                raise ValueError("Requested object not held bilaterally at terminal")
            done = True
        else:
            raise ValueError("No synthetic extension/retry of failed labels")
        return obs, reward, done, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Never overwrite an earlier audit')
    plan = json.loads((candidate/'plan.json').read_text())
    rows = json.loads((candidate/'index.json').read_text())
    if (not plan['candidate_only'] or len(rows) != 4
            or tuple(r['task']['name'] for r in rows) != TASKS
            or plan.get('final_test_used') or plan.get('development_seeds_used')):
        raise ValueError('Exact four predeclared TRAIN-only cases required')
    frozen = dict(plan['source_hashes'])
    for path in (candidate/'plan.json', candidate/'index.json', Path(__file__).resolve()):
        frozen[str(path)] = sha256_file(path)
    for row in rows:
        path = candidate/row['file']
        frozen[str(path)] = sha256_file(path)
    if any(sha256_file(Path(path)) != value for path, value in frozen.items()):
        raise ValueError('Historical source/collector evidence changed before audit')
    output.mkdir(parents=True)
    atomic_json(output/'plan.json', {
        'scope': 'isolated_TRAIN_wrongheld_native64_candidate_NOT_VLA_SR_or_fitting_admission',
        'candidate_only': True, 'source': plan['source'], 'parent': str(candidate),
        'selected_source_files': plan['selected_source_files'], 'planned_episodes': 4,
        'source_hashes': frozen, 'physical_startup_supervised': False,
        'runtime_or_scoring_changed': False, 'model_actions_replaced': False,
        'final_test_used': False, 'development_seeds_used': False})
    results, new_index = [], []
    for number, row in enumerate(rows):
        result = {'task': row['task']['name'], 'passed': False, 'error': None}
        fresh = output/('native64_'+row['file'])
        with np.load(candidate/row['file'], allow_pickle=False) as archive:
            native = np.asarray(archive['next_joint_targets'], np.float32)
            closures = np.asarray(native_gripper_closure(archive), np.float32)
        def environment():
            env = audit_environment(Path(plan['source']), image_size=384, control_hz=10,
                                    seed=0, required_success_steps=1, use_semantic_success=False)
            env.dist_threshold, env.obj_threshold = recorded_distance_thresholds(row)
            return env
        env = environment()
        try:
            capture = NativeWrongHeldCapture(env, row, native, closures)
            class Sequence:
                def __init__(self, _env): pass
                def reset(self, _task): self.cursor = 0
                def predict(self, _obs):
                    i = self.cursor
                    self.cursor += 1
                    return env.joint_targets_to_canonical(native[i], float(closures[i]))
            arrays, meta = collect_episode(capture, TaskSpec.from_dict(row['task']), policy_factory=Sequence)
            if len(arrays['canonical_actions']) != len(native):
                raise ValueError('Predeclared native sequence truncated')
            arrays['next_joint_targets'] = native.copy()
            arrays['next_requested_gripper_closure'] = closures.copy()
            arrays['audit_pre_action_PRIVILEGED_state64'] = np.asarray(capture.states, np.float32)
            arrays['audit_previous_front_rgb'] = np.asarray([v[0] for v in capture.history], np.uint8)
            arrays['audit_previous_wrist_rgb'] = np.asarray([v[1] for v in capture.history], np.uint8)
            arrays['audit_applied_servo_joint_targets'] = np.asarray(capture.servo)
            arrays['audit_requested_canonical_before_safety'] = np.asarray(capture.requests, np.float32)
            arrays['audit_any_held_object_before'] = np.asarray(capture.before)
            arrays['audit_any_held_object_after'] = np.asarray(capture.after)
            checks = feedback_checks(capture.states, arrays['canonical_actions'], capture.after)
            np.savez_compressed(fresh, **arrays)
            folder = output/f'case{number:03d}'
            folder.mkdir()
            for camera in ('front', 'wrist'):
                for label, i in [('initial', 0), ('released', 1), ('middle', len(native)//2)]:
                    Image.fromarray(arrays[camera+'_rgb'][i]).save(folder/f'{label}_{camera}.png')
                Image.fromarray(getattr(env.observe(), camera+'_rgb')).save(folder/f'terminal_{camera}.png')
            result.update(frames=len(native), feedback_checks=checks,
                          actual_native_collection_strict_success=bool(meta['success']))
            new_index.append({**copy.deepcopy(row), **meta, 'file': fresh.name,
                              'payload_sha256': sha256_file(fresh), 'fresh_native_observations': True})
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        finally:
            env.close()
        if not result['error']:
            env = environment()
            try:
                with np.load(fresh, allow_pickle=False) as data:
                    replay = NativeWrongHeldCapture(env, row, native, closures)
                    obs = replay.reset(TaskSpec.from_dict(row['task']))
                    state_max = servo_max = 0.
                    pixel_maxima = {}
                    for i in range(len(native)):
                        reference = {k: data[k][i:i+1] for k in ('joint_positions', 'gripper_closure', 'front_rgb', 'wrist_rgb')}
                        current_errors = reset_observation_errors(obs, env.normalized_gripper_closure(), reference)
                        validate_reset_observation(current_errors)
                        previous = SimpleNamespace(joint_positions=obs.joint_positions,
                            front_rgb=replay.previous_rgb[0], wrist_rgb=replay.previous_rgb[1])
                        prev_reference = {**reference, **{c+'_rgb': data['audit_previous_'+c+'_rgb'][i:i+1] for c in ('front', 'wrist')}}
                        previous_errors = reset_observation_errors(previous, env.normalized_gripper_closure(), prev_reference)
                        validate_reset_observation(previous_errors)
                        for label, errors in [('current', current_errors), ('previous', previous_errors)]:
                            for key, value in errors.items():
                                pixel_maxima[label+'_'+key] = max(pixel_maxima.get(label+'_'+key, 0), value)
                        error = float(np.max(np.abs(scene_state(env, obs)-data['audit_pre_action_PRIVILEGED_state64'][i])))
                        state_max = max(state_max, error)
                        if error > 1e-6 or (_interaction_audit(env)['held_object'] or '') != data['audit_any_held_object_before'][i]:
                            raise ValueError('Causal state/held identity before label differs')
                        action = env.joint_targets_to_canonical(native[i], float(closures[i]))
                        np.testing.assert_array_equal(np.asarray(action, np.float32), data['audit_requested_canonical_before_safety'][i])
                        obs, _, _, _ = replay.step(action)
                        error = float(np.max(np.abs(replay.servo[-1]-data['audit_applied_servo_joint_targets'][i])))
                        servo_max = max(servo_max, error)
                        if error > 1e-6 or replay.after[-1] != data['audit_any_held_object_after'][i]:
                            raise ValueError('Applied servo/held identity after label differs')
                result.update(passed=True, exact_causal_state_max_error=state_max,
                              applied_servo_max_error=servo_max, existing_RGB_gate_sensor_maxima=pixel_maxima)
            except Exception as error:
                result['error'] = f'{type(error).__name__}: {error}'
            finally:
                env.close()
        results.append(result)
        atomic_json(output/'index.json', new_index)
        atomic_json(output/'progress.json', {'completed_attempts': len(results), 'planned_attempts': 4, 'rows': results})
        print(json.dumps(result), flush=True)
    unchanged = all(sha256_file(Path(path)) == value for path, value in frozen.items())
    proof = {'passed': unchanged and len(results) == 4 and all(r['passed'] for r in results),
        'scope': 'offline_four_TRAIN_wrongheld_native64_physics_and_feedback_NOT_VLA_SR',
        'rows': results, 'bound_files': frozen, 'original_sources_unchanged': unchanged,
        'neural_weights_loaded': False, 'new_VLA_trials': 0, 'training_qualified': False,
        'automatically_added_to_live_training': False, 'model_SR_established': False,
        'remaining_gates': ['larger independent-layout/task/target coverage',
            'explicit separate recovery-data admission and native mixed-loader gates',
            'own CUDA/precision/reload qualification before any additional fitting']}
    atomic_json(output/'audit.json', proof)
    if not proof['passed']:
        raise ValueError('Preserve the failed cases; do not train')


if __name__ == '__main__':
    main()
