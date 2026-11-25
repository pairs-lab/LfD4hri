"""TRAIN-only actual-physics off-manifold corrections, never deployed control.

Old frozen model/robot/camera code remains unchanged. Perturbations are real
canonical motor actions, not teleports. Only subsequent expert corrections
become supervised labels. Every candidate (including failure) is preserved.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
from VLA.collect_demos import collect_episode
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.simstate_scene64 import scene_state

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64/audit.json'
NATIVE = ROOT/'VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts'


def upright_error(columns):
    columns = np.asarray(columns, np.float64)
    rotation = np.column_stack((columns[:3], columns[3:], np.cross(columns[:3], columns[3:])))
    x = rotation[:, 0].copy()
    x[2] = 0.
    x /= np.linalg.norm(x)
    z = np.array([0., 0., -1.])
    target = np.column_stack((x, np.cross(z, x), z))
    relative = (Rotation.from_matrix(rotation).inv()*Rotation.from_matrix(target)).as_euler('xyz')
    tilt = np.arccos(np.clip(-rotation[2, 2], -1, 1))
    return relative, float(tilt)


class UprightCorrectionTeacher(CentralSectionClearanceTeacher):
    """Supervised labels only. Causal measured orientation restoration."""
    def predict(self, observation):
        relative, tilt = upright_error(observation.ee_rotation_6d)
        if tilt > .008:
            held = self.env.gripper.grasped_object == self.env.pick_id
            # Restore orientation above obstacles, not while descending through
            # them. This is a teacher label, NEVER a VLA runtime action repair.
            height = float(self.env.gripper.grasp_center()[2])
            return np.r_[0., 0., .006 if height < .18 else 0.,
                np.clip(.65*relative, -.025, .025), 1. if held else -1.].astype(np.float32)
        return super().predict(observation)


class PerturbedStart:
    def __init__(self, env, source, row, numeric, kind):
        self.env, self.source, self.row, self.numeric, self.kind = env, source, row, numeric, kind
        self.states, self.prefix = [], []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        if task != TaskSpec.from_dict(self.row['task']):
            raise ValueError('Do not change the TRAIN task, target or scene')
        obs = reset_source_episode(self.env, self.source, self.row)
        raw, actions = self.numeric['scene64'], self.numeric['canonical7']
        candidates = np.flatnonzero((raw[:, 9] <= .23) & (raw[:, 9] >= .18) & (raw[:, 31] == 0))
        if not len(candidates):
            raise ValueError('No predeclared safe high unheld anchor in original TRAIN source')
        anchor = int(candidates[0])
        for i in range(anchor):
            np.testing.assert_allclose(scene_state(self.env, obs), raw[i], rtol=0, atol=1e-6)
            obs, _, done, info = self.env.step(actions[i])
            self.prefix.append({'kind': 'original_TRAIN_command', 'action': actions[i].tolist()})
            if done or info.get('safety_violation'):
                raise ValueError('Exact original TRAIN prefix terminated/aborted')
        np.testing.assert_allclose(scene_state(self.env, obs), raw[anchor], rtol=0, atol=1e-6)
        perturb = ([.012, -.010, 0., .045, -.040, .025, -1.] if self.kind == 'wrist_tilt' else
                   [-.015, .012, 0., 0., 0., 0., 1.])
        for _ in range(1 if self.kind == 'wrist_tilt' else 8):
            action = np.asarray(perturb, np.float32)
            # Empty-close stays in the same safe air position after its first
            # measured lateral perturbation. It never sweeps0.12m while closed.
            if self.kind == 'closed_empty' and self.prefix and self.prefix[-1]['kind'] == self.kind:
                action[:3] = 0
            obs, _, done, info = self.env.step(action)
            self.prefix.append({'kind': self.kind, 'action': action.tolist()})
            if done or info.get('safety_violation') or self.env.gripper.grasped_object is not None:
                raise ValueError('Perturbation failed safe-air/unheld gate; preserve case')
        state = scene_state(self.env, obs)
        if self.kind == 'closed_empty' and state[6] < .8:
            raise ValueError('The intended actual measured empty closure was not reached')
        if self.kind == 'wrist_tilt' and upright_error(obs.ee_rotation_6d)[1] < .02:
            raise ValueError('The intended physical tilt was not reached')
        self.initial_state = state.tolist()
        self.original_anchor = anchor
        self.env.step_count = self.env.success_streak = 0
        self.env._last_success = False
        self.last_obs = self.env.observe()
        return self.last_obs

    def step(self, action):
        self.states.append(scene_state(self.env, self.last_obs))
        result = self.env.step(action)
        self.last_obs = result[0]
        return result


def replay_candidate(env, source, row, numeric, kind, arrays):
    proxy = PerturbedStart(env, source, row, numeric, kind)
    obs = proxy.reset(TaskSpec.from_dict(row['task']))
    max_error = 0.
    rgb_differences = []
    for i, action in enumerate(arrays['canonical_actions']):
        actual = scene_state(env, obs)
        expected = arrays['audit_pre_action_PRIVILEGED_state64'][i]
        error = float(np.max(np.abs(actual-expected)))
        max_error = max(max_error, error)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        for camera in ('front', 'wrist'):
            actual_rgb = getattr(obs, camera+'_rgb')
            expected_rgb = arrays[camera+'_rgb'][i]
            difference = np.abs(actual_rgb.astype(np.int16)-expected_rgb.astype(np.int16))
            count, maximum = int(np.count_nonzero(difference)), int(difference.max())
            if count:
                rgb_differences.append({'frame': i, 'camera': camera, 'different_channel_values': count, 'maximum_LSB_error': maximum})
            # A preserved pilot had ONE pixel channel differ by1LSB while
            # all measured before64/action/outcome checks reproduced. Record
            # this numerical rasterization tolerance, NOT a bit-exact claim.
            if maximum > 1 or count > 16:
                raise ValueError(f'RGB replay differs beyond declared tiny rasterization tolerance: {camera}, frame{i}, max{maximum}, count{count}')
        obs, _, done, info = proxy.step(action)
        if bool(info['raw_success']) != bool(arrays['next_raw_success'][i]):
            raise ValueError('Correction outcome timeline not reproduced')
        if done != (i == len(arrays['canonical_actions'])-1):
            raise ValueError('Correction termination not reproduced')
    return {'pre_action64_reproduced_at1e6_and_RGB_with_declared_rasterization_tolerance': True,
            'all_RGB_bit_exact': not rgb_differences, 'RGB_rasterization_differences': rgb_differences,
            'maximum_RGB_LSB_tolerance': 1, 'maximum_different_channel_values_per_frame_tolerance': 16,
            'maximum_state64_error': max_error,
            'controls': len(arrays['canonical_actions']), 'terminal_success': bool(info['success']),
            'safety_violation': info.get('safety_violation')}


def review_images(out, stem, arrays):
    n = len(arrays['canonical_actions'])
    near_close = np.flatnonzero(arrays['canonical_actions'][:, 6] > .5)
    events = sorted({0, min(n-1, 4), int(near_close[0]) if len(near_close) else n//2, n-1})
    sheet = np.full((len(events)*384, 2*384, 3), 240, np.uint8)
    for j, i in enumerate(events):
        for k, camera in enumerate(('front', 'wrist')):
            sheet[j*384:(j+1)*384, k*384:(k+1)*384] = arrays[camera+'_rgb'][i]
    cv2.imwrite(str(out/(stem+'_actual_sensor_sheet.png')), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return {'events': events, 'native_sensor_resolution': [384, 384], 'source': 'actual_before_label_RGB_NOT_video_camera'}


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    proof = json.loads(PROJECTION.read_text())
    source = Path(proof['source'])
    if sha256_file(source/'index.json') != proof['source_index_sha256']:
        raise ValueError('Original TRAIN source changed')
    manifest = json.loads((source/'dataset_manifest.json').read_text())
    if manifest['split'] != 'train' or manifest['final_test_used']:
        raise ValueError('TRAIN-only physical examples required')
    rows = json.loads((source/'index.json').read_text())
    native = {r['source']: r for r in json.loads((NATIVE/'index.json').read_text())}
    targets = args.targets.split(',')
    chosen = [next(r for r in rows if r['task']['name'] == task and r['task']['pick_name'] == target
        and not r.get('paired_scene_source') and not r.get('recovery_start_actions_not_training_labels')
        and not r['file'].endswith('_closedhover.npz')) for task in ('touch', 'pick', 'move', 'pick_place') for target in targets]
    planned = 2*len(chosen)
    atomic_json(out/'plan.json', {'scope': 'TRAIN_only_actual_physics_pose_recovery_candidate_NOT_VLA_success_rate',
        'source': str(source), 'source_sha256': sha256_file(source/'index.json'),
        'script_sha256': sha256_file(Path(__file__)), 'selected_source_files': [r['file'] for r in chosen],
        'source_scene_groups': [r['scene_group'] for r in chosen], 'kinds': ['wrist_tilt', 'closed_empty'],
        'candidate_episodes': planned, 'targets': targets, 'prefix_and_fault_actions_are_NOT_training_labels': True,
        'RGB_replay_tolerance_explicit': {'maximum_LSB_error': 1, 'maximum_different_channel_values_per_frame': 16},
        'source_selection_uses_model_outcomes': False, 'validation_development_final_geometry_used': False,
        'physics_assets_camera_controller_success_not_modified': True, 'teacher_at_deployment': False,
        'candidate_only_NOT_automatically_admitted': True})
    env = audit_environment(source, robot='ur5e', image_size=384, control_hz=10, seed=0,
        required_success_steps=1, success_threshold_scale=1., use_semantic_success=False)
    results = []
    try:
        for row in chosen:
            nr = native[str(source/row['file'])]
            with np.load(NATIVE/nr['cache'], allow_pickle=False) as loaded:
                numeric = {key: loaded[key] for key in ('scene64', 'canonical7')}
            for kind in ('wrist_tilt', 'closed_empty'):
                stem = row['task']['name']+'_'+row['task']['pick_name'].replace(' ', '_')+'_'+kind
                result = {'source_file': row['file'], 'kind': kind, 'task': row['task'], 'success': False}
                proxy = PerturbedStart(env, source, row, numeric, kind)
                try:
                    arrays, meta = collect_episode(proxy, TaskSpec.from_dict(row['task']), policy_factory=UprightCorrectionTeacher)
                    arrays['audit_pre_action_PRIVILEGED_state64'] = np.asarray(proxy.states, np.float32)
                    path = out/(stem+'.npz')
                    np.savez_compressed(path, **arrays)
                    result.update(meta, file=path.name, payload_sha256=sha256_file(path),
                        perturbation_actions_NOT_labels=proxy.prefix, initial_state64=proxy.initial_state,
                        sensor_review=review_images(out, stem, arrays))
                    if meta['success'] and not meta.get('safety_violation'):
                        result['replay'] = replay_candidate(env, source, row, numeric, kind, arrays)
                        result['candidate_replay_pass'] = True
                except Exception as error:
                    result['error'] = f'{type(error).__name__}: {error}'
                    result['perturbation_actions_NOT_labels'] = proxy.prefix
                results.append(result)
                atomic_json(out/'index.json', results)
                atomic_json(out/'status.json', {'status': 'collecting_candidates', 'completed': len(results), 'planned': planned,
                    'actual_expert_successes_NOT_neural_SR': sum(r['success'] for r in results)})
                print(json.dumps({k: result.get(k) for k in ('kind', 'file', 'success', 'candidate_replay_pass', 'error', 'safety_violation')}), flush=True)
        atomic_json(out/'audit.json', {'scope': 'candidate_TRAIN_expert_corrections_NOT_neural_SR', 'rows': results,
            'all_declared_candidate_replay_pass': len(results) == planned and all(r.get('candidate_replay_pass') for r in results),
            'model_weights_or_checkpoint_changed': False, 'training_admission_pending_additional_checks': True,
            'future_labels_in_observations': False, 'original_sources_unchanged': True})
        atomic_json(out/'status.json', {'status': 'candidates_completed_NOT_automatically_training_ready', 'completed': len(results),
            'planned': planned, 'actual_expert_successes_NOT_neural_SR': sum(r['success'] for r in results)})
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', default='blue block', help='Predeclared TRAIN vocabulary, never selected from eval outcomes')
    main(p.parse_args())
