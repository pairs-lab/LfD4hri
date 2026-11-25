"""Qualify new TRAIN corrections, preserve ALL original episodes, widen units.

All new sensor/action timelines already have real-physics replay receipts.
This additionally checks current-pose FK and native action conversions and
fits TRAIN-only normalizers with explicit physical range floors. No eval data.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.common.environment import ROBOT_SPECS
from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import (absolute20_from_canonical, canonical_from_absolute20,
    native8, pack_native20, bounds_stats)
from VLA.scene64_conditioning import normalization
from config import BOUNDS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts'


def floor_bounds(stats, radii):
    result = copy.deepcopy(stats)
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    center = (lo+hi)/2
    radius = np.maximum((hi-lo)/2, np.asarray(radii))
    result['q01'], result['q99'] = (center-radius).tolist(), (center+radius).tolist()
    result['physical_half_span_floor'] = list(radii)
    return result


def physical_statistics(raw, proprio, actions):
    stats, original = normalization(raw)
    floor = np.zeros(67)
    floor[:6] = .05
    floor[7:10] = .03
    floor[11:17] = .03
    floor[18:24] = .05
    floor[24:27] = .01
    floor[27] = .01
    floor[32:35] = .05
    floor[35:41] = .10
    floor[41], floor[42] = .5, 5.
    for n in range(6):
        floor[43+4*n:46+4*n] = .05
        floor[46+4*n] = .02
    stats['std'] = np.maximum(stats['std'], floor).tolist()
    return {'scene67': stats, 'regularization': {**original,
        'schema': 'TRAIN_empirical_with_explicit_physical_range_floors_r52',
        'first18_empirical_preserved': False,
        'physical_std_floor67': floor.tolist(), 'all67_slots_retained': True,
        'first18_NO_longer_frozen_to_near_constant_old_statistics': True},
        'adapter': {
            'proprio': floor_bounds(bounds_stats(proprio), [0., 0., 0., .05, .05, .10, 0., 0.]),
            'action': floor_bounds(bounds_stats(actions, mask=[True]*6+[False]), [0., 0., 0., .03, .03, .08, 0.])},
        'split': 'train', 'validation_or_test_values_used': False}


def main(args):
    out, candidate = args.output.resolve(), args.candidate.resolve()
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((BASE/'audit.json').read_text())
    index = json.loads((BASE/'index.json').read_text())
    if sha256_file(BASE/'index.json') != old['index_sha256'] or sha256_file(BASE/'stats.json') != old['stats_sha256']:
        raise ValueError('Frozen base TRAIN metadata changed')
    plan = json.loads((candidate/'plan.json').read_text())
    rows = json.loads((candidate/'index.json').read_text())
    state = json.loads((candidate/'status.json').read_text())
    if len(rows) != plan['candidate_episodes'] or state['completed'] != plan['candidate_episodes'] or state['status'].startswith('collecting'):
        raise ValueError('Every predeclared candidate must finish; no cherry-picked partial bank')
    source = Path(plan['source'])
    if sha256_file(source/'index.json') != plan['source_sha256']:
        raise ValueError('Original TRAIN index changed')
    source_rows = {r['file']: r for r in json.loads((source/'index.json').read_text())}
    if json.loads((source/'dataset_manifest.json').read_text())['split'] != 'train':
        raise ValueError('Only TRAIN geometry allowed')
    accepted = [r for r in rows if r.get('success') and r.get('candidate_replay_pass') and not r.get('safety_violation')]
    if plan.get('both_roll_and_pitch_signs_per_task'):
        for task in ('touch', 'pick', 'move', 'pick_place'):
            measured = np.array([r['initial_state64'] for r in accepted
                if r['task']['name'] == task and r['kind'] == 'wrist_tilt'])
            # Real measured tilt is verified in collection/replay; commanded
            # signs here independently certify predeclared intervention support.
            faults = np.array([next(x['action'] for x in r['perturbation_actions_NOT_labels']
                if x['kind'] == 'wrist_tilt') for r in accepted
                if r['task']['name'] == task and r['kind'] == 'wrist_tilt'])
            if not len(measured) or set(map(tuple, np.sign(faults[:, 3:5]).astype(int))) != {(1,1), (1,-1), (-1,1), (-1,-1)}:
                raise ValueError('All four physical tilt intervention quadrants required in each task')
    strata = defaultdict(set)
    for row in accepted:
        strata[row['task']['name'], row['kind']].add(row['task']['pick_name'])
    expected = {(task, kind) for task in ('touch', 'pick', 'move', 'pick_place') for kind in ('wrist_tilt', 'closed_empty')}
    if set(strata) != expected or min(map(len, strata.values())) < 3:
        raise ValueError('Need all4 tasks/both interventions/at least3 target identities per stratum')
    # Absolute immutable cache references preserve every original frame, with
    # no RGB duplication and no modification of active-fit data/statistics.
    result = [{**row, 'cache': str(BASE/row['cache'])} for row in index]
    raw_all, native_all, action_all = [], [], []
    for row in index:
        with np.load(BASE/row['cache'], allow_pickle=False) as a:
            raw_all.append(a['scene64']); native_all.append(a['native8']); action_all.append(a['adapter_action7'])
    client = p.connect(p.DIRECT)
    robot = ROBOT_SPECS['ur5e']
    body = p.loadURDF(str(robot.urdf), useFixedBase=True, physicsClientId=client)
    errors, roundtrips, workspace_clipped = [], [], 0
    try:
        for i, row in enumerate(accepted):
            original = source_rows[row['source_file']]
            if original['task'] != row['task']:
                raise ValueError('Corrective source instruction/target/scene changed')
            path = candidate/row['file']
            if sha256_file(path) != row['payload_sha256']:
                raise ValueError('Corrective image/label bytes changed')
            with np.load(path, allow_pickle=False) as a:
                raw, xyz, six = a['audit_pre_action_PRIVILEGED_state64'], a['ee_position'], a['ee_rotation_6d']
                q, closure, canonical = a['joint_positions'], a['gripper_closure'], a['canonical_actions']
                if any(a[c+'_rgb'].shape != (len(raw), 384, 384, 3) for c in ('front', 'wrist')):
                    raise ValueError('Actual camera shape/domain differs')
                np.testing.assert_allclose(raw[:, :7], np.c_[q, closure], rtol=0, atol=1e-6)
                if len(raw) > 1:
                    np.testing.assert_allclose(q[1:], a['next_observed_joint_positions'][:-1], rtol=0, atol=1e-6)
                np.testing.assert_allclose((canonical[:, 6]+1)/2, a['next_requested_gripper_closure'], rtol=0, atol=1e-6)
            labels = absolute20_from_canonical(xyz, six, canonical)
            back = canonical_from_absolute20(labels, xyz, six)
            effective = canonical.copy()
            effective[:, :3] = np.clip(xyz+np.clip(canonical[:, :3], -.05, .05),
                (BOUNDS[0, 0]-.12, BOUNDS[1, 0]-.05, .10),
                (BOUNDS[0, 1]+.12, BOUNDS[1, 1]+.05, .72))-xyz
            effective[:, 3:6] = np.clip(effective[:, 3:6], -.25, .25)
            workspace_clipped += int(np.any(np.abs(effective-canonical) > 2e-6, axis=1).sum())
            error = float(np.abs(back-effective).max())
            if error > 2e-6:
                raise ValueError(f'Corrective requested/native roundtrip differs: {error}')
            roundtrips.append(error)
            for joints, pos, rotation in zip(q, xyz, six, strict=True):
                for j, value in zip(robot.arm_joint_ids, joints, strict=True):
                    p.resetJointState(body, j, float(value), physicsClientId=client)
                s = p.getLinkState(body, robot.mount_link, computeForwardKinematics=True, physicsClientId=client)
                measured = np.asarray(p.getMatrixFromQuaternion(s[5])).reshape(3, 3)[:, :2].T.reshape(6)
                errors.append(max(float(np.max(np.abs(np.asarray(s[4])-pos))), float(np.max(np.abs(measured-rotation)))))
            prefix = native8(xyz, six, raw[:, 27])
            adapter = canonical.copy(); adapter[:, 6] = 1-(canonical[:, 6]+1)/2
            cache = out/f'corrective_{i:03d}.npz'
            np.savez_compressed(cache, state20=pack_native20(xyz, six, closure), native8=prefix,
                action20=labels, adapter_action7=adapter, scene64=raw, canonical7=canonical)
            result.append({'cache': cache.name, 'cache_sha256': sha256_file(cache), 'source': str(path),
                'source_sha256': sha256_file(path), 'frames': len(raw), 'task': row['task']['name'],
                'instruction': row['task']['prompt'], 'source_kind': 'pose_corrective',
                'scene_group': original['scene_group'], 'pick_name': row['task']['pick_name'],
                'recovery_kind': row['kind'], 'TRAIN_only_physical_intervention_prefix_NOT_labels': True})
            raw_all.append(raw); native_all.append(prefix); action_all.append(adapter)
    finally:
        p.disconnect(client)
    if not errors or max(errors) > 1e-5:
        raise ValueError('Actual URDF/FK mismatch')
    stats = physical_statistics(np.concatenate(raw_all), np.concatenate(native_all), np.concatenate(action_all))
    atomic_json(out/'index.json', result)
    atomic_json(out/'stats.json', stats)
    atomic_json(out/'admission.json', {'all_attempts': len(rows), 'qualified_episodes': len(accepted),
        'all_failures_preserved_at_source': True, 'excluded': [r for r in rows if r not in accepted],
        'target_coverage_by_task_and_intervention': {str(k): sorted(v) for k, v in strata.items()},
        'data_does_NOT_certify_recovery_for_wrong_object_or_unexpected_drop': True})
    bound = {**old['bound_files'], str(BASE/'audit.json'): sha256_file(BASE/'audit.json'),
        str(candidate/'index.json'): sha256_file(candidate/'index.json'),
        str(candidate/'plan.json'): sha256_file(candidate/'plan.json'), str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    if plan.get('multidirection_extension_sha256'):
        extension = ROOT/'VLA/collect_multidirection_feedback_r52.py'
        if sha256_file(extension) != plan['multidirection_extension_sha256']:
            raise ValueError('Physical intervention extension source changed')
        bound[str(extension)] = sha256_file(extension)
    atomic_json(out/'audit.json', {'passed': True, 'scope': 'TRAIN_native_correction_admission_NOT_neural_SR',
        'episodes': len(result), 'frames': sum(r['frames'] for r in result),
        'original_full_source_coverage': True, 'all_frames_actual_URDF_FK_checked': True,
        'future_labels_in_observations': False, 'bound_files': bound,
        'index_sha256': sha256_file(out/'index.json'), 'stats_sha256': sha256_file(out/'stats.json'),
        'all_original_episodes_retained': len(index), 'qualified_new_episodes': len(accepted),
        'max_new_FK_component_error': max(errors), 'max_new_native_roundtrip_error': max(roundtrips),
        'new_frames_with_unchanged_shared_workspace_clipping': workspace_clipped,
        'normalizer_schema_changed_EXPLICITLY_new_checkpoint_required': True,
        'parent_fit_data_stats_or_checkpoint_modified': False})
    print(json.dumps({'output': str(out), 'episodes': len(result), 'new_qualified': len(accepted), 'FK_error': max(errors)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
