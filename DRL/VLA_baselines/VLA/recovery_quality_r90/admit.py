"""Qualify completed TRAIN corrections with actual URDF FK and causal labels.

Keep old statistics: this is an explicit new-data revision, not a silent
change to the numeric representation of existing checkpoint weights.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import pybullet as p

from config import BOUNDS
from VLA.common.artifact_io import atomic_json
from VLA.common.environment import ROBOT_SPECS
from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import (
    absolute20_from_canonical, canonical_from_absolute20, native8, pack_native20)
from .audit import BANK, OUT, TASKS, read, regimes
from . import collect


def main(a):
    candidate, out = a.candidate.resolve(), a.output.resolve()
    plan, rows, status = (read(candidate/n) for n in ('plan.json', 'index.json', 'status.json'))
    if status['status'] != 'pilot_complete_pending_data_admission_NOT_neural_SR' or len(rows) != plan['candidate_episodes']:
        raise ValueError('Every predeclared attempt must finish, including failures')
    source = Path(plan['source'])
    if sha256_file(source/'index.json') != plan['source_sha256'] or plan['selection_uses_DEV_instances_or_final_outcomes']:
        raise ValueError('Only immutable TRAIN scenes')
    if sha256_file(Path(collect.__file__)) != plan['script_sha256']:
        raise ValueError('Actual collector source changed')
    source_rows = {r['file']: r for r in read(source/'index.json')}
    accepted = [r for r in rows if r.get('candidate_replay_pass') and r['success'] and not r.get('safety_violation')]
    coverage = defaultdict(set)
    for row in accepted:
        coverage[row['task']['name'], row['kind']].add(row['task']['pick_name'])
    expected = {(t, k) for t in TASKS for k in plan['kinds']}
    if set(coverage) != expected or min(map(len, coverage.values())) < a.minimum_targets:
        raise ValueError('Insufficient four-task/kind target coverage; preserve failures, no blind FT')
    out.mkdir(parents=True, exist_ok=False)
    previous = read(BANK/'audit.json')
    if sha256_file(BANK/'index.json') != previous['index_sha256']:
        raise ValueError('Original TRAIN bank changed')
    index, errors, totals, gaps = [], [], Counter(), []
    robot = ROBOT_SPECS['ur5e']
    client = p.connect(p.DIRECT)
    body = p.loadURDF(str(robot.urdf), useFixedBase=True, physicsClientId=client)
    try:
        for i, row in enumerate(accepted):
            if row['task'] != source_rows[row['source_file']]['task']:
                raise ValueError('Original TRAIN task, instruction or target changed')
            path = candidate/row['file']
            if sha256_file(path) != row['payload_sha256']:
                raise ValueError('Sensor/label source changed')
            with np.load(path, allow_pickle=False) as a0:
                raw, xyz, six = (a0[n] for n in ('audit_pre_action_PRIVILEGED_state64', 'ee_position', 'ee_rotation_6d'))
                q, closure, canonical = (a0[n] for n in ('joint_positions', 'gripper_closure', 'canonical_actions'))
                np.testing.assert_allclose(raw[:, :7], np.c_[q, closure], rtol=0, atol=1e-6)
                np.testing.assert_allclose(raw[0], row['initial_state64'], rtol=0, atol=1e-6)
                np.testing.assert_allclose(q[1:], a0['next_observed_joint_positions'][:-1], rtol=0, atol=1e-6)
                np.testing.assert_allclose((canonical[:, 6]+1)/2, a0['next_requested_gripper_closure'], rtol=0, atol=1e-6)
                if len(raw) != row['steps']:
                    raise ValueError('Pre-action chronology/count mismatch')
                for camera in ('front', 'wrist'):
                    if a0[camera+'_rgb'].shape != (len(raw), 384, 384, 3):
                        raise ValueError('Native actual observation images required')
                    if a0[camera+'_pick_mask'].shape != (len(raw), 384, 384):
                        raise ValueError('Actual target visibility audit masks required')
                visibility = {camera: {
                    'frames_with_target_pixels': int((a0[camera+'_pick_mask'].reshape(len(raw), -1).sum(1)>0).sum()),
                    'minimum_target_pixels': int(a0[camera+'_pick_mask'].reshape(len(raw), -1).sum(1).min()),
                } for camera in ('front', 'wrist')}
            labels = absolute20_from_canonical(xyz, six, canonical)
            back = canonical_from_absolute20(labels, xyz, six)
            effective = canonical.copy()
            effective[:, :3] = np.clip(xyz+np.clip(canonical[:, :3], -.05, .05),
                (BOUNDS[0,0]-.12, BOUNDS[1,0]-.05, .10),
                (BOUNDS[0,1]+.12, BOUNDS[1,1]+.05, .72))-xyz
            effective[:, 3:6] = np.clip(effective[:, 3:6], -.25, .25)
            np.testing.assert_allclose(back, effective, rtol=0, atol=2e-6)
            for joints, pos, rot in zip(q, xyz, six, strict=True):
                for joint, value in zip(robot.arm_joint_ids, joints, strict=True):
                    p.resetJointState(body, joint, float(value), physicsClientId=client)
                s = p.getLinkState(body, robot.mount_link, computeForwardKinematics=True, physicsClientId=client)
                measured = np.asarray(p.getMatrixFromQuaternion(s[5])).reshape(3,3)[:, :2].T.reshape(6)
                error = max(float(np.max(abs(np.asarray(s[4])-pos))), float(np.max(abs(measured-rot))))
                if error > 1e-5:
                    raise ValueError('Measured URDF FK differs')
                errors.append(error)
            cache = out/f'near_miss_{i:03d}.npz'
            adapter = canonical.copy(); adapter[:,6] = (1-canonical[:,6])/2
            np.savez_compressed(cache, scene64=raw, canonical7=canonical,
                native8=native8(xyz, six, raw[:,27]), state20=pack_native20(xyz, six, closure),
                action20=labels, adapter_action7=adapter)
            index.append({'cache': str(cache), 'cache_sha256': sha256_file(cache),
                'source': str(path), 'source_sha256': row['payload_sha256'], 'frames': len(raw),
                'task': row['task']['name'], 'instruction': row['task']['prompt'],
                'source_kind': 'near_miss_corrective', 'pick_name': row['task']['pick_name'],
                'recovery_kind': row['kind'], 'visibility': visibility,
                'scene_group': plan['source_scene_groups'][plan['source_files'].index(row['source_file'])],
                'intervention_actions_excluded_from_training_labels': True})
            for k, mask in regimes(raw, canonical).items(): totals[k] += int(mask.sum())
    finally:
        p.disconnect(client)
    atomic_json(out/'index.json', index)
    atomic_json(out/'ADMISSION.json', {
        'passed': True, 'split': 'train', 'candidate': str(candidate),
        'all_attempts': len(rows), 'qualified_episodes': len(index), 'frames': sum(r['frames'] for r in index),
        'minimum_targets_per_task_kind': a.minimum_targets,
        'actual_targets_per_task_kind': {str(k): sorted(v) for k,v in coverage.items()},
        'rejected_evidence_retained': [r for r in rows if r not in accepted],
        'maximum_actual_URDF_FK_component_error': max(errors),
        'native_roundtrip_passed': True, 'RGB_and_state_replayed_before_admission': True,
        'pre_action_no_future_labels': True, 'regime_frames': dict(totals),
        'full_original_bank': str(BANK), 'original_index_sha256': previous['index_sha256'],
        'stats_sha256_preserved': previous['stats_sha256'],
        'new_index_sha256': sha256_file(out/'index.json'),
        'binding': {str(Path(__file__).resolve()): sha256_file(Path(__file__)),
            str(Path(collect.__file__).resolve()): sha256_file(Path(collect.__file__)),
            str(candidate/'plan.json'): sha256_file(candidate/'plan.json'),
            str(candidate/'index.json'): sha256_file(candidate/'index.json')},
        'training_or_VLA_success_NOT_claimed': True})
    print(json.dumps({'admitted_expert_episodes': len(index), 'frames': sum(r['frames'] for r in index),
        'FK_max': max(errors), 'regimes': dict(totals)}), flush=True)


if __name__ == '__main__':
    p0 = argparse.ArgumentParser(description=__doc__)
    p0.add_argument('--candidate', type=Path, default=OUT/'near_miss_sweep')
    p0.add_argument('--output', type=Path, default=OUT/'qualified_recovery')
    p0.add_argument('--minimum-targets', type=int, default=10)
    main(p0.parse_args())
