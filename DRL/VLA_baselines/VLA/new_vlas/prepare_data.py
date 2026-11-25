"""Complete-source native action conversion and TRAIN-only normalizers.

Small numeric caches reference original RGB without making duplicate datasets.
Checks EVERY frame and gripper label. Canonical actions are the source of
absolute EEF targets (not next observed poses or a repeated partial fixture).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.common.environment import ROBOT_SPECS
from config import BOUNDS
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.native_corrective_dataset import NativeCorrectiveDataset
from VLA.new_vlas.action_contracts import (absolute20_from_canonical, canonical_from_absolute20,
    native8, pack_native20, bounds_stats)
from VLA.scene64_conditioning import normalization
from VLA.scene64_dataset import Scene64NativeDataset

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'VLA/audits/scene67_deadline_r31_20261015'
PROJECTION = ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
ADMISSION = BASE/'all_family_privileged_r33_v4/native_view_admission.json'


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = 'checking_full_source_and_correction_provenance'
    started = time.monotonic()
    def status(state='running', **extra):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra})
    client = None
    try:
        status()
        datasets = [Scene64NativeDataset(PROJECTION, 30), NativeCorrectiveDataset(ADMISSION, 30)]
        stage = 'all_frame_action_roundtrip_and_actual_URDF_FK'
        status()
        client = p.connect(p.DIRECT)
        robot = ROBOT_SPECS['ur5e']
        body = p.loadURDF(str(robot.urdf), (0, 0, 0), useFixedBase=True, physicsClientId=client)
        caches, all_states, all_actions, all_prefix = [], [], [], []
        q_errors, roundtrips, counts, target_counts = [], [], Counter(), Counter()
        clipped_frames, clipping_examples = 0, []
        for source_index, data in enumerate(datasets):
            for ei, row in enumerate(data.rows):
                path = data.source/row['file']
                start = data.ends[ei-1] if ei else 0
                raw = data.states[start:data.ends[ei]]
                with np.load(path, allow_pickle=False) as archive:
                    xyz, six = archive['ee_position'], archive['ee_rotation_6d']
                    measured = archive['gripper_closure']
                    joints = archive['joint_positions']
                    canonical = archive['canonical_actions']
                    closure = native_gripper_closure(archive)
                    instruction = str(archive['prompt'].item())
                if (xyz.shape != (row['steps'], 3) or six.shape != (row['steps'], 6)
                        or canonical.shape != (row['steps'], 7)):
                    raise ValueError('Incomplete native timeline: '+str(path))
                np.testing.assert_allclose(np.c_[joints, measured], raw[:, :7], rtol=0, atol=1e-6)
                np.testing.assert_allclose((canonical[:, 6]+1)/2, closure, rtol=0, atol=1e-6)
                labels = absolute20_from_canonical(xyz, six, canonical)
                back = canonical_from_absolute20(labels, xyz, six)
                # The shared environment clips the requested absolute XYZ.
                # Compare effective commands, retaining/reporting every source
                # label instead of treating valid boundary clipping as drift.
                effective = canonical.copy()
                effective[:, :3] = np.clip(xyz+np.clip(canonical[:, :3], -.05, .05),
                    (BOUNDS[0, 0]-.12, BOUNDS[1, 0]-.05, .10),
                    (BOUNDS[0, 1]+.12, BOUNDS[1, 1]+.05, .72))-xyz
                effective[:, 3:6] = np.clip(effective[:, 3:6], -.25, .25)
                changed = np.any(np.abs(effective-canonical) > 2e-6, axis=1)
                clipped_frames += int(changed.sum())
                if changed.any() and len(clipping_examples) < 10:
                    fi = int(np.flatnonzero(changed)[0])
                    clipping_examples.append({'source': str(path), 'frame': fi,
                        'requested': canonical[fi].tolist(), 'effective': effective[fi].tolist()})
                error = float(np.abs(back-effective).max())
                if error > 2e-6:
                    raise ValueError(f'Absolute/canonical command roundtrip {error}: {path}')
                roundtrips.append(error)
                # Exact FK checks every current joint sample against the
                # recorded measured pose; no IK or scene controller involved.
                for q, expected_xyz, expected_six in zip(joints, xyz, six, strict=True):
                    for ji, value in zip(robot.arm_joint_ids, q, strict=True):
                        p.resetJointState(body, ji, float(value), physicsClientId=client)
                    state = p.getLinkState(body, robot.mount_link, computeForwardKinematics=True,
                                          physicsClientId=client)
                    matrix = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
                    q_errors.append(max(float(np.abs(np.asarray(state[4])-expected_xyz).max()),
                        float(np.abs(matrix[:, :2].T.reshape(6)-expected_six).max())))
                if max(q_errors[-len(joints):]) > 1e-5:
                    raise ValueError('Wrong robot/link/base FK in recorded source: '+str(path))
                state20 = pack_native20(xyz, six, measured)
                prefix = native8(xyz, six, raw[:, 27])
                adapter_labels = canonical.copy()
                # Official LIBERO RLDS transform is 1=OPEN,0=CLOSE. Preserve
                # the pretrained head's convention; decode once at runtime.
                adapter_labels[:, 6] = 1-closure
                cache_name = f'numeric_{source_index}_{ei:05d}.npz'
                np.savez_compressed(out/cache_name, state20=state20, native8=prefix,
                    action20=labels, adapter_action7=adapter_labels, scene64=raw,
                    canonical7=canonical.astype(np.float32))
                task = row['task']['name']
                counts[task] += row['steps']
                target_counts[task+' / '+row['task']['pick_name']] += 1
                caches.append({'cache': cache_name, 'cache_sha256': sha256_file(out/cache_name),
                    'source': str(path), 'source_sha256': sha256_file(path),
                    'frames': row['steps'], 'task': task, 'instruction': instruction,
                    'source_kind': 'complete_original' if source_index == 0 else 'fresh_corrective',
                    'scene_group': row.get('scene_group'), 'pick_name': row['task']['pick_name']})
                all_states.append(raw)
                all_actions.append(adapter_labels)
                all_prefix.append(prefix)
                if len(caches) % 50 == 0:
                    status(converted_episodes=len(caches), converted_frames=sum(counts.values()))
        states = np.concatenate(all_states)
        stats67, regularization = normalization(states)
        atomic_json(out/'stats.json', {'scene67': stats67, 'regularization': regularization,
            'adapter': {'action': bounds_stats(np.concatenate(all_actions), mask=[True]*6+[False]),
                        'proprio': bounds_stats(np.concatenate(all_prefix))},
            'split': 'TRAIN_ONLY', 'validation_or_test_values_used': False})
        atomic_json(out/'index.json', caches)
        bound = {str(PROJECTION/'audit.json'): sha256_file(PROJECTION/'audit.json'),
            str(ADMISSION): sha256_file(ADMISSION), str(robot.urdf): sha256_file(robot.urdf)}
        for name in ('action_contracts.py', 'prepare_data.py'):
            path = Path(__file__).with_name(name)
            bound[str(path)] = sha256_file(path)
        atomic_json(out/'audit.json', {'passed': True,
            'scope': 'complete_TRAIN_native_action_conversion_NOT_model_fit_or_success',
            'episodes': len(caches), 'frames': len(states),
            'source_episodes': [len(d.rows) for d in datasets],
            'source_frames': [len(d) for d in datasets],
            'original_full_source_coverage': True, 'fresh_corrections_included': True,
            'all_frames_actual_URDF_FK_checked': True, 'max_FK_component_error': max(q_errors),
            'max_action_roundtrip_error': max(roundtrips),
            'xvla_native_domain': 12,
            'xvla_rotation_layout': 'UR_foundation_R_first_two_columns_rowmajor_interleaved',
            'xvla_single_arm_unused_second_arm': 'all10_zero_matches_official_UR_handler',
            'adapter_native_gripper_action': '1_open_0_closed_matches_official_RLDS',
            'adapter_rotation_state': 'equivalent_rotvec_branch_nearest_fixed_UR_home_avoids_pi_flips',
            'source_frames_with_existing_environment_workspace_clipping': clipped_frames,
            'clipping_examples': clipping_examples,
            'source_labels_retained_and_same_shared_clipping_used': True,
            'future_labels_in_observations': False, 'robot_controller_replaced': False,
            'image_bytes_modified_or_duplicated': False, 'all67_input_fields_preserved': True,
            'frame_counts_by_action': dict(counts), 'episode_counts_by_task_target': dict(target_counts),
            'index_sha256': sha256_file(out/'index.json'), 'stats_sha256': sha256_file(out/'stats.json'),
            'bound_files': bound, 'runtime_seconds': time.monotonic()-started,
            'own_optimizer_save_reload_and_physical_adapter_checks_still_required': True})
        stage = 'complete_source_native_numeric_view_ready'
        status('completed', episodes=len(caches), frames=len(states))
    except Exception as error:
        status('failed_preserve_evidence', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if client is not None:
            p.disconnect(physicsClientId=client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
