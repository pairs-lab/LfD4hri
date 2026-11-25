"""Fresh all-TRAIN support and actual development rollout audit, CPU only.

No model weights/actions/data/statistics are changed. Distinguish supplied
feature slots from actual unsaturated information and recovery label coverage.
Never read the cancelled800 reserved cohort or feed development into training.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import NUMERIC_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts'
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016'
PROGRESS = [
    BASE/'xvla7000_native_chunk8/progress.json',
    ROOT/'VLA/audits/scene67_deadline_r31_20261015/Smol_native_chunk2_conditional120_r41/progress.json',
]


def numeric_batch(raw):
    raw = np.asarray(raw, np.float32)
    angles = np.stack((np.sin(raw[:, 35:38]), np.cos(raw[:, 35:38])), -1).reshape(-1, 6)
    return np.c_[raw[:, :35], angles, raw[:, 38:]].astype(np.float32)


def support(raw, actions):
    held = raw[:, 17] > .5
    attached = raw[:, 31] > .5
    closed_empty = (raw[:, 6] > .8) & ~attached
    unheld_up = ~attached & (actions[:, 6] > .5) & (actions[:, 2] > .005)
    near = np.linalg.norm(raw[:, 11:13], axis=1) < .02
    return {
        'frames': len(raw), 'correct_bilateral_held': int(held.sum()),
        'wrong_attached': int((raw[:, 30] > .5).sum()),
        'closed_empty': int(closed_empty.sum()),
        'closed_empty_with_reopen_label': int((closed_empty & (actions[:, 6] < -.5)).sum()),
        'unheld_close_with_upward_command_NOT_automatically_invalid': int(unheld_up.sum()),
        'bilateral_lost_between_causal_before_frames': int((held[:-1] & ~held[1:]).sum()),
        'any_attachment_lost_between_causal_before_frames': int((attached[:-1] & ~attached[1:]).sum()),
        'near_target_unheld_closing_labels': int((near & ~held & (actions[:, 6] > .5)).sum()),
        'nonzero_roll_pitch_action_labels': int((np.max(np.abs(actions[:, 3:5]), axis=1) > 1e-5).sum()),
        'tool_tilt_over3degrees': int((np.arccos(np.clip(-np.cross(raw[:, 18:21], raw[:, 21:24])[:, 2], -1, 1)) > np.deg2rad(3)).sum()),
    }


def saturation(raw, stats):
    z = (numeric_batch(raw)-np.asarray(stats['mean'], np.float32))/(np.asarray(stats['std'], np.float32)+1e-8)
    clipped = np.abs(z) > 5
    selected = np.flatnonzero(clipped.any(0))
    return {'any_field_saturated_frames': int(clipped.any(1).sum()), 'frames': len(raw),
        'features': [{'index': int(i), 'name': NUMERIC_NAMES[i], 'frames': int(clipped[:, i].sum()),
            'fraction': float(clipped[:, i].mean()), 'max_abs_before_clip': float(np.abs(z[:, i]).max())}
            for i in sorted(selected, key=lambda i: -clipped[:, i].sum())]}


def analyze_rollout(row, trace):
    raw = np.asarray([f['pre_action_scene64'] for f in trace], np.float32)
    actions = np.asarray([f['action'] for f in trace], np.float32)
    close = np.flatnonzero(actions[:, 6] > .2)
    first = int(close[0]) if len(close) else None
    info = {'seed': row['seed'], 'task': row['task'], 'steps': len(trace),
        'success': row['success'], 'safety_violation': row.get('safety_violation'),
        'observed_correct_attachment': any(f.get('before', {}).get('correct_object_held') or
            f.get('after', {}).get('correct_object_held') for f in trace),
        'first_close_control0': first,
        'first_close_XY_target_center_error_m': float(np.linalg.norm(raw[first, 11:13])) if first is not None else None,
        'first_close_pad_height_fraction': float(raw[first, 38]) if first is not None else None,
        'first_close_requested_target_Z_relative_to_pad_m': float(raw[first, 13]) if first is not None else None,
        'unheld_lift_controls': int(((raw[:, 17] < .5) & (actions[:, 2] > .005) & (actions[:, 6] > .2)).sum()),
        'queued_action_age_steps_max': max((f['step']-1-f.get('chunk_query_step', f['step']-1) for f in trace), default=0)}
    if row['success']:
        category = 'predeclared_region_success'
    elif row.get('safety_violation'):
        category = 'native_safety_abort'
    elif row['task'] == 'touch':
        category = 'touch_target_not_reached_before_timeout_CLOSE_not_required'
    elif not info['observed_correct_attachment'] and first is None:
        category = 'never_close_before_timeout'
    elif not info['observed_correct_attachment']:
        category = 'close_or_move_but_never_verified_attachment'
    else:
        category = 'post_attachment_not_at_goal_before_timeout'
    info['observed_category_NOT_model_intrinsic_cause'] = category
    return info, raw


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((DATA/'index.json').read_text())
    stats = json.loads((DATA/'stats.json').read_text())['scene67']
    hashes = {str(DATA/name): sha256_file(DATA/name) for name in ('index.json', 'stats.json', 'audit.json')}
    totals = defaultdict(Counter)
    all_raw = []
    tilt_anchors = []
    for ei, row in enumerate(index):
        path = DATA/row['cache']
        if sha256_file(path) != row['cache_sha256']:
            raise ValueError('Immutable TRAIN cache changed')
        with np.load(path, allow_pickle=False) as a:
            raw, actions = a['scene64'], a['canonical7']
        measured = support(raw, actions)
        totals[row['task']].update(measured)
        all_raw.append(raw)
        if measured['nonzero_roll_pitch_action_labels']:
            tilt_anchors.append({'cache': row['cache'], 'source': row['source'], 'count': measured['nonzero_roll_pitch_action_labels']})
    raw = np.concatenate(all_raw)
    train_report = {'episodes': len(index), 'frames': len(raw), 'by_task': dict(totals),
        'bounded67_saturation': saturation(raw, stats), 'nonzero_tilt_label_episodes': tilt_anchors,
        'unique_scene_groups': len({r['scene_group'] for r in index}),
        'physical_roll_pitch_ground_truth_restoration_NOT_guaranteed_by_yaw_only_teacher': True}
    atomic_json(args.output/'TRAIN_support.json', train_report)
    reports = []
    for path in PROGRESS:
        p = json.loads(path.read_text())
        rows, matrices, trace_hashes = [], [], {}
        cache = {}
        for row in p['rows']:
            trajectory = Path(row['trajectory'])
            if trajectory not in cache:
                cache[trajectory] = json.loads(trajectory.read_text())
                trace_hashes[str(trajectory)] = sha256_file(trajectory)
            result, values = analyze_rollout(row, cache[trajectory][row['task']])
            rows.append(result); matrices.append(values)
        failed = [r for r in rows if not r['success']]
        report = {'progress': str(path), 'trials': len(rows), 'successes': sum(r['success'] for r in rows),
            'failure_categories': dict(Counter(r['observed_category_NOT_model_intrinsic_cause'] for r in failed)),
            'failures_never_correctly_attached': sum(not r['observed_correct_attachment'] for r in failed),
            'bounded67_saturation_IF_applied': saturation(np.concatenate(matrices), stats),
            'rows': rows, 'progress_sha256': sha256_file(path), 'trajectory_sha256': trace_hashes}
        atomic_json(args.output/(path.parent.name+'_rollout_audit.json'), report)
        reports.append({k: report[k] for k in ('progress', 'trials', 'successes', 'failure_categories', 'failures_never_correctly_attached')})
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError('TRAIN source metadata changed')
    atomic_json(args.output/'audit.json', {'scope': 'CPU_readonly_actual_TRAIN_and_development_feedback_support_NOT_new_SR',
        'TRAIN': {'episodes': len(index), 'frames': len(raw), 'by_task': dict(totals)},
        'development': reports, 'source_hashes': hashes, 'weights_or_inputs_changed': False,
        'cancelled800_or_reserved_test_used': False, 'readiness_for_new_finetune_claimed': False})
    print(json.dumps({'TRAIN': dict(totals), 'development': reports}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
