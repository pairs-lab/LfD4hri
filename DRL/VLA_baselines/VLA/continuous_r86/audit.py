"""Read-only closed-loop evidence, not inferred cognitive/model diagnoses.

Usable while an existing owner is running. Does not call its stateful policy
server, edit bound source, change thresholds or supply privileged actions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

TASKS = ('touch', 'pick', 'move', 'pick_place')


def read(p):
    return json.loads(Path(p).read_text())


def trace_metrics(trace):
    if not trace:
        raise ValueError('Empty episode is an integration error, not SR0')
    pre = np.asarray([r['pre_action_scene64'] for r in trace], np.float64)
    post = np.asarray([r['post_action_scene64'] for r in trace], np.float64)
    action = np.asarray([r['action'] for r in trace], np.float64)
    executed = np.asarray([r['executed_canonical_action'] for r in trace], np.float64)
    if pre.shape != (len(trace), 64) or action.shape != (len(trace), 7) or post.shape != pre.shape:
        raise ValueError('Malformed actual trace')
    if not all(np.isfinite(v).all() for v in (pre, post, action, executed)):
        raise ValueError('Nonfinite trace')
    rel = pre[:, 11:14]
    distance = np.linalg.norm(post[:, 11:14], axis=1)
    length = np.linalg.norm(rel, axis=1) * np.linalg.norm(action[:, :3], axis=1)
    valid = length > 1e-9
    alignment = np.divide(np.sum(rel * action[:, :3], axis=1), length,
                          out=np.zeros(len(trace)), where=valid)
    hold = post[:, 17] > .5
    contact = (post[:, 28:30] > .5).any(axis=1)
    safety = next((r['safety_violation'] for r in trace if r.get('safety_violation')), None)
    # This is a descriptive failure signature. A safety abort may be caused by
    # policy error, plant/controller error, or both; no causal attribution here.
    signature = ('success' if any(r['success'] for r in trace) else
                 'safety_abort' if safety else
                 'hold_lost_before_goal' if hold.any() and not hold[-1] else
                 'held_but_goal_not_reached' if hold.any() else
                 'contact_without_verified_hold' if contact.any() else
                 'no_verified_target_contact')
    return {
        'steps': len(trace), 'signature_NOT_causal_diagnosis': signature,
        'initial_target_distance_m': float(np.linalg.norm(rel[0])),
        'minimum_target_distance_m': float(distance.min()),
        'terminal_target_distance_m': float(distance[-1]),
        'first5_action_target_cosine': float(alignment[:5][valid[:5]].mean()) if valid[:5].any() else None,
        'action_target_cosine': float(alignment[valid].mean()) if valid.any() else None,
        'ever_target_contact': bool(contact.any()), 'ever_correct_hold': bool(hold.any()),
        'hold_then_loss_transitions': int(np.sum(hold[:-1] & ~hold[1:])),
        'close_requested_without_verified_hold_fraction': float(np.mean((action[:, 6] > 0) & (pre[:, 17] < .5))),
        'safety_violation': safety,
        'max_requested_executed_action_delta': float(abs(action-executed).max()),
        'pre_raw_min': pre.min(axis=0).tolist(), 'pre_raw_max': pre.max(axis=0).tolist(),
    }


def cohort_audit(progress_path, *, require_complete=False):
    p = Path(progress_path)
    progress = read(p)
    plan = read(p.parent/'PLAN.json')
    role = progress['cohort_role']
    # Final800 is measurement-only: do NOT expose final failure traces to the
    # adaptation/selection loop or tune from them.
    final = role == 'fresh_secondary_final800'
    if final and not plan['method']['frozen_after_confirmation']:
        raise ValueError('Final observations without frozen method')
    rows = progress['rows']
    keys = [(int(r['seed']), r['task']) for r in rows]
    expected = {(int(s), t) for s in plan['cohort']['seeds'] for t in TASKS}
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Duplicate or unexpected evaluation instance')
    complete = set(keys) == expected and progress['status'].startswith('completed_actual_')
    if require_complete and not complete:
        raise ValueError('A partial/integration-error cohort cannot pass promotion')
    if progress['completed_trials'] != len(rows) or progress['planned_trials'] != len(expected):
        raise ValueError('Actual coverage counter mismatch')
    summaries = {}
    all_details = []
    for task in TASKS:
        selected = [r for r in rows if r['task'] == task]
        count = len(selected)
        wins = sum(bool(r['success']) for r in selected)
        reported = progress['per_action'][task]
        if (reported['n'], reported['successes']) != (count, wins):
            raise ValueError('Reported outcomes disagree with actual rows')
        summary = {'n': count, 'successes': wins, 'SR_percent': 100*wins/count if count else None,
                   'safety_aborts': sum(bool(r.get('safety_violation')) for r in selected)}
        if not final:
            details = []
            for row in selected:
                trace = read(row['trajectory'])[task]
                m = trace_metrics(trace)
                if m['steps'] != row['steps'] or abs(m['minimum_target_distance_m']-row['minimum_target_distance_m']) > 1e-5:
                    raise ValueError('Episode trace/outcome mismatch')
                if bool(any(s['success'] for s in trace)) != bool(row['success']):
                    raise ValueError('Success ledger differs from controller receipt')
                details.append({'seed': row['seed'], 'task': task, **m})
            summary['descriptive_signatures_NOT_model_causes'] = dict(Counter(
                r['signature_NOT_causal_diagnosis'] for r in details))
            for k in ('initial_target_distance_m', 'minimum_target_distance_m', 'terminal_target_distance_m',
                      'action_target_cosine', 'close_requested_without_verified_hold_fraction'):
                values = [r[k] for r in details if r[k] is not None]
                summary['mean_'+k] = float(np.mean(values)) if values else None
            summary['episodes_with_correct_hold'] = sum(r['ever_correct_hold'] for r in details)
            all_details.extend(details)
        summaries[task] = summary
    return {'audit_passed': True, 'complete': complete, 'role': role,
            'source': str(p.resolve()), 'source_sha256_snapshot': sha256_file(p),
            'actual_trials': len(rows), 'expected_trials': len(expected), 'per_action': summaries,
            'DEV_trace_details': all_details,
            'final_traces_used_for_adaptation': False,
            'old_interaction_audit_privileged_policy_input_false_is_legacy_metadata': True,
            'actual_privileged_input_evidence': 'NEURAL_QUERIES receipts + payload/map digests; not that legacy flag',
            'causal_claims_not_justified': ['memory loss', 'intrinsic model weakness', 'camera caused every failure'],
            'metric': 'R62 semantic first-hit, NOT strict1cm; put release/settle not required'}


def training_audit(checkpoint, data):
    from VLA.replacement_r84.data import sample, normalize_actions, denormalize_actions
    cp = Path(checkpoint)
    meta, ref = read(cp/'TRAINING.json'), read(cp/'RELOAD_ANCHORS.json')
    if not meta['completed'] or len(ref['anchors']) != len(ref['predictions']):
        raise ValueError('Incomplete optimization or prediction evidence')
    rows = []
    for (ei, fi), pred in zip(ref['anchors'], ref['predictions'], strict=True):
        s = sample(data, ei, fi, meta['stats'])
        predicted = np.asarray(pred, np.float64)
        target = s['action'][:len(predicted)].copy()
        # Dita saved10 future canonical actions; VLANeXt8. Native conversion
        # must preserve tail convention; do not compare invalid padded labels.
        if len(predicted) > len(target):
            n = data.numeric(ei); ids = fi+np.arange(len(predicted))
            target = n['canonical7'][np.minimum(ids, data.rows[ei]['frames']-1)].copy()
            target[ids >= data.rows[ei]['frames'], :6] = 0
        else:
            target[s['action_is_pad'][:len(predicted)], :6] = 0
        if predicted.shape != target.shape or not np.isfinite(predicted).all():
            raise ValueError('Prediction/label shape or finite contract')
        decoded = denormalize_actions(normalize_actions(target, meta['stats']), meta['stats'])
        rows.append({'task': s['task'], 'episode': int(ei), 'frame': int(fi),
            'TRAIN_anchor_NOT_validation': True,
            'xyz_MAE_m': float(abs(predicted[:, :3]-target[:, :3]).mean()),
            'rotation_MAE_rad': float(abs(predicted[:, 3:6]-target[:, 3:6]).mean()),
            'gripper_sign_agreement': float(np.mean((predicted[:, 6]>0) == (target[:, 6]>0))),
            'action_roundtrip_clipping_error': float(abs(decoded-target).max()),
            'label0': target[0].tolist(), 'predicted0': predicted[0].tolist()})
    logs = cp/'updates.jsonl'
    updates = [json.loads(line) for line in logs.read_text().splitlines()] if logs.exists() else []
    return {'audit_passed': True, 'checkpoint': str(cp), 'updates': meta['updates'],
            'effective_batch': meta['effective_batch'], 'source_sha256': sha256_file(cp/'TRAINING.json'),
            'last_loss_mean': float(np.mean([r['loss'] for r in updates[-50:]])) if updates else None,
            'TRAIN_anchors': rows, 'NOT_heldout_validation_or_robot_SR': True,
            'loss_decrease_is_not_closed_loop_success': True,
            'full_native_recipe_budget_reproduced': meta.get('native20k128_recipe_reproduced', meta.get('native_10k32_budget_reached', False))}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--progress', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()
    atomic_json(a.output, cohort_audit(a.progress, require_complete=a.require_complete))
