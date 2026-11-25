"""Read-only symptom analysis of saved REAL neural trajectories.

Diagnostic thresholds do not change success, actions, dataset or checkpoints.
Overlapping symptoms are not asserted causal explanations/model weaknesses.
"""
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def inspect_episode(row, trace):
    counts = Counter()
    correct = wrong = previously_correct = False
    minimum_xy = math.inf
    minimum_xyz = math.inf
    close_near_object_seen = False
    for frame in trace:
        state = frame.get('pre_action_scene64')
        if not isinstance(state, list) or len(state) != 64:
            counts['missing_full64_frame'] += 1
            continue
        action = frame['action']
        after = frame.get('after', {})
        held = bool(state[17]) or bool(after.get('correct_object_held'))
        correct = correct or held
        wrong = wrong or bool(state[30]) or bool(after.get('held_object') is not None and not after.get('correct_object_held'))
        if previously_correct and not held:
            counts['correct_hold_lost_transitions'] += 1
        previously_correct = held
        xy = math.hypot(state[11], state[12])
        xyz = math.sqrt(sum(v*v for v in state[11:14]))
        minimum_xy = min(minimum_xy, xy)
        minimum_xyz = min(minimum_xyz, xyz)
        radius = .5 * math.hypot(state[24], state[25])
        # Current tool height ABOVE the current target AABB top, not floor Z.
        above_top = -state[13] - .5 * state[26]
        if action[2] < -.005 and above_top < .08 and xy > radius + .025:
            counts['descending_low_outside_target_xy_envelope'] += 1
        if action[6] > .5 and above_top < .06 and xy < radius + .04:
            close_near_object_seen = True
        if close_near_object_seen and not held and action[2] > .01:
            counts['upward_command_after_near_object_close_without_verified_hold'] += 1
        # Use BEFORE-command hold, not after release, or the very opening
        # action that drops a body would hide its own diagnostic evidence.
        if (row['task'] in ('move', 'pick_place') and state[17]
                and action[6] < -.5 and math.hypot(state[14], state[15]) > .08):
            counts['OPEN_while_correctly_held_outside_goal_xy8cm'] += 1
    result = {'seed': row['seed'], 'task': row['task'], 'success': bool(row['success']),
        'safety_violation': row.get('safety_violation'), 'saved_controls': len(trace),
        'ever_correct_hold': correct, 'ever_wrong_hold': wrong,
        'minimum_pad_target_xy_m': minimum_xy if math.isfinite(minimum_xy) else None,
        'minimum_pad_target_xyz_m': minimum_xyz if math.isfinite(minimum_xyz) else None,
        'overlapping_diagnostic_counts_NOT_causal_findings': dict(counts)}
    if not trace:
        result['trace_missing_or_empty'] = True
    elif isinstance(trace[-1].get('pre_action_scene64'), list) and len(trace[-1]['pre_action_scene64']) == 64:
        state, action = trace[-1]['pre_action_scene64'], trace[-1]['action']
        result['terminal_before_last_command_NOT_success_redefinition'] = {
            'target_xyz_distance_m': math.sqrt(sum(v*v for v in state[11:14])),
            'target_xy_distance_m': math.hypot(state[11], state[12]),
            'goal_xy_distance_m': math.hypot(state[14], state[15]),
            'correct_bilateral_hold': bool(state[17]), 'action': action}
        if (row['task'] == 'touch' and not row['success'] and not row.get('safety_violation')
                and result['terminal_before_last_command_NOT_success_redefinition']['target_xyz_distance_m'] < .12
                and action[2] < -.001):
            counts['touch_terminal_within12cm_still_descending_NOT_predicted_extra_success'] += 1
            result['overlapping_diagnostic_counts_NOT_causal_findings'] = dict(counts)
    return result


def analyze(progress_path, output):
    progress_path, output = Path(progress_path), Path(output)
    if not progress_path.exists():
        return
    progress = json.loads(progress_path.read_text())
    rows = progress.get('rows', [])
    if not rows:
        return
    identity = [(r['seed'], r['task'], bool(r['success']), r.get('steps')) for r in rows]
    implementation_hash = sha256_file(Path(__file__))
    if output.exists():
        previous = json.loads(output.read_text())
        if (previous.get('terminal_rows_identity') == [list(v) for v in identity]
                and previous.get('implementation_sha256') == implementation_hash):
            return
    cases, hashes, cache = [], {}, {}
    for row in rows:
        trajectory = Path(row['trajectory'])
        if trajectory not in cache:
            raw=trajectory.read_bytes()
            cache[trajectory] = json.loads(raw)
            hashes[str(trajectory)] = hashlib.sha256(raw).hexdigest()
        cases.append(inspect_episode(row, cache[trajectory].get(row['task'], [])))
    failed = [c for c in cases if not c['success']]
    symptoms = Counter()
    for case in failed:
        if case['safety_violation']:
            symptoms['physical_safety_abort'] += 1
        if case['task'] != 'touch' and not case['ever_correct_hold']:
            symptoms['never_verified_correct_hold'] += 1
        if case['ever_wrong_hold']:
            symptoms['ever_wrong_object_held'] += 1
        for key, count in case['overlapping_diagnostic_counts_NOT_causal_findings'].items():
            if count:
                symptoms[key] += 1
    atomic_json(output, {'scope': 'actual_neural_trace_symptoms_NOT_proven_model_or_setup_causes',
        'source_progress': str(progress_path.resolve()), 'completed_trials': len(rows),
        'planned_trials': progress.get('planned_trials'), 'progress_status': progress.get('status'),
        'successes': sum(c['success'] for c in cases), 'failed_completed_trials': len(failed),
        'terminal_rows_identity': identity, 'overlapping_failure_episode_counts': dict(symptoms),
        'cases': cases, 'trajectory_sha256': hashes, 'progress_sha256': sha256_file(progress_path),
        'implementation_sha256': implementation_hash, 'scoring_or_physics_changed': False,
        'partial_cohort_is_NOT_final_SR': len(rows) != progress.get('planned_trials'),
        'automatic_remediation_claimed': False})


def audit_registered(base):
    base = Path(base)
    roots = [base/'openpi_native32_actual_fit_r32/pi0_development40',
        base/'scene67_resume_r33_v2/development120']
    roots.extend(base/'scene67_resume_parallel_r34'/attempt/'development120' for attempt in ('parallel_attempt','exclusive_attempt'))
    roots.extend(base/'all_family_privileged_r33_v4'/(model+'_development120') for model in ('openvla', 'rdt', 'cogact'))
    roots.extend(base/'openpi_EEF_repair_r33_v4'/(model+'_EEF_development120') for model in ('pi0', 'pi0_fast'))
    roots.append(base/'Smol_native20_r36')
    # Separate R37/R38 ablations keep their own symptoms and denominators.
    # A safety-shield row is never merged into the unshielded VLA baseline.
    roots.extend(base/name for name in ('Smol_contactstop_closedloop_r37','Smol_bounded_feedback40_r38'))
    for pointer,default in (('Smol_GTcrop_motion_r36_active.json','Smol_GTcrop_motion_r36_v3'),
                            ('Smol_native20_extension_r36_active.json','Smol_native20_extension_r36_v2')):
        path=base/pointer
        root=Path(json.loads(path.read_text())['cycle']) if path.exists() else base/default
        if root.parent!=base:raise ValueError('Unexpected R36 analysis root')
        roots.append(root/'development120' if 'GTcrop' in root.name else root)
    for root in roots:
        analyze(root/'progress.json', root/'failure_symptoms_r33.json')


if __name__ == '__main__':
    audit_registered(Path(__file__).resolve().parent/'audits/scene67_deadline_r31_20261015')
