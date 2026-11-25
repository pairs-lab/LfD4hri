"""Bounded read-only diagnostics for the two active native VLA evaluations.

Never loads a GPU model, sends an action, changes scoring, rerolls a scene or
resumes a cancelled queue. Episode symptoms overlap and are NOT root causes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('touch', 'pick', 'move', 'pick_place')


def diagnose_trace(trace, task):
    if not trace:
        return {'controls': 0, 'symptoms': []}
    raw = np.asarray([r['pre_action_scene64'] for r in trace])
    actions = np.asarray([r['action'] for r in trace])
    before = np.asarray([r['before']['correct_object_held'] for r in trace])
    after = np.asarray([r['after']['correct_object_held'] for r in trace])
    distance = np.linalg.norm(raw[:, 11:14], axis=1)
    near = distance < .08
    # Never classify motion toward a goal while holding as wrong-target reach.
    reach = (raw[:, 31] < .5) & (np.linalg.norm(raw[:, 11:13], axis=1) > .03)
    substantial_xy = np.linalg.norm(actions[:, :2], axis=1) > .003
    moving_away = reach & substantial_xy & (np.sum(raw[:, 11:13]*actions[:, :2], axis=1) < 0)
    empty_lift = (near & (actions[:, 6] > .5) & (actions[:, 2] > .01)
                  & ~before & ~after)
    empty_measured = (raw[:, 6] > .95) & (raw[:, 31] < .5)
    released_before_goal = before & (actions[:, 6] < -.5) & (np.linalg.norm(raw[:, 14:16], axis=1) > .08)
    symptoms = []
    if task != 'touch' and not np.any(before | after):
        symptoms.append('never_attached_correct_target')
    if task != 'touch' and not np.any(raw[:, 17] > .5):
        symptoms.append('no_pre_action_verified_bilateral_hold')
    if moving_away.any(): symptoms.append('unheld_reach_command_points_away_from_target')
    if empty_lift.any(): symptoms.append('near_target_close_and_lift_over1cm_without_attachment')
    if (before & ~after).any(): symptoms.append('lost_or_released_correct_hold')
    if released_before_goal.any(): symptoms.append('opened_correct_hold_outside_goal_xy8cm')
    if (raw[:, 30] > .5).any(): symptoms.append('wrong_body_attachment_seen')
    safety = [r['safety_violation'] for r in trace if r.get('safety_violation')]
    if safety: symptoms.append('existing_safety_guard_aborted')
    return {'controls': len(trace), 'symptoms': symptoms,
        'min_pad_to_target_center_m': float(distance.min()),
        'terminal_goal_object_xy_m': float(np.linalg.norm(np.asarray(trace[-1]['after']['goal_xyz_m'])[:2]
                                        -np.asarray(trace[-1]['after']['object_base_xyz_m'])[:2])),
        'correct_attachment_before_or_after_steps': int((before | after).sum()),
        'pre_action_bilateral_hold_steps': int((raw[:, 17] > .5).sum()),
        'unheld_reach_away_steps': int(moving_away.sum()),
        'near_empty_close_lift_over1cm_steps': int(empty_lift.sum()),
        'measured95pct_empty_closed_steps': int(empty_measured.sum()),
        'measured95pct_empty_closed_OPEN_labels': int((empty_measured & (actions[:, 6] < -.5)).sum()),
        'safety': safety}


def snapshot(family):
    model = ROOT / 'VLA' / family / 'privileged_r43_native_v3_20261016/test1'
    report = {'family': family, 'actual_completed_episode_SR': None,
        'policy_receives_privileged67': True,
        'legacy_interaction_helper_privileged_policy_input_false_annotation_is_stale': True,
        'symptoms_overlap_and_are_NOT_proven_root_causes': True}
    train_status = model / 'train/status.json'
    if train_status.exists(): report['train'] = json.loads(train_status.read_text())
    progress = model / 'development40/progress.json'
    if not progress.exists():
        report['evaluation_status'] = 'not_started'
        return report
    state = json.loads(progress.read_text())
    report['evaluation_status'] = state['status']
    report['completed_trials'] = state['completed_trials']
    report['planned_trials'] = state['planned_trials']
    report['integration_error'] = state.get('error')
    rows, counts = [], Counter()
    for row in state['rows']:
        trace = json.loads(Path(row['trajectory']).read_text())[row['task']]
        diagnostic = diagnose_trace(trace, row['task'])
        diagnostic.update(seed=row['seed'], task=row['task'], success=row['success'], trajectory=row['trajectory'])
        rows.append(diagnostic)
        if not row['success']: counts.update(diagnostic['symptoms'])
    report['rows'] = rows
    report['failure_symptoms_overlapping'] = dict(counts)
    n = len(rows)
    report['actual_completed_episode_SR'] = None if n == 0 else sum(r['success'] for r in rows)/n
    report['by_action'] = {task: {'successes': sum(r['success'] for r in rows if r['task'] == task),
        'trials': sum(r['task'] == task for r in rows)} for task in TASKS}
    report['partial_cohort_NOT_final_rate'] = n < 40
    return report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    while True:
        reports = [snapshot(family) for family in ('xvla', 'vla_adapter')]
        state = {'pid': os.getpid(), 'status': 'monitoring_actual_native_pipeline',
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'reports': reports,
            'GPU_models_loaded': False, 'policy_or_scoring_modified': False,
            'additional_training_launched': False, 'reserved_test_scenes_used': False}
        terminal = lambda r: r['evaluation_status'].startswith(('completed40', 'failed')) or r.get('train', {}).get('status', '').startswith('failed')
        if all(terminal(r) for r in reports): state['status'] = 'completed_both_terminal'
        if time.monotonic()-started > args.hours*3600: state['status'] = 'bounded_monitor_deadline_reached'
        atomic_json(args.output / 'status.json', state)
        if state['status'] != 'monitoring_actual_native_pipeline': break
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.output.exists() or args.output.with_name(args.output.name+'_owner.json').exists():
            raise FileExistsError('Do not duplicate a watcher')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-m', 'VLA.watch_new_vla_diagnostics_r44',
                '--output', str(args.output.resolve()), '--hours', str(args.hours)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': process.pid})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}))
    else:
        run(args)
