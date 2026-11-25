"""Read-only trajectory analysis; never changes RDT actions or task scores."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from VLA.common.artifact_io import atomic_json


def analyze(root):
    progress = json.loads((root / 'development40/progress.json').read_text())
    rows = []
    for result in progress['rows']:
        trace = json.loads(Path(result['trajectory']).read_text())[result['task']]
        states = np.asarray([s['pre_action_scene64'] for s in trace], dtype=np.float64)
        commands = np.asarray([s['action'] for s in trace], dtype=np.float64)
        before = [s['before'] for s in trace]
        after = [s['after'] for s in trace]
        # Audit post-action contact too: terminal successful states might not
        # be present as the following policy input because the episode ends.
        held = np.asarray([b['correct_object_held'] for b in before + after], bool)
        pad_dist = np.linalg.norm(states[:, 11:14], axis=1)
        post_pad_dist = np.asarray([np.linalg.norm(np.asarray(s['object_center_xyz_m'])
            - np.asarray(s['pad_grasp_center_xyz_m'])) for s in after])
        goal_xy = np.asarray([np.linalg.norm(np.asarray(s['object_base_xyz_m'])[:2]
            - np.asarray(s['goal_xyz_m'])[:2]) for s in after])
        start_object = np.asarray(before[0]['object_base_xyz_m'])
        peak_rise = max(s['object_base_xyz_m'][2] - start_object[2] for s in after)
        flags = []
        if not result['success']:
            if result['safety_violation']:
                flags.append('safety_abort')
            if result['task'] != 'touch' and not np.any(held):
                flags.append('no_correct_attachment_observed')
            if min(pad_dist.min(), post_pad_dist.min()) > .04:
                flags.append('never_reached_target_center_within_4cm')
            if np.any((commands[:, 6] > .2) & (pad_dist > .08) & (states[:, 17] == 0)):
                flags.append('close_requested_while_over_8cm_from_center')
            if np.any(held) and result['task'] in ('move', 'pick_place'):
                flags.append('correct_hold_observed_but_goal_not_achieved')
        rows.append({**result, 'failure_signatures_overlap_NOT_causal_proof': flags,
            'all_steps_full64_input': states.shape == (len(trace), 64),
            'correct_attachment_observed_before_or_after': bool(held.any()),
            'minimum_pad_target_center_distance_m': float(min(pad_dist.min(), post_pad_dist.min())),
            'initial_pad_target_center_distance_m': float(pad_dist[0]),
            'final_pad_target_center_distance_m': float(post_pad_dist[-1]),
            'minimum_after_object_goal_XY_distance_m': (float(goal_xy.min())
                if result['task'] in ('move', 'pick_place') else None),
            'peak_object_base_rise_m': float(peak_rise),
            'close_request_fraction': float(np.mean(commands[:, 6] > .2)),
            'initial_prompt': json.loads((Path(result['trajectory']).parent / 'observations'
                / result['task'] / 'step_0000.json').read_text())['prompt']})
    counts = {}
    for task in ('touch', 'pick', 'move', 'pick_place'):
        subset = [r for r in rows if r['task'] == task]
        counts[task] = {'completed': len(subset), 'planned': 10,
            'successes': sum(r['success'] for r in subset),
            'safety_aborts': sum(bool(r['safety_violation']) for r in subset)}
    summary = {'scope': 'actual_completed_RDT40_trajectory_diagnostics_NOT_model_ceiling',
        'completed': len(rows), 'planned': 40, 'status': progress['status'],
        'successes': sum(r['success'] for r in rows), 'by_action': counts,
        'failure_signatures_overlap_NOT_causal_proof': dict(Counter(flag for r in rows
            for flag in r['failure_signatures_overlap_NOT_causal_proof'])),
        'actual_TRAIN_probe': None, 'rows': rows,
        'simulator_controls_executed_by_analysis': 0, 'policy_or_scoring_changed': False}
    probe = root / 'actual_TRAIN_fit_probe/audit.json'
    if probe.exists():
        proof = json.loads(probe.read_text())
        summary['actual_TRAIN_probe'] = {k: proof[k] for k in ('status', 'completed_requests',
            'repeat_input_max_abs_difference', 'median_native_joint_MAE_rad', 'gripper_sign_correct')}
    atomic_json(root / 'live_failure_analysis.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2), flush=True)
    return summary


def watch(root):
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        result = analyze(root)
        state = json.loads((root / 'status.json').read_text())
        if state['status'] != 'running':
            atomic_json(root / 'analysis_watch_status.json', {'pid': os.getpid(),
                'status': 'completed' if result['completed'] == 40 else 'preserved_partial',
                'completed': result['completed'], 'policy_or_scores_changed': False})
            return
        atomic_json(root / 'analysis_watch_status.json', {'pid': os.getpid(),
            'status': 'watching_actual_trials', 'completed': result['completed'],
            'policy_or_scores_changed': False})
        time.sleep(30)
    atomic_json(root / 'analysis_watch_status.json', {'pid': os.getpid(),
        'status': 'finite_12h_watch_expired', 'policy_or_scores_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    root = args.run.resolve()
    if args.launch:
        marker = root / 'analysis_watch_launch.json'
        if marker.exists():
            raise FileExistsError('No duplicate analyzer')
        with (root / 'analysis_watch.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'VLA.analyze_rdt40_scene67_r42',
                '--run', str(root), '--watch'], cwd=Path(__file__).resolve().parents[1],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        atomic_json(marker, {'pid': child.pid, 'finite_watch_hours': 12,
            'simulator_controls_executed': 0, 'GPU_model_loaded': False})
        print(json.dumps({'analyzer_pid': child.pid}))
    elif args.watch:
        watch(root)
    else:
        analyze(root)
