"""Prioritize the existing JEPA development40; gate larger runs and audit real SR.

No bound training, model or environment code changes. Only the R65 coordinator
is held once its development40 child exists; inference/evaluation continue.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from VLA.cancel_fast_r66 import process, same_process, send, utc, module
from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
R65 = ROOT / 'VLA/audits/jepa_fast800_r65_20261018'
SOURCE = R65 / 'vla_jepa/development40'
OUT = ROOT / 'VLA/audits/jepa_quick40_20261019'
TASKS = ('touch', 'pick', 'move', 'pick_place')
LIMITS = (80, 120, 160, 200)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-half), min(1., center+half)]


def audit(out):
    progress = read(SOURCE / 'progress.json')
    plan = read(SOURCE / 'plan.json')
    cohort = read(plan['cohort'])
    rows = progress['rows']
    seeds = cohort['seeds']
    if (progress['status'] != 'completed_actual_cohort' or len(rows) != 40 or len(seeds) != 10
            or progress['planned_trials'] != 40 or progress['role'] != 'development_selection'):
        raise ValueError('Require the complete real 40-episode development cohort')
    lookup = {(r['seed'], r['task']): r for r in rows}
    if set(lookup) != {(s, t) for s in seeds for t in TASKS}:
        raise ValueError('Duplicate/missing task or seed')
    expected = R65 / 'vla_jepa/fit2500/step-002500'
    if Path(progress['checkpoint']).resolve() != expected.resolve():
        raise ValueError('Not the completed 2500-update JEPA checkpoint')
    reload = read(R65 / 'vla_jepa/fresh_reload_final.json')
    if not reload['passed'] or reload['actual_optimizer_updates'] != 2500:
        raise ValueError('Completed checkpoint reload proof absent')
    journal = [json.loads(line) for line in (SOURCE / 'NEURAL_QUERY_RECEIPTS.jsonl').read_text().splitlines()]
    qi = controls = 0
    grasp_diagnostics = {t: {'ever_correct_object_held': 0,
                             'successful_episodes_ending_still_held': 0} for t in TASKS}
    for seed in seeds:
        traces = read(SOURCE / f'test_seed{seed}/trajectory.json')
        for task in TASKS:
            row = lookup[seed, task]
            trace = traces[task]
            if not trace or len(trace) != row['steps'] or trace[-1]['success'] != row['success']:
                raise ValueError('Episode outcome differs from saved actual trajectory')
            grasp_diagnostics[task]['ever_correct_object_held'] += int(any(
                e.get('after', {}).get('correct_object_held', False) for e in trace))
            grasp_diagnostics[task]['successful_episodes_ending_still_held'] += int(
                row['success'] and trace[-1].get('after', {}).get('correct_object_held', False))
            groups = []
            for entry in trace:
                query = entry['neural_receipt']['query_step']
                if not groups or query != groups[-1][0]:
                    groups.append((query, []))
                groups[-1][1].append(entry)
            for _, chunk in groups:
                receipt = journal[qi]
                qi += 1
                native = receipt['actual_native_input_receipt']
                if (receipt['seed'] != seed or receipt['task'] != task or receipt['native_views'] != 3
                        or not receipt['GT_map_SHA256'] or len(native['actual_Qwen_image_grid_thw']) != 3
                        or native['actual_DiT_state_dimensions'] != 67):
                    raise ValueError('Actual native input or chronological query receipt mismatch')
                actions = np.asarray(receipt['actions'], dtype=np.float32)
                if actions.shape != (7, 7) or not np.isfinite(actions).all() or len(chunk) > plan['prefix']:
                    raise ValueError('Invalid native JEPA chunk/prefix')
                for i, entry in enumerate(chunk):
                    if (entry['neural_receipt']['input_sha256'] != receipt['input_sha256']
                            or not np.array_equal(np.asarray(entry['action'], np.float32), actions[i])):
                        raise ValueError('Executed command not from actual JEPA chunk')
                controls += len(chunk)
    if qi != len(journal) or qi != progress['native_queries'] or controls != progress['motor_controls']:
        raise ValueError('Neural query/motor-control counts differ')
    by_task = {}
    for task, limit in zip(TASKS, LIMITS):
        group = [r for r in rows if r['task'] == task]
        k = sum(bool(r['success']) for r in group)
        by_task[task] = {'episodes': len(group), 'successes': k, 'SR': k / len(group),
                         'wilson_95': wilson(k, len(group)),
                         'safety_aborts': sum(bool(r['safety_violation']) for r in group),
                         'failed_at_step_limit': sum(not r['success'] and not r['safety_violation']
                                                     and r['steps'] >= limit for r in group),
                         'posthoc_grasp_diagnostics_NOT_new_success_criterion': grasp_diagnostics[task]}
    evidence = [SOURCE / name for name in ('plan.json', 'progress.json', 'NEURAL_QUERY_RECEIPTS.jsonl')]
    result = {'label': 'MEASURED_QUICK40_DEVELOPMENT_NOT_800_FINAL_OR_FIVE_SEEDS',
              'UTC': utc(), 'model': 'vla_jepa', 'checkpoint': str(expected),
              'actual_optimizer_updates': 2500, 'actual_episodes': 40, 'per_action': 10,
              'actual_native_queries': qi, 'actual_motor_controls': controls,
              'real_neural_action_and_input_audit_passed': True, 'prefix': plan['prefix'],
              'source': str(SOURCE), 'role': 'development_selection',
              'all_safety_aborts_retained_in_denominator': True,
              'intrinsic_model_failure_cause_established': False,
              'PUT_metric': 'goal-region first hit, no release/settle requirement',
              'by_task': by_task, 'overall_successes': sum(r['success'] for r in rows),
              'overall_SR': sum(r['success'] for r in rows) / 40,
              'source_sha256': {str(p): sha(p) for p in evidence}}
    atomic_json(out / 'MEASURED_QUICK40.json', result)
    lines = ['# JEPA: 40 actual episodes — 10 per action', '',
             '**Actual measured development check, not a synthetic forecast or the final 800-episode result.**', '',
             f'Checkpoint: `{expected}`. Actual optimizer updates: 2500. Execution prefix: {plan["prefix"]}.', '',
             '|Action|Successes|Measured SR|Wilson 95% CI|Safety aborts|Failed at step limit|',
             '|---|---:|---:|---:|---:|---:|']
    for task, r in by_task.items():
        lo, hi = r['wilson_95']
        lines.append(f'|{task}|{r["successes"]}/{r["episodes"]}|{100*r["SR"]:.1f}%|'
                     f'{100*lo:.1f}%–{100*hi:.1f}%|{r["safety_aborts"]}|{r["failed_at_step_limit"]}|')
    lines += ['', f'Total: {result["overall_successes"]}/40. All {qi} native queries and {controls} '
              'passed neural motor commands were checked against chronological receipts.', '',
              f'Post-hoc PUT audit: {grasp_diagnostics["pick_place"]["successful_episodes_ending_still_held"]} '
              f'of {by_task["pick_place"]["successes"]} successful PUT episodes ended while still holding the correct object. '
              'These must not be called completed release/settle demonstrations.', '',
              'Inputs: front RGB + wrist RGB + causal GT diagram, state67. '
              'Native neural commands; existing shared low-level controller/contact assistance and safety remain unchanged.', '',
              'This cohort is development_selection and the prefix was selected on separate pilot8 episodes. '
              'It is not an independent final-test estimate or proof of five-seed variance. '
              'PUT checks the goal region, not full release/settle. MOVE/PUT do not require prior verified grasp.', '',
              'The R65 coordinator is intentionally held after launching this 40-episode check; '
              'the evaluator and model server were not paused. Confirmation40/800 do not start until the coordinator is resumed. '
              'pi0-FAST remains cancelled separately.', '',
              'No claim that every failure is intrinsic to the model is justified by this quick check.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def launch(out):
    out.mkdir(parents=True, exist_ok=False)
    status = read(R65 / 'owner_status.json')
    owner = process(status['pid'])
    if not owner or module(owner['argv']) != 'VLA.extension_r65.run':
        raise ValueError('Expected live R65 owner absent')
    atomic_json(out / 'REQUEST.json', {'UTC': utc(), 'requested_total': 40, 'per_action': 10,
        'owner_identity': owner, 'existing_cohort': str(SOURCE),
        'hold_only_coordinator_while_evaluator_and_server_continue': True,
        'prior_800_authorization_preserved_but_no_large_run_before_quick40_report': True})
    with (out / 'watch.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'VLA.jepa_quick40_r67', 'watch', '--output', str(out)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    atomic_json(out / 'LAUNCH.json', {'watch_pid': child.pid, 'UTC': utc()})
    print({'watch_pid': child.pid, 'output': str(out)}, flush=True)


def watch(out):
    with (out / '.watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = read(out / 'REQUEST.json')['owner_identity']
        held = False
        while True:
            if not same_process(owner):
                raise RuntimeError('R65 owner exited; no invented 40-episode result')
            state = read(R65 / 'owner_status.json')
            child = process(state.get('active_child_pid') or 0)
            active40 = (child and module(child['argv']) == 'VLA.extension_r65.evaluate'
                        and '--output' in child['argv']
                        and Path(child['argv'][child['argv'].index('--output') + 1]).resolve() == SOURCE.resolve())
            if not held and active40:
                send(owner, signal.SIGSTOP)
                held = True
                atomic_json(out / 'COORDINATOR_HELD.json', {'UTC': utc(), 'owner_identity': owner,
                    'evaluation_child_not_stopped': child, 'server_pid_not_stopped': state['active_server_pid'],
                    'reason': 'User requested quick40 assessment before the larger benchmark'})
            p = SOURCE / 'progress.json'
            progress = read(p) if p.exists() else {}
            rows = progress.get('rows', [])
            atomic_json(out / 'STATUS.json', {'UTC': utc(), 'status': 'quick40_running' if rows else 'waiting_for_scheduled_quick40',
                'coordinator_held': held, 'completed_episodes': len(rows), 'planned_episodes': 40,
                'successes': sum(r['success'] for r in rows), 'watch_pid': __import__('os').getpid()})
            if progress.get('status') == 'completed_actual_cohort':
                if not held:
                    raise ValueError('Quick40 completed without the requested coordinator hold; inspect timing')
                audit(out)
                atomic_json(out / 'STATUS.json', {'UTC': utc(), 'status': 'completed40_measured_and_audited',
                    'completed_episodes': 40, 'planned_episodes': 40, 'coordinator_held': True,
                    'ready_to_report_before_800': True})
                return
            if 'error' in progress:
                raise RuntimeError('Real quick40 integration error: ' + progress['error'])
            time.sleep(2)


def resume(out):
    held = read(out / 'COORDINATOR_HELD.json')
    owner = held['owner_identity']
    if not same_process(owner):
        raise ValueError('Original held coordinator absent; do not signal a different process')
    send(owner, signal.SIGCONT)
    atomic_json(out / 'COORDINATOR_RESUMED.json', {'UTC': utc(), 'owner_identity': owner,
                                                'prior_large_benchmark_schedule_continues': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['launch', 'watch', 'audit', 'resume'])
    p.add_argument('--output', type=Path, default=OUT)
    a = p.parse_args()
    try:
        globals()[a.mode](a.output)
    except Exception as exc:
        if a.output.exists():
            atomic_json(a.output / 'ERROR_NOT_MODEL_SR.json', {'UTC': utc(), 'error': f'{type(exc).__name__}: {exc}',
                                                              'no_missing_episode_counted_as_failure': True})
        raise
