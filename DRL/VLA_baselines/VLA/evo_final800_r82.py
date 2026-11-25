"""User-requested Evo-1 fixed800 after a FAILED behavioral gate.

Reuses unmodified R80 server/evaluator and checkpoint. Never relabels the gate
as passed, refits on final outcomes, rerolls failed scenes, or fabricates seeds.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import requests

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.cross_vla800_r62.common import SEEDS, TASKS, matrix, resources
from VLA.evo_transfer_r80.common import ROOT, OUT as R80, BASE, PY, SIM, ENV, PORT, read, bindings, verify

OUT = ROOT / 'VLA/audits/evo1_user_requested800_r82_20261020'


def utc():
    return datetime.now(timezone.utc).isoformat()


def report(state):
    lines = ['# Evo-1 — user-requested fixed 800', '',
             'Behavioral gate FAILED (DEV 0/40; CONF 0/40). The user explicitly requested800 anyway.',
             'Same learned checkpoint, native R80 model/action path, physics, safety and semantic success.',
             'PUT means goal-region first hit, NOT mandatory release/settle. This is NOT a strict1cm result.',
             'Inputs: front + wrist RGB + causal privileged GT map + numeric simulator feedback.',
             '', f"Status: {state['status']}; stage: {state['stage']}.",
             f"Completed: {state.get('completed_trials', 0)}/800 actual trials.", '',
             '| Action | Successes / completed | SR |', '|---|---:|---:|']
    for task in TASKS:
        v = state.get('per_action', {}).get(task, {'successes': 0, 'n': 0, 'SR_percent': None})
        sr = 'not yet measured' if not v['n'] else f"{100*v['successes']/v['n']:.1f}%"
        lines.append(f"| {task} | {v['successes']} / {v['n']} | {sr} |")
    lines += ['', 'Partial results are not final800; safety aborts remain failures in the denominator.',
              'One fixed cohort is not five empirical evaluation/training seeds.',
              'No new video or image frames are saved by this final evaluation.', '']
    if state.get('error'):
        lines += ['Integration error (not a model task failure): ' + state['error'], '']
    dest = OUT / 'REPORT.md'
    pending = dest.with_suffix('.md.tmp')
    pending.write_text('\n'.join(lines))
    pending.replace(dest)


def final_audit(frozen):
    progress = read(OUT / 'episodes/progress.json')
    if (progress['status'] != 'completed_actual_final800' or not progress['final800'] or
            progress['checkpoint'] != frozen['checkpoint']):
        raise ValueError('Final800 incomplete or wrong checkpoint')
    results = matrix(progress['rows'])
    expected = {(seed, task) for seed in SEEDS for task in TASKS}
    queries = {}
    with (OUT / 'episodes/NEURAL_QUERIES.jsonl').open() as stream:
        for line in stream:
            q = json.loads(line)
            key = (q['seed'], q['task'], q['index'])
            receipt = q['native_input']
            if (key in queries or key[:2] not in expected or receipt['numeric'] != [1, 67] or
                    receipt['state'] != [1, 91] or receipt['pixels'] != [3, 3, 448, 448] or
                    receipt['future_labels_teacher_phase_or_success']):
                raise ValueError('Invalid or duplicate actual native inference receipt')
            actions = np.asarray(q['actions'], dtype=np.float32)
            if actions.shape != (50, 7) or not np.isfinite(actions).all():
                raise ValueError('Invalid neural action chunk')
            queries[key] = q
    controls = 0
    for seed in SEEDS:
        traces = read(OUT / f'episodes/seed{seed}/trajectory.json')
        if set(traces) != set(TASKS):
            raise ValueError('Missing task trajectory')
        for task in TASKS:
            row = next(r for r in progress['rows'] if r['seed'] == seed and r['task'] == task)
            trace = traces[task]
            if len(trace) != row['steps'] or not trace or trace[-1]['success'] != row['success']:
                raise ValueError('Row / trajectory mismatch')
            for index, step in enumerate(trace):
                q = queries[(seed, task, index // frozen['prefix'])]
                predicted = np.asarray(q['actions'][index % frozen['prefix']], dtype=np.float32)
                actual = np.asarray(step['action'], dtype=np.float32)
                if not np.array_equal(predicted, actual) or step['query_sha256'] != q['input_sha256']:
                    raise ValueError('Executed action was not from the frozen native chunk')
                controls += 1
    if len(queries) != progress['native_queries'] or controls != progress['motor_controls']:
        raise ValueError('Inference/control counters disagree')
    for task in TASKS:
        expected_successes = int(results[:, TASKS.index(task)].sum())
        if progress['per_action'][task]['n'] != 200 or progress['per_action'][task]['successes'] != expected_successes:
            raise ValueError('Per-action denominator or success count mismatch')
    verify(frozen['code'])
    verify(frozen['evidence_files_sha256'])
    result = {'passed': True, 'UTC': utc(), 'all800_unique_fixed_trials': True,
              'each_action200': True, 'all_executed_actions_verified_against_actual_neural_chunks': True,
              'native_queries': len(queries), 'motor_controls': controls,
              'per_action': progress['per_action'], 'behavioral_gate_passed': False,
              'behavioral_gate_override': 'explicit user request, NOT a claim of readiness',
              'safety_aborts_retained': sum(bool(r['safety_violation']) for r in progress['rows']),
              'progress_sha256': sha256_file(OUT / 'episodes/progress.json')}
    atomic_json(OUT / 'FINAL_AUDIT.json', result)
    return progress


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / '.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'owner_status.json').exists():
            raise FileExistsError('One-shot800 owner already declared; do not duplicate or overwrite')
        started = time.monotonic()
        server = child = server_log = None
        stage = 'validating_completed_checkpoint_and_explicit_user_override'
        state = {}

        def save(status='running', **extra):
            nonlocal state
            p = OUT / 'episodes/progress.json'
            progress = read(p) if p.exists() else {}
            state = {'status': status, 'stage': stage, 'pid': os.getpid(), 'updated_UTC': utc(),
                     'active_server_pid': server.pid if server and server.poll() is None else None,
                     'active_child_pid': child.pid if child and child.poll() is None else None,
                     'elapsed_wall_s': time.monotonic() - started,
                     'planned_trials': 800, 'completed_trials': progress.get('completed_trials', 0),
                     'per_action': progress.get('per_action', {}), 'behavioral_gate_passed': False, **extra}
            atomic_json(OUT / 'owner_status.json', state)
            report(state)

        try:
            save()
            previous = read(R80 / 'owner_status.json')
            if previous['status'] != 'completed' or previous['active_child_pid'] or previous['active_server_pid']:
                raise ValueError('Prior finite workflow must be complete with no owned workers')
            selection = read(R80 / 'DEVELOPMENT_SELECTION.json')
            cp = Path(selection['checkpoint']).resolve()
            if cp != R80 / 'fit4096' or selection['prefix'] != 10:
                raise ValueError('Unexpected pre-final selection: do not silently change checkpoint/prefix')
            meta = read(cp / 'TRAINING.json')
            reload_file = R80 / 'fit4096_FRESH_RELOAD.json'
            reload_proof = read(reload_file)
            if (not meta['completed'] or meta['actual_optimizer_updates'] != 4096 or
                    not reload_proof['passed'] or reload_proof['checkpoint'] != str(cp)):
                raise ValueError('Completed4096 and exact fresh reload required')
            cohort = read(R80 / 'final800_cohort.json')
            if cohort['role'] != 'final_fixed_evaluation800' or cohort['seeds'] != SEEDS:
                raise ValueError('Wrong fixed800 cohort')
            if cohort != read(ROOT / 'VLA/audits/flower800_r77_20261019/COHORT800.json'):
                raise ValueError('Cohort differs from FLOWER/R62')
            if (R80 / 'final800').exists():
                raise FileExistsError('Prior final800 exists: do not duplicate it')
            bound = bindings()
            bound[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
            evidence = [cp / 'learned.safetensors', cp / 'stats.json', cp / 'TRAINING.json', reload_file,
                        BASE / 'mp_rank_00_model_states.pt', R80 / 'DEVELOPMENT_SELECTION.json',
                        R80 / 'confirmation40/progress.json', R80 / 'final800_cohort.json']
            digests = {str(p): sha256_file(p) for p in evidence}
            if digests[str(cp / 'learned.safetensors')] != meta['learned_sha256']:
                raise ValueError('Learned checkpoint hash mismatch')
            verify(bound)
            frozen = {'checkpoint': str(cp), 'checkpoint_sha256': meta['learned_sha256'],
                      'prefix': selection['prefix'], 'code': bound, 'evidence_files_sha256': digests,
                      'selected_before_final800': True, 'actual_optimizer_updates': 4096,
                      'behavioral_gate_passed': False, 'confirmation': previous['per_action'],
                      'behavioral_gate_override': 'User explicitly requested800 after finetune on2026-10-20',
                      'gate_override_changes_no_model_controller_physics_metric_or_denominator': True,
                      'per_action_planned': 200, 'final800_used_for_selection_or_gradients': False}
            atomic_json(OUT / 'FROZEN_BEFORE_800.json', frozen)
            atomic_json(OUT / 'COHORT800.json', cohort)
            stage = 'waiting_safe_exclusive_model_resources'
            for _ in range(240):
                current = resources()
                compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                if current['GPU_free_MiB'] >= 13500 and current['RAM_available_MiB'] >= 6000 and not compute:
                    break
                save(resource_admission=current)
                time.sleep(15)
            else:
                raise TimeoutError('Finite one-hour resource wait; no other job preempted')
            stage = 'loading_frozen_actual_Evo1_native_server'
            server_log = (OUT / 'server.log').open('x')
            server = subprocess.Popen([str(PY), '-u', '-m', 'VLA.evo_transfer_r80.server',
                '--checkpoint', str(cp), '--reload-proof', str(reload_file)], cwd=ROOT, env=ENV,
                stdin=subprocess.DEVNULL, stdout=server_log, stderr=subprocess.STDOUT)
            save()
            with requests.Session() as session:
                session.trust_env = False
                for _ in range(180):
                    if server.poll() is not None:
                        raise RuntimeError('Native server exited during load; see server.log')
                    try:
                        response = session.get(f'http://127.0.0.1:{PORT}/health', timeout=2)
                        response.raise_for_status()
                        health = response.json()
                        break
                    except requests.RequestException:
                        save()
                        time.sleep(2)
                else:
                    raise TimeoutError('Native server startup exceeded six minutes')
            if (health['checkpoint'] != str(cp) or health['checkpoint_sha256'] != frozen['checkpoint_sha256'] or
                    not health['actual_neural_model'] or health['Oracle_waypoint_or_geometric_action_repair']):
                raise ValueError('Wrong actual native server')
            atomic_json(OUT / 'SERVER_HEALTH.json', health)
            stage = 'running_fixed800_actual_neural_no_video'
            with (OUT / 'evaluation.log').open('x') as log:
                child = subprocess.Popen([str(SIM), '-u', '-m', 'VLA.evo_transfer_r80.evaluate',
                    '--cohort', str(OUT / 'COHORT800.json'), '--output', str(OUT / 'episodes'),
                    '--prefix', str(frozen['prefix']), '--frozen-method', str(OUT / 'FROZEN_BEFORE_800.json')],
                    cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    save()
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'Actual evaluator exited{child.returncode}; preserve partial trials as-is')
            child = None
            stage = 'independent800_coverage_and_neural_action_audit'
            save()
            final_audit(frozen)
            stage = 'completed800_actual_per_action_and_independent_audit'
            save('completed')
        except Exception as error:
            save('integration_error_NOT_zero_model_SR', error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            for process in (child, server):
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            if server_log:
                server_log.close()
            if state:
                state.update(active_child_pid=None, active_server_pid=None, updated_UTC=utc())
                atomic_json(OUT / 'owner_status.json', state)
                report(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        OUT.mkdir(exist_ok=True)
        with (OUT / 'owner.log').open('x') as log:
            process = subprocess.Popen([str(PY), '-u', '-m', 'VLA.evo_final800_r82'],
                cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'owner_pid': process.pid, 'output': str(OUT)}))
    else:
        main()
