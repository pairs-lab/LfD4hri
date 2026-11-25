"""Finite8/model task-prefix reproduction on EXISTING loaded native servers.

No server/trainer is stopped or started. Requests are inference-only; X's
runtime forks/restores RNG per query and Adapter is deterministic eval. Both
simulators pause during synchronous inference, so no wall-clock physics race.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.artifact_io import atomic_json
from VLA.evaluate_task_chunks_r52 import choose_schedule

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016/chunk_feedback_r50'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    def publish(stage, **kw):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'stage': stage, 'rows': rows,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), **kw})
    atomic_json(out/'plan.json', {'scope': 'matched8_selected_task_schedule_reproduction_NOT_independent_SR',
        'models': ['xvla', 'vla_adapter'], 'scheduled_episodes_each_model': 8,
        'task_selection': 'per-task most success, then fewer safety aborts, then native, then longer',
        'model_native_actions_remain_unchanged': True, 'teacher_or_target_geometry_controller': False,
        'source_selection_is_existing_DEVELOPMENT2_per_action': True,
        'no_new_GPU_model_and_no_existing_job_interrupted': True,
        'R50_native_runtime_predict_restores_X_RNG_and_Adapter_eval_is_deterministic': True,
        'minimum_required_complete_prefix_candidates': 3})
    for family, port in [('xvla', 5847), ('vla_adapter', 5848)]:
        selection = BASE/(family+'_prefix_selection.json')
        until = time.monotonic()+8*3600
        while not selection.exists():
            publish('waiting_actual_'+family+'_complete_prefix_sweep')
            if time.monotonic() > until:
                raise TimeoutError('No complete source selection after8hours')
            time.sleep(10)
        old = json.loads(selection.read_text())
        paths = [Path(c['evidence']) for c in old['candidates']]
        schedule = choose_schedule(paths, family)
        schedule_path = out/(family+'_schedule.json')
        atomic_json(schedule_path, schedule)
        folder = out/(family+'_matched8')
        with (out/(family+'_matched8.log')).open('x') as log:
            child = subprocess.Popen([str(SIM), '-u', '-m', 'VLA.evaluate_task_chunks_r52',
                '--schedule', str(schedule_path), '--port', str(port), '--output', str(folder)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'})
            publish(family+'_matched8_running', child_pid=child.pid)
            while child.poll() is None:
                time.sleep(5)
            code = child.returncode
        row = {'family': family, 'exit_code': code, 'progress': str(folder/'progress.json')}
        if not code:
            p = json.loads((folder/'progress.json').read_text())
            row.update(successes=p['successes'], trials=p['completed_trials'],
                scope='selected_DEVELOPMENT_matched8_NOT_generalized_SR', task_prefix=p['task_prefix'])
        else:
            row['integration_error_NOT_zero_SR'] = True
        rows.append(row)
        publish('completed_'+family+'_matched8')
    publish('finite_task_schedule_checks_completed_NOT_independent40')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.with_suffix('.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'VLA.watch_task_chunks_r52',
                '--output', str(a.output.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'owner_pid': child.pid, 'output': str(a.output.resolve())}), flush=True)
    else:
        try:
            main(a)
        except Exception as error:
            if a.output.is_dir():
                atomic_json(a.output/'error.json', {'error': f'{type(error).__name__}: {error}', 'no_fake_SR': True})
            raise
