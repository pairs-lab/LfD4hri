"""One finite extra exposure pass after current VLA fits/native evaluations.

No current job is preempted. Fixed last20k checkpoint; held-out monitor never
feeds the optimizer and is not used to select a favorable reported cohort.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016'
DATA = ROOT/'VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
XVLA = ROOT/'VLA/.venvs/xvla/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    stage, child = 'waiting_current_two_model_native_evaluations_NO_preemption', None
    def status(state='running', **extra):
        atomic_json(args.output/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'active_child_pid': None if child is None else child.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra})
    def run(python, module, arguments, label):
        nonlocal child, stage
        stage = label
        with (args.output/(label+'.log')).open('x') as log:
            child = subprocess.Popen([str(python), '-m', module, *arguments], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            status()
            while child.poll() is None:
                if (model/'train/latest_checkpoint.json').exists():
                    checkpoint = Path(read(model/'train/latest_checkpoint.json')['checkpoint'])
                    monitor = checkpoint/'heldout_monitor.json'
                    if monitor.exists():
                        atomic_json(args.output/'latest_heldout_monitor.json', read(monitor))
                time.sleep(10)
            code = child.returncode
        child = None
        if code: raise RuntimeError(f'{label} failed with exit{code}; evidence retained')
        status()
    model = ROOT/'VLA/xvla/privileged_r47_bounded_acquisition20k_20261016/test1'
    try:
        atomic_json(args.output/'plan.json', {'scope': 'one_bounded_extra_XVLA_exposure_with_independent_validation_monitor',
            'wait': str(BASE/'native_execution40_each_r45/status.json'),
            'cumulative_updates_from': 7000, 'cumulative_updates_to': 20000,
            'additional_actual_optimizer_updates': 13000, 'effective_microbatch': 4,
            'data_source_and_stats_unchanged': True, 'acquisition_balanced_and_bounded67': True,
            'validation_used_for_optimization_or_normalizer': False, 'fixed_last20000_selection': True,
            'new_wrongheld_or_drop_recovery_data_claimed': False,
            'native_queue40_after_fit': True, 'reserved_final_test_used': False,
            'expected_extra_fit_order_of_magnitude_hours_NOT_guarantee': 4,
            'success_rate_guarantee': None})
        status()
        until = time.monotonic()+18*3600
        while True:
            progress = read(BASE/'native_execution40_each_r45/status.json')
            if progress['status'].startswith(('completed', 'failed')): break
            if time.monotonic() > until: raise TimeoutError('Current owned pipeline not terminal before18h')
            time.sleep(30)
        fixture = BASE/'heldout_validation8_native_fixture'
        if not read(fixture/'audit.json')['passed']: raise ValueError('Validation physical replay failed')
        parent = ROOT/'VLA/xvla/privileged_r44_bounded67_20261016/test1/train/checkpoints/step-007000'
        if read(parent/'config.json')['actual_optimizer_updates'] != 7000:
            raise ValueError('Own actual7k parent required')
        model.mkdir(parents=True, exist_ok=False)
        common = ['--data', str(DATA), '--accumulation', '4', '--learning-rate', '.0001',
            '--total-updates', '20000', '--validation-fixture', str(fixture)]
        gate, reload = args.output/'native_gate7002', args.output/'native_reload7002'
        run(XVLA, 'VLA.fit_xvla_validation_r47', ['--mode', 'gate', '--output', str(gate),
            '--updates', '7002', '--resume', str(parent), *common], 'own_native_gate2_NEW_updates')
        cp = gate/'checkpoints/step-007002'
        run(XVLA, 'VLA.stable_native_r44.reload_gate', ['--checkpoint', str(cp), '--output', str(reload)],
            'own_fresh_process_reload_after_new_gate')
        run(XVLA, 'VLA.fit_xvla_validation_r47', ['--mode', 'long', '--output', str(model/'train'),
            '--updates', '20000', '--resume', str(cp), '--reload-proof', str(reload/'audit.json'), *common],
            'actual_native_acquisition_balanced_fit_to20000')
        cp = Path(read(model/'train/latest_checkpoint.json')['checkpoint'])
        selection = BASE/'native_execution40_each_r45/xvla_prefix_selection.json'
        prefix = read(selection).get('selected', {}).get('execution_steps', 30) if selection.exists() else 30
        run(SIM, 'VLA.evaluate_native_chunks_r45', ['--family', 'xvla', '--port', '5850', '--seed-count', '10',
            '--execution-steps', str(prefix), '--own-server-checkpoint', str(cp), '--output', str(model/'development40')],
            'xvla20k_native_queue_development40')
        progress = read(model/'development40/progress.json')
        stage = 'finite_extended_exposure_and_actual40_finished_review_evidence'
        status('completed', successes=progress['successes'], trials=progress['completed_trials'],
               progress=str(model/'development40/progress.json'))
    except Exception as error:
        status('failed_preserve_evidence_NO_blind_retry', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.output.exists() or args.output.with_name(args.output.name+'_owner.json').exists():
            raise FileExistsError('Do not duplicate exposure continuation')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            process = subprocess.Popen([str(SIM), '-m', 'VLA.run_xvla_exposure20k_r47',
                '--output', str(args.output.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': process.pid})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}))
    else:
        main(args)
