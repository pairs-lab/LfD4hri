"""Fix gate/long-entrypoint mismatch without editing hash-bound VLA sources.

The old coordinator alone is retired; its running X-VLA evaluation finishes.
Adapter then uses the SAME acquisition-balanced entrypoint in gate and long
mode. Existing R45/R47 owners consume a transparent status bridge afterward.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from VLA.common.artifact_io import atomic_json
from VLA.stable_native_r44.fit_balanced import SAMPLER

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016'
OLD = BASE/'bounded_native_pipeline_v2/status.json'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PYTHON = ROOT/'VLA/.venvs/openvla/bin/python'
DATA = ROOT/'VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts'


def read(path):
    return json.loads(Path(path).read_text())


def fit_command(mode, output, updates, resume=None, reload=None):
    command = [str(PYTHON), '-m', 'VLA.stable_native_r44.fit_balanced', '--family', 'vla_adapter',
        '--mode', mode, '--data', str(DATA), '--output', str(output), '--updates', str(updates),
        '--total-updates', '6000', '--accumulation', '4', '--learning-rate', '.0001']
    if resume is not None: command += ['--resume', str(resume)]
    if reload is not None: command += ['--reload-proof', str(reload)]
    return command


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original = read(OLD)
    stage, child, results = 'retiring_only_old_coordinator_NOT_its_evaluator', None, original['results']
    atomic_json(args.output/'original_coordinator_status.json', original)
    def status(state='running', **extra):
        value = {'pid': os.getpid(), 'status': state, 'stage': stage,
            'active_child_pid': None if child is None else child.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results,
            'supersedes_old_coordinator_pid': original['pid'], 'current_owner_output': str(args.output.resolve()),
            'bridge_keeps_existing_native_eval_and_extended_fit_owners_serial': True, **extra}
        atomic_json(args.output/'status.json', value)
        atomic_json(OLD, value)
    def run(command, label, checked_plan=None):
        nonlocal child, stage
        stage = label
        with (args.output/(label+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            status()
            checked = False
            while child.poll() is None:
                if checked_plan is not None and checked_plan.exists() and not checked:
                    actual = read(checked_plan)
                    if actual.get('training_sampler') != SAMPLER or actual['sampling'] != SAMPLER['schema']:
                        child.terminate(); child.wait(timeout=30)
                        raise ValueError('Actual long/gate plan lost acquisition-balanced sampler')
                    atomic_json(args.output/(label+'_verified_actual_recipe.json'), actual)
                    checked = True
                time.sleep(5)
            code = child.returncode
        child = None
        if code: raise RuntimeError(f'{label} exited{code}; no blind retry or fabricated zero SR')
        if checked_plan is not None and read(checked_plan).get('training_sampler') != SAMPLER:
            raise ValueError('Completed recipe differs from intended sampler')
        status()
    try:
        proc = Path('/proc')/str(original['pid'])
        if proc.exists():
            if b'VLA.stable_native_r44.workflow_v2' not in (proc/'cmdline').read_bytes():
                raise ValueError('Old PID identity changed; do not signal')
            state = (proc/'stat').read_text().split(') ', 1)[1][0]
            if state != 'T': raise ValueError('Require old coordinator already STOPPED before takeover')
            os.kill(original['pid'], signal.SIGTERM)
            os.kill(original['pid'], signal.SIGCONT)
            until = time.monotonic()+30
            while proc.exists() and time.monotonic() < until: time.sleep(.2)
            if proc.exists() and (proc/'stat').read_text().split(') ', 1)[1][0] != 'Z':
                raise ValueError('Old coordinator not retired')
        atomic_json(args.output/'plan.json', {'reason': 'old_gate_used_fit_balanced_but_long_used_train_entrypoint',
            'old_X7000_actual_recipe_was_motion_weighted_NOT_acquisition_balanced': True,
            'old_fit_and_all_results_preserved': True, 'running_X_evaluation_interrupted': False,
            'Adapter_gate_and_long_same_entrypoint': 'VLA.stable_native_r44.fit_balanced',
            'expected_training_sampler': SAMPLER, 'actual_plan_and_checkpoint_recipe_checked': True,
            'Adapter_redundant_first_action_only40_skipped': True,
            'Adapter_native8_development40_already_queued_under_R45': True,
            'training_data_stats_physics_camera_success_unchanged': True})
        stage = 'waiting_existing_X7000_first_action_diagnostic_to_finish_NO_preemption'
        status()
        progress = ROOT/'VLA/xvla/privileged_r44_bounded67_20261016/test1/development40/progress.json'
        until = time.monotonic()+3600
        while True:
            p = read(progress)
            if p['status'].startswith('completed40'): break
            if p['status'].startswith('failed') or time.monotonic() > until:
                raise ValueError('Existing X diagnostic not completed; preserve evidence')
            time.sleep(5)
        results.append({'family': 'xvla', 'label': 'first_action_only40_DIAGNOSTIC_NOT_native_queue_SR',
            'successes': p['successes'], 'trials': p['completed_trials'], 'progress': str(progress)})
        model = ROOT/'VLA/vla_adapter/privileged_r44_bounded67_20261016/test1'
        model.mkdir(parents=True, exist_ok=False)
        gate, reload = args.output/'Adapter_native_gate2', args.output/'Adapter_native_reload2'
        run(fit_command('gate', gate, 2), 'Adapter_acquisition_balanced_gate2', gate/'plan.json')
        cp = gate/'checkpoints/step-000002'
        run([str(PYTHON), '-m', 'VLA.stable_native_r44.reload_gate', '--checkpoint', str(cp),
            '--output', str(reload)], 'Adapter_own_fresh_reload2')
        run(fit_command('long', model/'train', 6000, cp, reload/'audit.json'),
            'Adapter_actual_acquisition_balanced_fit6000', model/'train/plan.json')
        cp = Path(read(model/'train/latest_checkpoint.json')['checkpoint'])
        if read(cp/'config.json').get('training_sampler') != SAMPLER:
            raise ValueError('Saved checkpoint recipe lost sampler')
        stage = 'balanced_Adapter_fit_completed_native40_owners_can_continue'
        results.append({'family': 'vla_adapter', 'label': 'own_balanced_native6000_fit_not_SR',
            'checkpoint': str(cp), 'actual_optimizer_updates': 6000})
        status('completed_balanced_Adapter_fit_native40_queued')
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
            raise FileExistsError('Do not duplicate dispatch owner')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            process = subprocess.Popen([str(SIM), '-m', 'VLA.run_balanced_dispatch_r49',
                '--output', str(args.output.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': process.pid})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}))
    else:
        main(args)
