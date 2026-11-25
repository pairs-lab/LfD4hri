"""Finite prioritized new-VLA pipeline: qualify -> fit -> real development40.

No historical queue is resumed, no final seeds opened, no process preempted,
no outcome-selected rerolls, and no automatic assertion of high success.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
N = ROOT/'VLA/audits/new_vlas_r43_20261016'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
DATA = N/'native_complete_data_v4_native_contracts'


def wait_completed(path, *, deadline=1800):
    until = time.monotonic()+deadline
    while True:
        if path.is_file():
            value = json.loads(path.read_text())
            state = value.get('status', '')
            if state == 'completed':
                return value
            if state.startswith('failed'):
                raise RuntimeError('Preflight failed, not ignored: '+str(path)+' '+str(value))
        if time.monotonic() > until:
            raise TimeoutError('Preflight did not complete: '+str(path))
        time.sleep(3)


def verify_lineage(checkpoint):
    cfg = json.loads((checkpoint/'config.json').read_text())
    saved = json.loads((checkpoint/'saved_values.json').read_text())
    for path, digest in cfg['code'].items():
        if sha256_file(path) != digest:
            raise ValueError('Native training/inference code changed after gate: '+path)
    for filename, key in (('config.json', 'config_sha256'), ('trainable_masters.pt', 'trainable_master_sha256'),
                          ('optimizer.pt', 'optimizer_sha256')):
        if sha256_file(checkpoint/filename) != saved[key]:
            raise ValueError('Own preflight checkpoint changed: '+filename)
    return cfg


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage, child, results = 'waiting_current_preflight', None, []
    def status(state='running', **extra):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'active_child_pid': None if child is None else child.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results, **extra})
    def run(python, module, arguments, label):
        nonlocal child, stage
        stage = label
        with (out/(label+'.log')).open('x') as log:
            child = subprocess.Popen([str(python), '-m', module, *arguments], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            status()
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f'{label} exited {code}; actual evidence retained')
        status()
    try:
        atomic_json(out/'plan.json', {'scope': 'NEW_VLA_prioritized_bounded_finetune_then_real40_each',
            'order': ['xvla', 'vla_adapter'], 'total_optimizer_updates_per_model': 6000,
            'microbatch': 1, 'gradient_accumulation': 4, 'data_root': str(DATA),
            'each_model_episodes': 40, 'episodes_each_action': 10,
            'selection': 'fixed_last6000_checkpoint_NOT_selected_on_eval_success',
            'privileged67_used_by_both_models': True, 'historical800_cancelled': True,
            'old_waiters_NOT_resumed': True, 'final_test_seeds_NOT_used': True,
            'no_oracle_action_fallback': True, 'success_rate_guarantee': None})
        status()
        wait_completed(DATA/'status.json')
        wait_completed(N/'public_checkpoint_full_hash_v1/status.json')
        public = json.loads((N/'public_checkpoint_full_hash_v1/audit.json').read_text())
        if not public.get('passed') or not public.get('all_public_LFS_payload_SHA256_verified'):
            raise ValueError('Both pinned public checkpoints need full LFS hash verification')
        for family, short in (('xvla', 'xvla'), ('vla_adapter', 'adapter')):
            python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
            gate_out = N/(short+'_optimizer_gate_v3')
            reload_out = N/(short+'_fresh_reload_v3')
            image_out = N/(short+'_actual_native_images_v3')
            if gate_out.exists():
                wait_completed(gate_out/'status.json')
            else:
                run(python, 'VLA.new_vlas.train', ['--family', family, '--mode', 'gate',
                    '--data', str(DATA), '--output', str(gate_out), '--updates', '2',
                    '--accumulation', '4', '--learning-rate', '0.0001'], short+'_optimizer_gate_v3')
            gate_cp = gate_out/'checkpoints/step-000002'
            if reload_out.exists():
                wait_completed(reload_out/'status.json')
            else:
                run(python, 'VLA.new_vlas.reload_gate', ['--checkpoint', str(gate_cp),
                    '--output', str(reload_out)], short+'_fresh_reload_v3')
            if not (image_out/'audit.json').exists():
                run(python, 'VLA.new_vlas.audit_images', ['--family', family,
                    '--data', str(DATA), '--output', str(image_out)], short+'_native_images_v3')
            proof_path = image_out/'audit.json'
            until = time.monotonic()+600
            while not proof_path.is_file():
                if time.monotonic() > until:
                    raise TimeoutError('Native image/online-state audit did not complete')
                time.sleep(3)
            proof = json.loads(proof_path.read_text())
            if not proof['passed'] or not proof['train_runtime_native_state_exact_for8anchors']:
                raise ValueError('Native image/state admission failed: '+family)
        review_path = N/'native_image_visual_review_v3.json'
        until = time.monotonic()+600
        while not review_path.is_file():
            if time.monotonic() > until:
                raise TimeoutError('Actual native224 images still need visual inspection')
            time.sleep(3)
        review = json.loads(review_path.read_text())
        if not review.get('reviewed_both_native224_contact_sheets'):
            raise ValueError('Do not waive actual native224 visual inspection')
        for path, digest in review['bound_files'].items():
            if sha256_file(path) != digest:
                raise ValueError('Visually reviewed image bytes changed')
        for family, short, port in (('xvla', 'xvla', 5843), ('vla_adapter', 'adapter', 5844)):
            try:
                gate_cp = N/(short+'_optimizer_gate_v3')/'checkpoints/step-000002'
                cfg = verify_lineage(gate_cp)
                reload_proof = N/(short+'_fresh_reload_v3')/'audit.json'
                model_root = ROOT/'VLA'/family/'privileged_r43_native_v3_20261016/test1'
                python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
                run(python, 'VLA.new_vlas.train', ['--family', family, '--mode', 'long',
                    '--data', str(DATA), '--output', str(model_root/'train'), '--updates', '6000',
                    '--total-updates', '6000', '--accumulation', '4', '--learning-rate', '0.0001',
                    '--resume', str(gate_cp), '--reload-proof', str(reload_proof)], family+'_finetune6000')
                completed_cp = Path(json.loads((model_root/'train/latest_checkpoint.json').read_text())['checkpoint'])
                run(SIM, 'VLA.new_vlas.evaluate', ['--checkpoint', str(completed_cp),
                    '--output', str(model_root/'development40'), '--port', str(port)], family+'_development40')
                progress = json.loads((model_root/'development40/progress.json').read_text())
                if progress['completed_trials'] != 40:
                    raise ValueError('Incomplete actual40 evaluation')
                results.append({'family': family, 'status': 'completed_train6000_and_actual40',
                    'checkpoint': str(completed_cp), 'progress': str(model_root/'development40/progress.json'),
                    'successes': progress['successes'], 'trials': progress['completed_trials']})
            except Exception as error:
                results.append({'family': family, 'status': 'failed_preserve_evidence_NOT_zero_SR',
                    'error': f'{type(error).__name__}: {error}'})
            status()
        stage = 'finite_new_model_pipeline_finished_review_actual_results'
        status('completed' if all(r['status'].startswith('completed') for r in results) else 'completed_with_failures_retained')
    except Exception as error:
        status('failed_preflight_no_long_fit_launched', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.output.exists() or args.output.with_name(args.output.name+'_owner.json').exists():
            raise FileExistsError('Never duplicate an existing new-model pipeline')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            proc = subprocess.Popen([str(SIM), '-m', 'VLA.new_vlas.workflow',
                '--output', str(args.output.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2'})
        atomic_json(args.output.with_name(args.output.name+'_owner.json'),
                    {'pid': proc.pid, 'output': str(args.output.resolve()), 'new_VLAs_only': True})
        print(json.dumps({'pid': proc.pid, 'output': str(args.output.resolve())}))
    else:
        main(args)
