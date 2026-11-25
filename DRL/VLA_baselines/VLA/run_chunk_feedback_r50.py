"""Finite matched short-chunk ablation, then40 development trials per VLA.

Keep actual Adapter optimization running. Replace only the idle R45 evaluator
owner, preserving its status/selection interface for the queued X20k owner.
Every action is still neural. No target-based action override, phase machine,
physics/scoring change, reserved-test tuning or integration-error fake zero.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016'
BRIDGE = BASE/'native_execution40_each_r45'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
TASKS = ('touch', 'pick', 'move', 'pick_place')
PAIRED_SEEDS = (20268001, 20268002)


def read(path):
    return json.loads(Path(path).read_text())


def candidate(progress, *, checkpoint, prefix, evidence):
    """Only exact-checkpoint, complete matched8 candidates may be selected."""
    if Path(progress['checkpoint']).resolve() != Path(checkpoint).resolve():
        raise ValueError('Do not compare different checkpoint weights as execution prefixes')
    if progress.get('native_execution_chunk') != prefix or not progress.get('privileged67'):
        raise ValueError('Actual prefix/privileged-input contract differs')
    rows = [r for r in progress['rows'] if r['seed'] in PAIRED_SEEDS]
    expected = {(seed, task) for seed in PAIRED_SEEDS for task in TASKS}
    if len(rows) != 8 or {(r['seed'], r['task']) for r in rows} != expected:
        raise ValueError('Need the identical complete8, not partial or rerolled scenes')
    if prefix < 4:
        raise ValueError('Known future-CLOSE loss at prefix1; do not silently reintroduce it')
    return {'execution_steps': prefix, 'successes': sum(bool(r['success']) for r in rows),
        'safety_aborts': sum(bool(r.get('safety_violation')) for r in rows),
        'trials': 8, 'evidence': str(evidence), 'checkpoint': str(checkpoint)}


def choose(candidates, native):
    if not candidates: raise ValueError('No complete matched pilot; do not manufacture a result')
    # Prespecified development-only selection. Prefer official native execution
    # on exact ties rather than claiming a gratuitous runtime modification.
    return max(candidates, key=lambda c: (c['successes'], -c['safety_aborts'],
        c['execution_steps'] == native, c['execution_steps']))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    stage, child, server, results = 'qualifying_idle_evaluation_owner_takeover', None, None, []
    def status(state='running', **extra):
        value = {'pid': os.getpid(), 'status': state, 'stage': stage,
            'active_child_pid': None if child is None else child.pid,
            'active_server_pid': None if server is None else server.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results,
            'current_owner_output': str(args.output.resolve()),
            'supersedes_IDLE_R45_owner_only': True, 'actual_training_interrupted': False, **extra}
        atomic_json(args.output/'status.json', value)
        atomic_json(BRIDGE/'status.json', value)
    def report():
        lines = ['# R50 — shorter feedback, genuine privileged VLA', '',
            'Development selection on matched8; then10 episodes/action/model. NOT reserved final test.',
            'Same checkpoint within each prefix comparison. All controls neural; shared physics/camera/scoring.', '',
            '| Model | Selected prefix | touch | pick | move | pick_place | Overall |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for result in results:
            if result['status'] != 'completed40':
                lines += ['', f"{result['family']}: {result['status']}; {result.get('error', '')}"]
                continue
            p = read(result['progress'])
            cells = []
            for task in TASKS:
                rows = [r for r in p['rows'] if r['task'] == task]
                cells.append(f"{sum(r['success'] for r in rows)}/{len(rows)}")
            lines.append(f"| {result['family']} | {result['execution_steps']} | "+' | '.join(cells)+f" | {p['successes']}/40 |")
        lines += ['', 'All failed/aborted episodes retained. Integration errors/unrun episodes are not task failures.',
            'Place scoring remains target-region reaching, not necessarily release/stable placement.',
            'Higher success is a hypothesis to test, not a promised result.']
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        (BRIDGE/'REPORT.md').write_text('# Superseded by R50 before evaluation started\n\n'+
            f'Current report: [R50]({args.output.resolve()}/REPORT.md).\n\n'+ '\n'.join(lines)+'\n')
    def run(command, label):
        nonlocal stage, child
        stage = label
        with (args.output/(label+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            status()
            while child.poll() is None:
                time.sleep(10); status()
            code = child.returncode
        child = None
        if code: raise RuntimeError(f'{label} exited{code}; all partial evidence preserved')
    def evaluate(family, port, prefix, count, folder, label):
        run([str(SIM), '-m', 'VLA.evaluate_native_chunks_r45', '--family', family, '--port', str(port),
            '--execution-steps', str(prefix), '--seed-count', str(count), '--output', str(folder)], label)
        p = read(folder/'progress.json')
        if p['completed_trials'] != count*4: raise ValueError('Incomplete declared cohort')
        return p
    try:
        previous = read(BRIDGE/'status.json')
        if previous['active_child_pid'] is not None or not previous['stage'].startswith('waiting_'):
            raise ValueError('Do not replace an active rollout/server/trainer')
        fit = read(BASE/'balanced_dispatch_r49/status.json')
        if fit['status'] != 'running':
            raise ValueError('Idle-owner takeover window changed; re-audit rather than race GPU owners')
        proc = Path('/proc')/str(previous['pid'])
        if proc.exists():
            if b'VLA.run_native_final40_r45' not in (proc/'cmdline').read_bytes():
                raise ValueError('Exact old PID identity required before retirement')
            os.kill(previous['pid'], signal.SIGTERM)
            until = time.monotonic()+30
            while proc.exists() and time.monotonic() < until: time.sleep(.2)
            if proc.exists() and (proc/'stat').read_text().split(') ', 1)[1][0] != 'Z':
                raise ValueError('Old idle owner did not exit')
        atomic_json(args.output/'old_IDLE_owner_snapshot.json', previous)
        atomic_json(args.output/'plan.json', {'scope': 'finite_closed_loop_feedback_hyperparameter_ablation',
            'hypothesis': 'prefix4_or6_retains_future_CLOSE_but_requeries_before3sec_old_chunk_finishes',
            'same_checkpoint_within_sweep': True, 'no_model_weights_modified_by_sweep': True,
            'new_pilot_prefixes_XVLA': [4, 6], 'existing_matched_X7000_baselines': [8, 30],
            'pilot_prefixes_Adapter': [8, 4, 6], 'pilot_episodes_each': 8,
            'selection': 'most_success_then_fewer_safety_aborts_then_official_native_then_longer_prefix',
            'model_native_horizons_NOT_changed': {'xvla': 30, 'vla_adapter': 8},
            'selected_prefix_execution40_each': True, 'episodes_each_action': 10,
            'privileged67': True, 'all_actions_still_neural': True,
            'geometry_or_contact_based_action_override': False,
            'physics_robot_gripper_camera_thresholds_and_horizons_changed': False,
            'reserved_final_test_used': False, 'SR_guarantee': None,
            'existing_Adapter_fit_NOT_interrupted': True,
            'extra_X20k_owner_will_wait_status_bridge_and_read_X_prefix_selection': True,
            'code': {str(p.resolve()): sha256_file(p) for p in
                (Path(__file__), ROOT/'VLA/evaluate_native_chunks_r45.py')}})
        stage = 'waiting_actual_Adapter_fit_completion_NO_preemption'
        status(); report()
        until = time.monotonic()+18*3600
        while True:
            fit = read(BASE/'balanced_dispatch_r49/status.json')
            if fit['status'].startswith(('completed', 'failed')): break
            if time.monotonic() > until: raise TimeoutError('Actual fit not terminal before18h')
            time.sleep(20); status()
        for family, port, native in (('xvla', 5847, 30), ('vla_adapter', 5848, 8)):
            result = {'family': family, 'status': 'not_started'}
            log = None
            try:
                model = ROOT/'VLA'/family/'privileged_r44_bounded67_20261016/test1'
                if read(model/'train/status.json')['status'] != 'completed':
                    raise ValueError('Own long fit not completed; do not substitute a gate or base model')
                cp = Path(read(model/'train/latest_checkpoint.json')['checkpoint'])
                from VLA.stable_native_r44.conditioning import require_contract
                require_contract(cp)
                from VLA.new_vlas.train import wait_gpu
                stage = family+'_waiting_exclusive_GPU'
                status(); wait_gpu(args.output)
                python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
                log = (args.output/(family+'_server.log')).open('x')
                server = subprocess.Popen([str(python), '-m', 'VLA.stable_native_r44.server',
                    '--checkpoint', str(cp), '--port', str(port)], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
                stage = family+'_loading_own_native_server'; status()
                atomic_json(args.output/(family+'_server_owner.json'), {'pid': server.pid, 'checkpoint': str(cp), 'port': port})
                until = time.monotonic()+1200
                while True:
                    if server.poll() is not None: raise RuntimeError('Own native server exited')
                    try:
                        response = requests.get(f'http://127.0.0.1:{port}/health', timeout=3)
                        response.raise_for_status(); metadata = response.json()
                        saved = read(cp/'saved_values.json')
                        if (metadata['checkpoint'] != str(cp.resolve()) or metadata['family'] != family
                                or metadata['checkpoint_master_sha256'] != saved['trainable_master_sha256']):
                            raise ValueError('Own actual checkpoint/server identity differs')
                        break
                    except requests.RequestException:
                        if time.monotonic() > until: raise TimeoutError('Own native server not ready')
                        time.sleep(3)
                candidates, errors = [], []
                if family == 'xvla':
                    for prefix, name in ((8, 'xvla7000_horizon8_pilot8'), (30, 'xvla7000_native_chunk8')):
                        path = BASE/name/'progress.json'
                        candidates.append(candidate(read(path), checkpoint=cp, prefix=prefix, evidence=path))
                for prefix in ([4, 6] if family == 'xvla' else [8, 4, 6]):
                    folder = args.output/f'{family}_prefix{prefix}_pilot8'
                    try:
                        p = evaluate(family, port, prefix, 2, folder, f'{family}_prefix{prefix}_matched8')
                        candidates.append(candidate(p, checkpoint=cp, prefix=prefix, evidence=folder/'progress.json'))
                    except Exception as error:
                        errors.append({'prefix': prefix, 'status': 'integration_error_NOT_fake_zero_SR',
                            'error': f'{type(error).__name__}: {error}'})
                    atomic_json(args.output/(family+'_pilot_candidates.json'), {'candidates': candidates, 'errors': errors})
                selected = choose(candidates, native)
                selection = {'rule': 'prespecified_matched8_development_selection_NOT_final_test',
                    'candidates': candidates, 'errors': errors, 'selected': selected, 'reserved_final_used': False}
                atomic_json(args.output/(family+'_prefix_selection.json'), selection)
                # Explicit compatibility interface for the already-running
                # finite X20k owner; it will use the selected prefix afterward.
                if family == 'xvla': atomic_json(BRIDGE/'xvla_prefix_selection.json', selection)
                folder = args.output/(family+'_development40')
                p = evaluate(family, port, selected['execution_steps'], 10, folder, family+'_selected_development40')
                result.update(status='completed40', execution_steps=selected['execution_steps'],
                    successes=p['successes'], trials=40, progress=str(folder/'progress.json'))
            except Exception as error:
                result.update(status='failed_integration_or_fit_NOT_fake_zero_SR', error=f'{type(error).__name__}: {error}')
            finally:
                if server is not None and server.poll() is None:
                    server.terminate()
                    try: server.wait(timeout=30)
                    except subprocess.TimeoutExpired: server.kill(); server.wait(timeout=10)
                server = None
                if log is not None: log.close()
            results.append(result); status(); report()
        stage = 'matched_prefix_sweeps_and_selected40_each_complete'
        status('completed' if all(r['status'] == 'completed40' for r in results) else 'completed_with_errors_retained')
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
            raise FileExistsError('Do not duplicate feedback experiment owner')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            process = subprocess.Popen([str(SIM), '-m', 'VLA.run_chunk_feedback_r50', '--output', str(args.output.resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': process.pid})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}))
    else:
        main(args)
