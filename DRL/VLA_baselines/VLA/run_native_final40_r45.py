"""After current trainers finish, serial native-queue development40 per VLA.

No GPU preemption, cancelled queue resumption, threshold change or reroll.
X-VLA execution-prefix selection uses the same first8 development scenes for
all available variants. This is hyperparameter selection, NOT final-test SR.
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
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def choose_xvla_prefix():
    candidates = []
    for prefix, path in ((30, BASE/'xvla7000_native_chunk8/progress.json'),
                         (8, BASE/'xvla7000_horizon8_pilot8/progress.json'),
                         (1, ROOT/'VLA/xvla/privileged_r44_bounded67_20261016/test1/development40/progress.json')):
        if path.exists():
            p = read(path)
            subset = [r for r in p['rows'] if r['seed'] in (20268001, 20268002)]
            if len(subset) == 8:
                candidates.append({'execution_steps': prefix, 'successes': sum(r['success'] for r in subset),
                    'trials': 8, 'evidence': str(path)})
    if not candidates:
        return 30, {'rule': 'no_complete_matched8_variants_default_official30', 'candidates': []}
    preference = {30: 2, 8: 1, 1: 0}
    selected = max(candidates, key=lambda c: (c['successes'], preference[c['execution_steps']]))
    return selected['execution_steps'], {'rule': 'best_success_count_on_identical8_DEVELOPMENT_scenes_tie30_then8_then1',
        'candidates': candidates, 'selected': selected, 'reserved_final_used': False}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    stage, child, results = 'waiting_current_bounded_training_pipeline_NO_preemption', None, []
    def status(state='running', **extra):
        atomic_json(args.output/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'active_child_pid': None if child is None else child.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results, **extra})
    def report():
        lines = ['# R45 — native VLA execution development40', '',
            'Same frozen PyBullet robot/gripper/cameras/success. Privileged67. No oracle policy.',
            'Development and execution-hyperparameter selection; NOT reserved final test.', '',
            '| Model | Prefix | touch | pick | move | pick_place | Overall |', '|---|---:|---:|---:|---:|---:|---:|']
        for result in results:
            if result['status'] != 'completed40':
                lines += ['', f"{result['family']}: {result['status']}; {result.get('error', '')}"]
                continue
            p = read(result['progress'])
            cells = []
            for task in ('touch', 'pick', 'move', 'pick_place'):
                subset = [r for r in p['rows'] if r['task'] == task]
                cells.append(f"{sum(r['success'] for r in subset)}/{len(subset)}")
            lines.append(f"| {result['family']} | {result['execution_steps']} | "+' | '.join(cells)+f" | {p['successes']}/40 |")
        lines += ['', 'Primary place success means target-region reaching, not necessarily release/stable placement.',
            'Every declared completed success/failure retained. Unrun or integration-error trials are not manufactured failures.']
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    try:
        atomic_json(args.output/'plan.json', {'scope': 'native_queue40_per_new_VLA_after_current_trainers',
            'families': ['xvla', 'vla_adapter'], 'episodes_each': 40, 'episodes_each_action': 10,
            'wait_current_pipeline': str(BASE/'bounded_native_pipeline_v2/status.json'),
            'XVLA_prefix_selection': 'identical8_development_seeds_count_successes_tie30_then8_then1',
            'Adapter_prefix': 8, 'native_checkpoint_weights_not_modified': True,
            'same_motor_Hz_horizons_safety_camera_and_scoring': True, 'reserved_test_used': False,
            'old_queues_or_Smol800_resumed': False, 'success_guarantee': None})
        until = time.monotonic()+18*3600
        status(); report()
        while True:
            p = read(BASE/'bounded_native_pipeline_v2/status.json')
            if p['status'].startswith(('completed', 'failed')): break
            if time.monotonic() > until: raise TimeoutError('Current owned pipeline did not reach terminal state')
            time.sleep(30)
        prefix, selection = choose_xvla_prefix()
        atomic_json(args.output/'xvla_prefix_selection.json', selection)
        for family, steps, port in (('xvla', prefix, 5847), ('vla_adapter', 8, 5848)):
            root = ROOT/'VLA'/family/'privileged_r44_bounded67_20261016/test1'
            checkpoint = root/'train/latest_checkpoint.json'
            result = {'family': family, 'execution_steps': steps, 'status': 'not_started'}
            try:
                if not checkpoint.exists() or read(root/'train/status.json')['status'] != 'completed':
                    raise ValueError('No own completed target-domain fit checkpoint; do not substitute a gate/base model')
                cp = Path(read(checkpoint)['checkpoint'])
                folder = args.output/(family+'_development40')
                stage = family+'_native_queue_development40'
                with (args.output/(family+'.log')).open('x') as log:
                    child = subprocess.Popen([str(SIM), '-m', 'VLA.evaluate_native_chunks_r45', '--family', family,
                        '--port', str(port), '--seed-count', '10', '--execution-steps', str(steps),
                        '--own-server-checkpoint', str(cp), '--output', str(folder)], cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
                    status()
                    code = child.wait()
                child = None
                if code: raise RuntimeError(f'Native queue evaluation exited{code}; preserve all partial trials')
                p = read(folder/'progress.json')
                if p['completed_trials'] != 40: raise ValueError('Incomplete40')
                result.update(status='completed40', successes=p['successes'], trials=40, progress=str(folder/'progress.json'))
            except Exception as error:
                result.update(status='failed_integration_or_fit_NOT_fake_zero_SR', error=f'{type(error).__name__}: {error}')
            results.append(result); status(); report()
        stage = 'native_development40_each_finished_review_actual_rates'
        status('completed' if all(r['status'] == 'completed40' for r in results) else 'completed_with_errors_retained')
    except Exception as error:
        status('failed_preserve_evidence', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.output.exists() or args.output.with_name(args.output.name+'_owner.json').exists():
            raise FileExistsError('Do not duplicate an evaluation owner')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            process = subprocess.Popen([str(SIM), '-m', 'VLA.run_native_final40_r45', '--output', str(args.output.resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': process.pid})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}))
    else:
        main(args)
