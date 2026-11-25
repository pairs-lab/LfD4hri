"""Bounded, sequential recording queue; no new neural weights or hidden trials.

Starts only after actual success+failure and shared-world paired media pilots
have passed. Resource guard defers to the existing training/inference jobs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.artifact_io import atomic_json

CASES = [
    (20268002, 'touch'), (20268001, 'move'), (20268005, 'pick_place'),
    (20268003, 'pick'), (20268008, 'pick'), (20268003, 'move'),
    (20268004, 'pick_place'), (20268001, 'touch'), (20268004, 'pick'),
]


def resources():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            values['available_RAM_MB'] = int(line.split()[1])/1024
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True)
    values['available_GPU_MB'] = float(result.stdout.splitlines()[0]) if result.returncode == 0 else 0.
    return values


def run(args):
    out, base = args.output.resolve(), args.base.resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'plan.json', {'candidates': [{'seed': s, 'task': t} for s, t in CASES],
        'bounded_new_candidates': len(CASES), 'already_recorded_candidates': 2,
        'maximum_total_candidates_this_queue': 11, 'source': 'fixed completed native10_queue2 neural traces',
        'no_new_NN_inference_or_finetune': True, 'no_rerolling_or_action_repairs': True,
        'Oracle_controller_explicitly_adapted_to_frozen_VLA_physical_world': True,
        'staged_Oracle_faults': False, 'no_promised_category_counts_before_observed_results': True})
    rows = []
    def publish(status, **kw):
        atomic_json(out/'status.json', {'status': status, 'pid': os.getpid(), 'rows': rows, **kw})
    paired = base/'01_RL_succeed_VLA_succeed/scene02_pick_peach_v1'
    while not (paired/'status.json').exists() or not json.loads((paired/'status.json').read_text())['status'].startswith('paired_media_audited'):
        publish('waiting_for_actual_paired_pilot_not_just_schedule')
        time.sleep(10)
    gate_paths = [base/'pilots/pick_seed20268001_v4_isolated/camera_audit.json',
        base/'pilots/place_failure_seed20268001_v1_isolated/camera_audit.json',
        paired/'SmolVLA/camera_audit.json', paired/'Oracle/camera_audit.json']
    if not all(json.loads(path.read_text())['media_decode_sync_pass'] for path in gate_paths):
        raise ValueError('Actual success/failure/paired pilot media gate did not pass')
    atomic_json(out/'passed_pilot_gate.json', {'actual_gate_paths': [str(p) for p in gate_paths], 'pass': True})
    def command(module, options, log_name):
        while True:
            available = resources()
            if available['available_RAM_MB'] >= 2000 and available['available_GPU_MB'] >= 800:
                break
            publish('waiting_for_resources_training_not_interrupted', resources=available)
            time.sleep(10)
        with (out/log_name).open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', module, *options], stdout=log, stderr=subprocess.STDOUT)
            publish('running', child_pid=proc.pid, module=module, resources=available)
            while proc.poll() is None:
                time.sleep(5)
            return proc.returncode
    for seed, task in CASES:
        candidate = f'{task}_seed{seed}'
        vla = out/candidate/'Smol_capture'
        opts = ['--seed', str(seed), '--task', task, '--output', str(vla)]
        code = command('VLA.qualitative_recorder_r51', opts, candidate+'_Smol.log')
        row = {'seed': seed, 'task': task, 'VLA_exit_code': code, 'VLA_source': str(vla)}
        if code:
            row['status'] = 'VLA_gate_failed_preserved_no_Oracle_or_fake_video'
            rows.append(row); publish('running'); continue
        sources = []
        for repeat in range(2):
            oracle = out/candidate/f'Oracle_capture_repeat{repeat}'
            opts = ['--seed', str(seed), '--task', task, '--paired-capture', str(vla), '--output', str(oracle)]
            code = command('VLA.qualitative_oracle_shared_r51', opts, f'{candidate}_Oracle{repeat}.log')
            sources.append(oracle)
            if code:
                break
        row['Oracle_exit_code'] = code
        if not code:
            target = base/'05_backups'/candidate
            opts = ['--oracle', str(sources[0]), '--oracle-repeat', str(sources[1]), '--vla', str(vla), '--output', str(target)]
            code = command('VLA.qualitative_paired_pilot_r51', opts, candidate+'_pair.log')
            row['pair_exit_code'] = code
            row['pair_output'] = str(target)
        row['status'] = 'paired_media_audited_pending_manual_review' if not code else 'gate_failed_preserved'
        rows.append(row); publish('running')
    publish('bounded_queue_completed_not_automatically_full11_category_suite')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    a = parser.parse_args()
    if a.detach:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.with_suffix('.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', 'VLA.qualitative_queue_r51',
                '--base', str(a.base.resolve()), '--output', str(a.output.resolve())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'started_owner_pid': proc.pid, 'output': str(a.output.resolve())}), flush=True)
    else:
        try:
            run(a)
        except Exception as error:
            if a.output.is_dir():
                atomic_json(a.output/'error.json', {'error': f'{type(error).__name__}: {error}', 'not_complete': True})
            raise
