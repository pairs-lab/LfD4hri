"""Finish actual neural robot8 for both families, then release held long-fit queue.

Only exact owned new-model PIDs may be continued. Historical queues stay held.
Any integration/preflight failure leaves long training unstarted for review.
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
from VLA.new_vlas.workflow import N, ROOT, SIM, verify_lineage
from VLA.wait_render_then_resume_queues import identity


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    owner = identity(args.workflow_pid)
    smoke = identity(args.smoke_pid)
    if (not owner or owner['state'] != 'T' or b'VLA.new_vlas.workflow' not in owner['tokens']
            or not smoke or smoke['state'] != 'T' or b'VLA.new_vlas.robot_smoke' not in smoke['tokens']):
        raise ValueError('Only the exact held new-model workflow/smoke may be resumed')
    atomic_json(args.output/'owner_receipt.json', {'workflow_pid': args.workflow_pid,
        'workflow_start_ticks': owner['start_ticks'], 'smoke_pid': args.smoke_pid,
        'smoke_start_ticks': smoke['start_ticks'], 'old_waiters_will_NOT_be_resumed': True})
    stage = 'waiting_final_native_Adapter_reload_and_image_checks'
    def save(state='running', **extra):
        atomic_json(args.output/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra})
    def wait_status(path, accepted):
        until = time.monotonic()+1800
        while True:
            if path.is_file():
                value = json.loads(path.read_text())
                state = value.get('status', '')
                if state in accepted:
                    return value
                if state.startswith('failed'):
                    raise RuntimeError('Actual preflight failed: '+str(value))
            if time.monotonic() > until:
                raise TimeoutError('Own finite preflight did not complete: '+str(path))
            time.sleep(3)
    try:
        save()
        wait_status(N/'adapter_fresh_reload_v3/status.json', {'completed'})
        until = time.monotonic()+600
        while not (N/'adapter_actual_native_images_v3/audit.json').is_file():
            if time.monotonic() > until:
                raise TimeoutError('Final actual image/input-parity audit unavailable')
            time.sleep(3)
        stage = 'actual_XVLA_robot8_original_SIM_NOT_SR'
        current = identity(args.smoke_pid)
        if not current or current['state'] != 'T' or current['start_ticks'] != smoke['start_ticks']:
            raise ValueError('Held smoke identity changed')
        save()
        os.kill(args.smoke_pid, signal.SIGCONT)
        wait_status(N/'xvla_actual_robot8_v1/status.json', {'completed8_native_controls_NOT_SR'})
        stage = 'actual_Adapter_robot8_original_SIM_NOT_SR'
        with (args.output/'adapter_robot8.log').open('x') as log:
            child = subprocess.Popen([str(SIM), '-m', 'VLA.new_vlas.robot_smoke', '--checkpoint',
                str(N/'adapter_optimizer_gate_v3/checkpoints/step-000002'), '--output',
                str(N/'adapter_actual_robot8_v1'), '--port', '5854'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            save(active_child_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError('Adapter actual robot integration failed; no long fit released')
        for family, short in (('xvla', 'xvla'), ('vla_adapter', 'adapter')):
            verify_lineage(N/(short+'_optimizer_gate_v3')/'checkpoints/step-000002')
            proof = json.loads((N/(short+'_actual_robot8_v1')/'audit.json').read_text())
            if not proof['passed'] or proof['neural_controls'] != 8:
                raise ValueError('Own actual neural robot8 incomplete: '+family)
        current = identity(args.workflow_pid)
        if not current or current['state'] != 'T' or current['start_ticks'] != owner['start_ticks']:
            raise ValueError('Held prioritized workflow identity changed')
        stage = 'release_exact_prioritized_new_model_workflow_after_all_checks'
        os.kill(args.workflow_pid, signal.SIGCONT)
        save('completed_released_new_model_finetune_queue', workflow_pid=args.workflow_pid,
            both_actual_neural_robot8_passed=True, old_waiters_resumed=False)
    except Exception as error:
        save('failed_no_long_fit_released_preserve_actual_evidence', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow-pid', type=int, required=True)
    parser.add_argument('--smoke-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
