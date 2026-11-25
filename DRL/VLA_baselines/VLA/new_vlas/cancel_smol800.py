"""User-requested cancellation, preserving the active seed's terminal evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.wait_render_then_resume_queues import identity

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'VLA/audits/scene67_deadline_r31_20261015'
OUT = ROOT/'VLA/audits/new_vlas_r43_20261016/smol800_user_cancellation'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    pointer = read(BASE/'Smol_800_r39_active.json')
    cycle = Path(pointer['cycle'])
    old_status = read(cycle/'status.json')
    owner = identity(pointer['pid'])
    if (not owner or owner['state'] != 'T' or owner['cwd'] != str(ROOT)
            or b'VLA.resume_smol800_r40' not in owner['tokens']):
        raise ValueError('Only the exact already-held800 owner may be cancelled')
    child = identity(old_status['active_child_pid']) if old_status.get('active_child_pid') else None
    if child and (child['cwd'] != str(ROOT) or b'VLA.resume_smol800_r40' not in child['tokens']
                  or str(cycle).encode() not in child['tokens']):
        raise ValueError('Unknown active child; no process is signalled')
    snapshots = {}
    waiters = [
        (921947, BASE/'Smol_GTcrop_motion_r36_priority_r38_v2', 'VLA.run_smol_priority_r38'),
        (724018, BASE/'all_family_privileged_r33_v4', 'VLA.run_all_privileged_r33'),
        (724050, BASE/'openpi_EEF_repair_r33_v4', 'VLA.run_openpi_eef_repair_r33'),
        (908576, BASE/'OpenVLA_balanced_recovery_r34_v4', 'VLA.run_openvla_balanced_recovery_r34')]
    for pid, directory, module in waiters:
        proc = identity(pid)
        if not proc or proc['state'] != 'T' or module.encode() not in proc['tokens']:
            raise ValueError('Only verified stopped old waiters may be recorded')
        state = read(directory/'status.json')
        snapshots[str(pid)] = {'start_ticks': proc['start_ticks'], 'cycle': str(directory),
            'module': module, 'previous_status': state}
        atomic_json(directory/'status.json', {**state, 'status': 'paused_by_user_new_VLA_priority',
            'user_new_VLA_priority_receipt': str(OUT/'old_waiters_held.json'),
            'trainer_interrupted': False, 'paused_at_utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(OUT/'old_waiters_held.json', {'waiters': snapshots,
        'user_requested_new_VLA_priority': True, 'no_actual_trainer_signalled': True,
        'existing_checkpoints_results_and_code_preserved': True})
    atomic_json(OUT/'original_status.json', old_status)
    original_progress = read(cycle/'progress.json')
    atomic_json(OUT/'original_progress.json', original_progress)
    until = time.monotonic()+600
    while child:
        current = identity(old_status['active_child_pid'])
        if not current or current['state'] == 'Z':
            break
        if current['start_ticks'] != child['start_ticks']:
            raise ValueError('Child PID identity changed; do not touch it')
        atomic_json(OUT/'status.json', {'status': 'finishing_owned_inflight_seed_no_new_seed',
            'pid': os.getpid(), 'coordinator_held': pointer['pid'],
            'child_preserved': old_status['active_child_pid'], 'planned800_cancelled': True})
        if time.monotonic() > until:
            raise TimeoutError('Preserved inflight child still running; coordinator remains held')
        time.sleep(5)
    rows = list(original_progress['rows'])
    seen = {(r['seed'], r['task']) for r in rows}
    if old_status.get('active_seed'):
        result = cycle/f'test_seed{old_status["active_seed"]}'/'seed_result.json'
        if not result.is_file():
            raise ValueError('Active worker exited without terminal evidence; preserve original prefix')
        for row in read(result)['rows']:
            if (row['seed'], row['task']) not in seen:
                rows.append(row)
                seen.add((row['seed'], row['task']))
    plan = read(cycle/'plan.json')
    expected = [(r['seed'], r['task']['name']) for r in plan['trials']]
    if [(r['seed'], r['task']) for r in rows] != expected[:len(rows)]:
        raise ValueError('Cancelled outcomes must remain the original ordered prefix')
    for row in rows:
        trace = read(row['trajectory'])[row['task']]
        if len(trace) != row['steps'] or bool(trace[-1]['success']) != row['success']:
            raise ValueError('Terminal evidence mismatch')
    current_owner = identity(pointer['pid'])
    if not current_owner or current_owner['state'] != 'T' or current_owner['start_ticks'] != owner['start_ticks']:
        raise ValueError('Paused coordinator identity changed')
    # SIGTERM is delivered when CONT releases this exact stopped coordinator;
    # it has no live neural child left and cannot launch another seed first.
    os.kill(pointer['pid'], signal.SIGTERM)
    os.kill(pointer['pid'], signal.SIGCONT)
    for _ in range(100):
        now = identity(pointer['pid'])
        if not now or now['state'] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Coordinator did not terminate; evidence still preserved')
    final = {**original_progress, 'rows': rows, 'completed_trials': len(rows),
        'successes': sum(r['success'] for r in rows), 'status': 'cancelled_by_user_preserve_partial',
        'user_cancelled_target800': True, 'unfinished_NOT_failures': 800-len(rows),
        'final_800_SR_established': False,
        'by_action': {task: {'trials': sum(r['task'] == task for r in rows),
            'successes': sum(r['task'] == task and r['success'] for r in rows)}
            for task in ('touch', 'pick', 'move', 'pick_place')}}
    # Existing old confidence summaries referred to the pre-cancellation prefix.
    final.pop('statistics_complete_seed_blocks', None)
    atomic_json(OUT/'progress.json', final)
    atomic_json(cycle/'progress.json', final)
    atomic_json(cycle/'status.json', {**old_status,
        'status': 'cancelled_by_user_preserve_partial', 'stage': 'user_priority_new_VLAs',
        'completed_trials': len(rows), 'active_child_pid': None, 'active_seed': None,
        'cancellation_receipt': str(OUT/'status.json'),
        'updated_at_utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(BASE/'Smol_800_r39_active.json', {**pointer,
        'status': 'cancelled_by_user', 'completed_trials': len(rows),
        'cancellation_receipt': str(OUT/'status.json')})
    atomic_json(OUT/'status.json', {'status': 'cancelled_by_user_complete', 'pid': os.getpid(),
        'completed_trials': len(rows), 'successes': final['successes'],
        'unfinished_NOT_failures': 800-len(rows), 'no_actual_trainer_signalled': True,
        'active_child_allowed_to_finish': True, 'old_waiters_held_for_new_VLAs': True,
        'weights_or_source_trajectories_deleted': False,
        'preserved_progress_sha256': sha256_file(OUT/'progress.json')})
    print(json.dumps(read(OUT/'status.json'), indent=2), flush=True)


if __name__ == '__main__':
    main()
