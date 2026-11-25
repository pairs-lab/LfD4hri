"""Cancel only R65's queued pi0-FAST, without changing bound code or JEPA.

R65 serializes JEPA then FAST but has no cancellation API. A detached guard
waits for its FAST phase, freezes that owner, verifies its process identity
and FAST-only children, then stops those processes. JEPA is never signalled.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OWNER_ROOT = ROOT / 'VLA/audits/jepa_fast800_r65_20261018'
OUT = ROOT / 'VLA/audits/pi0_fast_cancel_reference_20261019'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def process(pid):
    try:
        root = Path('/proc') / str(int(pid))
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'ppid': int(fields[1]), 'start_ticks': fields[19],
                'state': fields[0], 'argv': [a.decode(errors='replace') for a in (root / 'cmdline').read_bytes().split(b'\0') if a]}
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None


def same_process(expected):
    actual = process(expected['pid'])
    return actual if actual and actual['start_ticks'] == expected['start_ticks'] and actual['argv'] == expected['argv'] else None


def module(argv):
    return argv[argv.index('-m') + 1] if '-m' in argv and argv.index('-m') + 1 < len(argv) else None


def is_fast_stage(stage):
    return isinstance(stage, str) and stage.startswith('pi0_fast_')


def is_fast_child(info):
    argv = info['argv']
    name = module(argv)
    if name == 'VLA.extension_r65.train_fast':
        return True
    return (name in ('VLA.extension_r65.audit_inputs', 'VLA.extension_r65.server', 'VLA.extension_r65.evaluate')
            and '--model' in argv and argv[argv.index('--model') + 1:argv.index('--model') + 2] == ['pi0_fast'])


def children(parent):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():
            continue
        info = process(int(p.name))
        if info and info['ppid'] == parent:
            found.append(info)
    return found


def descendants(roots):
    result = list(roots)
    todo = list(roots)
    while todo:
        parent = todo.pop()
        new = children(parent['pid'])
        result.extend(new)
        todo.extend(new)
    return result


def send(info, sig):
    if same_process(info):
        os.kill(info['pid'], sig)


def stop_fast_phase(owner, owner_root, out):
    if not same_process(owner):
        return False
    send(owner, signal.SIGSTOP)
    resumed = False
    try:
        for _ in range(50):
            paused = same_process(owner)
            if not paused or paused['state'] in ('T', 't'):
                break
            time.sleep(.02)
        else:
            raise RuntimeError('Owner did not stop; refuse unsafe process-tree mutation')
        status = read(owner_root / 'owner_status.json')
        if not is_fast_stage(status['stage']):
            raise RuntimeError('Owner no longer in FAST phase; do not interrupt other models')
        direct = children(owner['pid'])
        if not all(is_fast_child(p) for p in direct):
            raise RuntimeError('Non-FAST owner child detected; all processes preserved')
        targets = descendants(direct)
        for info in targets:
            send(info, signal.SIGTERM)
        send(owner, signal.SIGTERM)
        send(owner, signal.SIGCONT)
        resumed = True
        until = time.monotonic() + 15
        while time.monotonic() < until and any(same_process(p) for p in [owner, *targets]):
            time.sleep(.2)
        for info in [*targets, owner]:
            if same_process(info):
                send(info, signal.SIGKILL)
        time.sleep(.2)
        if any(same_process(p) for p in [owner, *targets]):
            raise RuntimeError('Cancelled processes still present; inspect cancellation evidence')
        evidence = {'status': 'USER_CANCELLED_NOT_EVALUATED_NOT_ZERO_SR', 'UTC': utc(),
                    'owner_at_FAST_phase': owner, 'FAST_processes_stopped': targets,
                    'JEPA_was_not_signalled': True, 'existing_artifacts_deleted': False,
                    'model_failure_or_zero_SR_claimed': False}
        atomic_json(out / 'CANCEL_COMPLETED.json', evidence)
        folder = owner_root / 'pi0_fast'
        folder.mkdir(exist_ok=True)
        atomic_json(folder / 'USER_CANCELLED.json', evidence)
        status.update(status='finished_with_pi0_FAST_cancelled_by_user',
                      stage='pi0_FAST_user_cancelled_after_JEPA_no_remaining_FAST_launch',
                      active_child_pid=None, active_server_pid=None, UTC=utc())
        status.setdefault('models', {})['pi0_fast'] = {
            'status': 'user_cancelled_NOT_evaluated_NOT_model_failure', 'completed800': 0,
            'request': str(owner_root / 'USER_CANCEL_PI0_FAST.json')}
        atomic_json(owner_root / 'owner_status.json', status)
        from VLA.extension_r65.results import refresh
        refresh(owner_root, status)
        report = owner_root / 'REPORT.md'
        lines = report.read_text().splitlines()
        lines = ['|pi0_fast|USER CANCELLED|not measured|not measured|not measured|not measured|'
                 if line.startswith('|pi0_fast|') else line for line in lines]
        report.write_text('\n'.join(lines) + '\n\npi0-FAST cancelled explicitly by the user; '
                          'not a model failure or a measured 0% SR. See USER_CANCEL_PI0_FAST.json.\n')
        return True
    finally:
        if not resumed and same_process(owner):
            send(owner, signal.SIGCONT)


def request(owner_root, out):
    status = read(owner_root / 'owner_status.json')
    owner = process(status['pid'])
    if not owner or module(owner['argv']) != 'VLA.extension_r65.run':
        raise ValueError('Expected live R65 owner absent; do not target unknown processes')
    argv = owner['argv']
    if '--output' not in argv or Path(argv[argv.index('--output') + 1]).resolve() != owner_root.resolve():
        raise ValueError('Wrong owner output directory')
    marker = owner_root / 'USER_CANCEL_PI0_FAST.json'
    if marker.exists():
        raise FileExistsError('Existing user request preserved')
    out.mkdir(parents=True, exist_ok=True)
    value = {'model': 'pi0_fast', 'user_requested_cancel': True, 'UTC': utc(),
             'owner_root': str(owner_root.resolve()), 'owner_identity': owner,
             'stage_at_request': status['stage'], 'do_not_interrupt_JEPA': True,
             'artifacts_to_delete': [], 'FAST_observed_evaluation_cohorts_at_request': 0,
             'guard_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'control_output': str(out.resolve())}
    atomic_json(marker, value)
    atomic_json(out / 'CANCEL_REQUEST.json', value)
    with (out / 'cancel_guard.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'VLA.cancel_fast_r66', 'watch',
                                  '--owner-root', str(owner_root.resolve()), '--output', str(out.resolve())],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    atomic_json(out / 'GUARD_LAUNCH.json', {'guard_pid': child.pid, 'UTC': utc()})
    print(json.dumps({'guard_pid': child.pid, 'current_stage': status['stage'], 'request': str(marker)}), flush=True)


def watch(owner_root, out):
    with (out / '.cancel_guard.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        req = read(out / 'CANCEL_REQUEST.json')
        owner = req['owner_identity']
        while True:
            current = same_process(owner)
            state = read(owner_root / 'owner_status.json')
            if not current:
                value = {'status': 'owner_exited_no_future_FAST_launch_from_this_owner',
                         'UTC': utc(), 'owner_identity': owner}
                atomic_json(out / 'GUARD_STATUS.json', value)
                return
            if is_fast_stage(state.get('stage')):
                if stop_fast_phase(owner, owner_root, out):
                    atomic_json(out / 'GUARD_STATUS.json', {'status': 'completed_user_cancelled', 'UTC': utc()})
                    return
            atomic_json(out / 'GUARD_STATUS.json', {'status': 'armed_queued_FAST_blocked_JEPA_untouched',
                'guard_pid': os.getpid(), 'owner_pid': owner['pid'], 'current_owner_stage': state.get('stage'),
                'no_current_FAST_job': not is_fast_stage(state.get('stage')), 'UTC': utc()})
            time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['request', 'watch'])
    parser.add_argument('--owner-root', type=Path, default=OWNER_ROOT)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    request(args.owner_root, args.output) if args.mode == 'request' else watch(args.owner_root, args.output)
