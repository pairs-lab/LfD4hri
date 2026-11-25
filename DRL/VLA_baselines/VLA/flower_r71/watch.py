"""A finite, explicitly conditional TRAIN-only follow-up after final R70 tests.

Selection uses DEVELOPMENT, never final800 or confirmation outcomes. No
original checkpoint/optimizer/controller file is changed or removed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70.common import ROOT, R69, OUT as R70, PY, SIM, ENV, DEVELOPMENT, read, verify, freeze_test_code


def needs_rehearsal(dev):
    if dev['completed_trials'] != 40 or any(v['n'] != 10 for v in dev['per_action'].values()):
        raise ValueError('Do not trigger from a partial pilot')
    return dev['per_action']['pick']['successes'] < 5 or sum(r['success'] for r in dev['rows']) < 20


def candidate_wins(baseline, candidate):
    def totals(p):
        return sum(r['success'] for r in p['rows']), sum(bool(r['safety_violation']) for r in p['rows'])
    b, c = totals(baseline), totals(candidate)
    # No success sacrifice / no extra safety aborts merely to repair one task.
    return (c[0] > b[0] and c[1] <= b[1] and
            candidate['per_action']['pick']['successes'] >= baseline['per_action']['pick']['successes'])


def main(a):
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / '.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'status.json').exists(): raise FileExistsError('Duplicate rehearsal owner')
        bound = freeze_test_code()
        bound.update({str(p.resolve()): sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
        child = server = None; stage = 'wait_original_final_finetune_and_R70_tests'
        def status(state='waiting', **extra):
            atomic_json(out / 'status.json', {'status': state, 'stage': stage, 'pid': os.getpid(),
                'active_child_pid': None if child is None else child.pid,
                'active_server_pid': None if server is None else server.pid, **extra})
        def execute(py, module, args, label):
            nonlocal child
            verify(bound)
            with (out / (label + '.log')).open('x') as stream:
                child = subprocess.Popen([str(py), '-u', '-m', module, *map(str, args)],
                    cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None: status('running'); time.sleep(3)
                code = child.returncode; child = None
            if code: raise RuntimeError(label + f' integration exited{code}, not a model SR result')
        try:
            atomic_json(out / 'PREDECLARATION.json', {
                'conditional_on_final_development_pick_below50percent_OR_total_below50percent': True,
                'exactly_one_rehearsal128_updates_not_indefinite': True,
                'no_original_source_or_checkpoint_edited': True,
                'new_data_collected': False, 'no_drop_or_wrong_object_recovery_claim': True,
                'selection_does_NOT_read_confirmation_or_final800': True,
                'confirmation_seeds_if_selected': list(range(20268201, 20268211)),
                'candidate_selection': 'DEV successes strictly increase; safety aborts do not increase; pick does not decrease',
                'code': bound,
            })
            status(); deadline = time.monotonic() + 14 * 3600
            while True:
                verify(bound)
                state = read(R70 / 'watch_status.json')
                if state['status'] == 'completed': break
                if state['status'] != 'running': raise RuntimeError('R70 owner did not complete cleanly: ' + str(state))
                if time.monotonic() > deadline: raise TimeoutError('Finite14h wait elapsed')
                status(); time.sleep(15)
            dev = read(R70 / 'final_development40/progress.json')
            if not needs_rehearsal(dev):
                stage = 'skipped_rehearsal_development_threshold_met'; status('completed'); return
            # Use neither previous confirmation outcomes nor final800 to decide.
            prefix = int(read(R70 / 'PREFIX_SELECTION.json')['prefix'])
            sampler = read(R70 / 'CONTACT_REHEARSAL_SAMPLER_AUDIT.json')
            if not sampler['passed']: raise ValueError('Actual TRAIN sampler qualification failed')
            import psutil
            deadline = time.monotonic() + 1800
            while True:
                free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True)
                if int(free.strip().splitlines()[0]) >= 10500 and psutil.virtual_memory().available >= 5 * 2**30: break
                if time.monotonic() > deadline: raise TimeoutError('No safe single-trainer resource window')
                status(waiting_resources=True); time.sleep(15)
            stage = 'native128_grasp_transition_rehearsal'; status('running')
            cp = out / 'fit128'
            execute(PY, 'VLA.flower_r71.fit', ['--source', R69 / 'fit4096', '--output', cp], 'fit128')
            stage = 'fresh_process_native_exact_reload'; status('running')
            proof = out / 'FRESH_RELOAD.json'
            execute(PY, 'VLA.flower_r69.reload_fit', ['--checkpoint', cp, '--output', proof], 'fresh_reload')
            if not read(proof)['passed']: raise ValueError('Candidate fresh reload failed')
            stage = 'candidate_development40'; status('running')
            with (out / 'server.log').open('x') as stream:
                server = subprocess.Popen([str(PY), '-u', '-m', 'VLA.flower_r70.server', '--checkpoint', str(cp),
                    '--output', str(out / 'runtime')], cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
                session = requests.Session(); session.trust_env = False; deadline = time.monotonic() + 180
                try:
                    while True:
                        if server.poll() is not None: raise RuntimeError('Candidate own native server exited')
                        try:
                            h = session.get('http://127.0.0.1:5930/health', timeout=3); h.raise_for_status()
                            if h.json()['checkpoint'] != str(cp): raise ValueError('Different checkpoint on port5930')
                            break
                        except requests.RequestException:
                            if time.monotonic() > deadline: raise TimeoutError('Candidate reload startup exceeded180s')
                            status('running'); time.sleep(2)
                    execute(SIM, 'VLA.flower_r70.evaluate', ['--cohort', DEVELOPMENT,
                        '--output', out / 'final_development40', '--prefix', prefix], 'candidate_dev40')
                    new = read(out / 'final_development40/progress.json')
                    wins = candidate_wins(dev, new)
                    atomic_json(out / 'DEVELOPMENT_SELECTION.json', {'candidate_selected': wins,
                        'original_baseline_checkpoint_preserved': True, 'prefix_unchanged': prefix,
                        'original_per_action': dev['per_action'], 'candidate_per_action': new['per_action'],
                        'confirmation_and_final800_NOT_used_for_selection': True})
                    if wins:
                        stage = 'new_independent_confirmation40'; status('running')
                        cohort = out / 'CONFIRMATION40.json'
                        atomic_json(cohort, {'role': 'development_confirmation',
                            'seeds': list(range(20268201, 20268211)),
                            'predeclared_before_candidate_training': True})
                        execute(SIM, 'VLA.flower_r70.evaluate', ['--cohort', cohort,
                            '--output', out / 'final_confirmation40', '--prefix', prefix], 'new_confirmation40')
                finally: session.close()
            stage = 'actual_per_action_report'; status('running')
            execute(SIM, 'VLA.flower_r70.report', ['--output', out], 'measured_report')
            stage = 'completed_single_conditional_rehearsal_NOT800'; status('completed')
        except Exception as error:
            status('integration_error_NOT_model_SR', error=str(error)); raise
        finally:
            for process in (child, server):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'VLA/audits/flower_grasp_rehearsal_r71_20261019')
    p.add_argument('--launch', action='store_true'); a = p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True, exist_ok=True)
        with (a.output / 'owner.log').open('x') as stream:
            child = subprocess.Popen([str(PY), '-u', '-m', 'VLA.flower_r71.watch', '--output', str(a.output.resolve())],
                cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps({'owner_pid': child.pid, 'status': str(a.output / 'status.json')}))
    else: main(a)
