"""Observe a healthy R88 run; bounded recovery only for technical interruptions.

Never relax input-integrity/success guards or manufacture missing results. Never
kill/restart a healthy benchmark or signal a PID whose identity was not captured.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from VLA.common.provenance import sha256_file
from VLA.dita800_r87.run import ENV,SIM
from .common import (ROOT,OUT,read,atomic_json,utc,identity,alive,complete_rows,bindings,
                     parent_pid,live_children)
from .audit import certify

CURRENT=ROOT/'VLA/audits/receipt_recovery_r88_20261021'
ALLOWED={'VLA.receipt_recovery_r88.run','VLA.receipt_recovery_r88.evaluate',
         'VLA.benchmark_guardian_r89.attempt','VLA.benchmark_guardian_r89.evaluate',
         'VLA.dita_transfer_r84.server','VLA.continuous_r86.owner'}
OWNERS={'VLA.receipt_recovery_r88.run','VLA.benchmark_guardian_r89.attempt'}


def module_name(ident):
    command=ident['command'] if ident else []
    return command[command.index('-m')+1] if '-m' in command and command.index('-m')+1<len(command) else None


def checked_signal(expected,sig):
    if expected is None:raise ValueError('No captured process identity')
    now=identity(expected['pid'])
    if now is None or not alive(expected['pid']):return False
    if now!=expected:raise ValueError('PID identity changed, no signal sent')
    if module_name(expected) not in ALLOWED:raise ValueError('Not an owned benchmark/queue process')
    try:os.kill(expected['pid'],sig)
    except ProcessLookupError:return False
    return True


def capture(folder):
    state=read(folder/'status.json');result={}
    owner=state.get('pid')
    if not owner or not alive(owner):return result
    ident=identity(owner)
    if module_name(ident) not in OWNERS:raise ValueError('Benchmark owner PID no longer identifies the expected module')
    result['pid']=ident
    for key in ('server_pid','evaluator_pid'):
        pid=state.get(key)
        if pid and alive(pid):
            ident=identity(pid)
            if module_name(ident) not in ALLOWED or parent_pid(pid)!=owner:
                if not alive(owner):return result
                raise ValueError('Native child PID is not owned by this benchmark attempt')
            result[key]=ident
    return result


def merge_identities(known,fresh):
    for key,value in fresh.items():
        if key in known and known[key]!=value:raise ValueError('Captured process identity changed inside an attempt')
        known[key]=value


def hold_idle_queue(path):
    q=read(path);ident=identity(q['pid'])
    if (q.get('child_pid') or q.get('server_pid') or not q.get('resource_wait')
        or module_name(ident)!='VLA.continuous_r86.owner' or live_children(q['pid'])):return None
    if not checked_signal(ident,signal.SIGSTOP):return None
    held=False
    try:
        # SIGSTOP acknowledgment precedes the second status/children check.
        for _ in range(50):
            stat=(Path('/proc')/str(q['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]
            if stat in ('T','t'):break
            time.sleep(.02)
        else:raise RuntimeError('Idle queue did not acknowledge a safe pause')
        fresh=read(path)
        if fresh.get('child_pid') or fresh.get('server_pid') or live_children(q['pid']):return None
        held=True;return ident
    finally:
        if not held:checked_signal(ident,signal.SIGCONT)


def stop_captured(processes):
    # Only terminate owned interrupted/stalled attempts; never a healthy one.
    for key in ('pid','evaluator_pid','server_pid'):
        ident=processes.get(key)
        if ident and alive(ident['pid']):checked_signal(ident,signal.SIGTERM)
    limit=time.monotonic()+30
    while any(alive(v['pid']) for v in processes.values()) and time.monotonic()<limit:time.sleep(1)
    for ident in processes.values():
        if alive(ident['pid']):checked_signal(ident,signal.SIGKILL)


def retryable(folder,owner_missing=False,stalled=False):
    if any((folder/n).exists() for n in ('RECEIPT_MISMATCH.json','PREFIX_DIVERGENCE.json')):return False
    texts=[]
    for name in ('ERROR.json','evaluate.log','server.log'):
        p=folder/name
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-16000));texts.append(f.read().decode(errors='replace'))
    text='\n'.join(texts)
    fatal=('Incorrect actual model/input receipt','prefix differs','prefix does not match','State-restore',
           'Restored pre-action','Restored post-action','Nonfinite','Malformed','ValueError:',
           'TypeError:','KeyError:','AttributeError:','AssertionError:','source changed','hash changed')
    if any(t in text for t in fatal):return False
    transient=('ConnectionError','ReadTimeout','Connection reset','RemoteDisconnected','out of memory',
               'Connection refused','TimeoutError','Killed','BrokenPipeError')
    return owner_missing or stalled or any(t in text for t in transient)


def snapshot(folder,index):
    path=folder/'final800/progress.json';data,rows,keys=complete_rows(path)
    saved={**data,'original_progress_path':str(path),'original_progress_sha256':sha256_file(path)}
    target=OUT/f'resume_snapshot_{index:02d}.json';atomic_json(target,saved);return target


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.guardian.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'STARTED.json').exists():raise FileExistsError('No duplicate guardian')
        qualification=read(OUT/'PREFLIGHT.json')
        if not qualification.get('passed') or qualification.get('code')!=bindings():
            raise ValueError('Current guardian code has not passed CPU-only preflight qualification')
        atomic_json(OUT/'STARTED.json',{'UTC':utc(),'pid':os.getpid(),'code':bindings()})
        paused=None;folder=CURRENT;attempts=0;known={};deadline=time.monotonic()+48*3600
        source=None;frozen_code=bindings();child=None
        def status(kind,**extra):
            atomic_json(OUT/'status.json',{'UTC':utc(),'status':kind,'pid':os.getpid(),
                'active_attempt':str(folder),'technical_restarts':attempts,**extra})
        try:
            # Only hold the idle owner between training phases, not any live
            # optimizer/evaluator. It automatically resumes after completion/error.
            queue=ROOT/'VLA/audits/continuous_r86_20261021/LIVE.json'
            if queue.exists():
                paused=hold_idle_queue(queue)
                if paused:
                    atomic_json(OUT/'QUEUE_HOLD.json',{'identity':paused,'UTC':utc(),
                        'no_optimizer_or_episode_interrupted':True,'resume_automatically_after800':True})
            while True:
                if time.monotonic()>deadline:raise TimeoutError('48h guardian budget, not a claim of model failure/completion')
                if bindings()!=frozen_code:raise ValueError('Guardian source changed while monitoring')
                if child:child.poll()  # reap an ended recovery owner rather than leave a zombie
                state=read(folder/'status.json') if (folder/'status.json').exists() else {}
                merge_identities(known,capture(folder) if state else {})
                p=folder/'final800/progress.json'
                progress=read(p) if p.exists() else None
                if progress and progress['status']=='completed_actual_final800':
                    status('independent_final800_audit',completed_trials=800)
                    result=certify(p);atomic_json(OUT/'CERTIFIED_RESULT.json',result)
                    text='# Certified Dita1024 —800 actual trials\n\n| Action | Successes / n | SR |\n|---|---:|---:|\n'
                    for t,v in result['per_action'].items():text+=f"| {t} | {v['successes']}/{v['n']} | {v['SR_percent']:.2f}% |\n"
                    text+='\nOne actual cohort800,200/action. Not five measured seeds. All failures retained.\n'
                    (OUT/'REPORT.md').write_text(text)
                    status('completed_and_independently_certified800',per_action=result['per_action']);return
                owner=known.get('pid');owner_pid=owner['pid'] if owner else state.get('pid')
                missing=bool((owner_pid and not alive(owner_pid)) or (child and child.poll() is not None))
                stage=state.get('stage','')
                signal_files=[folder/'final800/NEURAL_QUERIES.jsonl',p]
                times=[x.stat().st_mtime for x in signal_files if x.exists()]
                stalled=stage=='actual_frozen800_no_video' and bool(times) and time.time()-max(times)>900
                failed=(state.get('status','').startswith('integration_error') or
                        (progress or {}).get('status','').startswith('integration_error'))
                if state.get('status')=='completed':raise ValueError('Owner reports completion without an actual complete800 ledger')
                if missing or stalled or failed:
                    if not retryable(folder,owner_missing=missing,stalled=stalled):
                        raise RuntimeError('Integrity/deterministic code error requires diagnosis; no guards bypassed')
                    if attempts>=3:raise RuntimeError('Three technical recoveries exhausted; evidence retained, no invented completion')
                    status('recovering_interrupted_attempt',completed_trials=progress.get('completed_trials') if progress else None)
                    stop_captured(known)
                    attempts+=1
                    if p.exists():source=snapshot(folder,attempts)
                    elif source is None:raise RuntimeError('No committed outcome ledger to resume safely')
                    atomic_json(OUT/f'INTERRUPTION_{attempts:02d}.json',{'UTC':utc(),'attempt':str(folder),
                        'missing':missing,'stalled':stalled,'failed':failed,'snapshot':str(source)})
                    folder=OUT/f'attempt{attempts:02d}';known={}
                    with (OUT/f'attempt{attempts:02d}.log').open('x') as log:
                        child=subprocess.Popen([str(SIM),'-u','-m','VLA.benchmark_guardian_r89.attempt',
                            '--output',str(folder),'--resume-progress',str(source),'--port',str(5992+attempts)],
                            cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True)
                    ident=identity(child.pid)
                    if ident and module_name(ident)=='VLA.benchmark_guardian_r89.attempt':known['pid']=ident
                status('monitoring_actual_frozen800',completed_trials=progress.get('completed_trials') if progress else None,
                    per_action=progress.get('per_action') if progress else None,
                    last_io_age_s=time.time()-max(times) if times else None)
                time.sleep(15)
        except BaseException as e:
            atomic_json(OUT/'ALERT.json',{'UTC':utc(),'error':str(e),'traceback':traceback.format_exc(),
                'active_attempt':str(folder),'NOT_model_SR_or_completed800':True})
            status('needs_diagnosis_NOT_completed800',error=str(e));raise
        finally:
            if paused and alive(paused['pid']):
                checked_signal(paused,signal.SIGCONT)
                atomic_json(OUT/'QUEUE_RESUMED.json',{'UTC':utc(),'identity':paused})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'guardian.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-u','-m','VLA.benchmark_guardian_r89.guardian'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'guardian_pid':child.pid,'status':str(OUT/'status.json')})
    else:main()
