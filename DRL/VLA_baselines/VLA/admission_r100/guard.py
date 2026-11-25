"""Guard only owned new workers; preserve the active R99 neural benchmark."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import ROOT,OUT,R99,ENV,read,atomic_json,utc
from VLA.continuous_r86.owner import identity,alive


def resources():
    fields={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    free,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(','))
    return {'RAM_available_MiB':fields['MemAvailable']/1024,'GPU_free_MiB':free,'GPU_total_MiB':total}


def admitted_fraction(active,r,args):
    if not active:return .86
    # Full FALCON reached the original5.76GiB allocation cap while >4GiB
    # remained free. Its measured native ESM needs more than that cap. Admit
    # at most46% AND retain2.8GiB headroom, without removing the2.2GiB live
    # preservation guard. Seer/GR00T keep their already measured37% limits.
    ceiling=.46 if 'falcon_esm' in args else .37
    return min(ceiling,max(0.,(r['GPU_free_MiB']-2800)/r['GPU_total_MiB']))


def r99_live():
    path=R99/'status.json'
    if not path.exists():return False
    value=read(path)
    return value.get('status')=='running' and alive(value.get('pid'))


def same_owned_process(expected):
    """PID/start-time/argv are stable; PPID may change after parent death."""
    if not expected or not alive(expected['pid']):return False
    actual=identity(expected['pid'])
    if not actual or any(actual.get(k)!=expected.get(k) for k in ('pid','start_ticks','command')):return False
    if actual.get('ppid')==expected.get('ppid'):return True
    # A living original parent plus a changed parent is not an admitted
    # orphan transition. Never relax the PID/start-time/argv ownership test.
    return bool(expected.get('ppid') and not alive(expected['ppid']))


def checked_stop(expected):
    if not expected or not alive(expected['pid']):return
    if not same_owned_process(expected):raise ValueError('Owned R100 PID identity changed, refusing signal')
    cmd=expected['command']
    if '-m' not in cmd or cmd[cmd.index('-m')+1] not in (
        'VLA.admission_r100.native_gate','VLA.admission_r100.train','VLA.admission_r100.server',
        'VLA.admission_r100.evaluate','VLA.admission_r100.prepare'):
        raise ValueError('Not an approved R100 worker; NEVER signal R99')
    os.kill(expected['pid'],signal.SIGTERM)


def run(python,module,args,label,*,during_R99=True,seconds=3600):
    active=r99_live();r=resources();root=OUT/'guard';root.mkdir(exist_ok=True)
    if (active and not during_R99) or r['RAM_available_MiB']<3500 or r['GPU_free_MiB']<(8000 if active else 13500):
        value={'status':'deferred_until_safe_resources','UTC':utc(),'R99_active':active,'resources':r,
            'NOT_native_model_failure':True}
        atomic_json(root/(label+'.json'),value);return value
    fraction=admitted_fraction(active,r,args)
    command=list(map(str,[python,'-u','-m',module,*args,'--memory-fraction',fraction]))
    log=root/(label+'.log')
    if log.exists():raise FileExistsError('Guard attempt log already exists, use a new attempt label')
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT)
    expected=None
    for _ in range(50):
        expected=identity(child.pid)
        if expected and expected['command']==command:break
        if child.poll() is not None:break
        time.sleep(.02)
    if child.poll() is None and (not expected or expected['command']!=command):
        child.terminate();child.wait(timeout=20);raise ValueError('New native worker identity did not stabilize')
    atomic_json(root/(label+'_IDENTITY.json'),{'identity':expected,'memory_fraction':fraction,'active_R99_at_launch':active})
    started=time.monotonic();reason=None;last=0
    try:
        while child.poll() is None:
            r=resources();now=time.monotonic()
            if r['RAM_available_MiB']<1400 or r['GPU_free_MiB']<2200:
                reason='resource_headroom_below_R99_preservation_guard';break
            if now-started>seconds:reason='finite_native_worker_deadline';break
            if active and r99_live():
                journal=R99/'final800_attempt00/NEURAL_QUERIES.jsonl'
                if journal.exists() and time.time()-journal.stat().st_mtime>60:
                    reason='R99_neural_query_heartbeat_stale_preserve_primary_benchmark';break
            if now-last>5:
                atomic_json(root/(label+'.json'),{'UTC':utc(),'status':'running_owned_native_worker',
                    'pid':child.pid,'resources':r,'elapsed_s':now-started,'R99_active':r99_live()});last=now
            time.sleep(1)
        if reason:
            checked_stop(expected)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if identity(child.pid)==expected:os.kill(child.pid,signal.SIGKILL)
                child.wait(timeout=20)
        value={'UTC':utc(),'status':'stopped_owned_worker_to_preserve_resources' if reason else (
            'completed_native_worker' if child.returncode==0 else 'native_worker_error_requires_diagnosis'),
            'returncode':child.returncode,'reason':reason,'log':str(log),'elapsed_s':time.monotonic()-started,
            'unrelated_processes_signaled':0,'model_SR_NOT_implied':True}
        atomic_json(root/(label+'.json'),value);return value
    finally:
        if child.poll() is None:checked_stop(expected)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--python',type=Path,required=True)
    p.add_argument('--module',default='VLA.admission_r100.native_gate');p.add_argument('--label',required=True)
    p.add_argument('--after-R99',action='store_true');p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    print(run(a.python,a.module,args,a.label,during_R99=not a.after_R99),flush=True)
