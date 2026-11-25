"""Native refinement -> measured DEV/CONF -> frozen new actual800.

R98's already-tested resource, heartbeat and durable evaluation methods run in
an isolated namespace with explicit R99 paths/callbacks. No historical module
globals or sources are modified. All neural/physics code stays native.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from types import FunctionType

from .common import (ROOT,OUT,SOURCE,PARENT,CPU,PY,SIM,PORT,DEV,CONF,FINAL,TASKS,ENV,
    read,utc,verify,bindings,identity,alive,tail,retryable,audit_result,atomic_json)
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import qualified
from VLA.dita_terminal_recovery_r95.identity import await_identity
from VLA.autorefinement_r97.common import score,metric_guard

MODULE='VLA.refinement_r99.owner'
ALLOWED={'VLA.refinement_r99.train','VLA.refinement_r99.evaluate',
    'VLA.continuous_r86.vlanext_server','VLA.recovery_quality_r90.train'}


def module_name(ident):
    cmd=ident.get('command',[]) if ident else []
    return cmd[cmd.index('-m')+1] if '-m' in cmd else None


def checked_signal(ident,sig):
    if not ident or not alive(ident['pid']):return
    if identity(ident['pid'])!=ident:raise ValueError('PID identity changed; do not signal')
    if module_name(ident) not in ALLOWED or sig not in (signal.SIGTERM,signal.SIGKILL):
        raise ValueError('Not an owned R99 worker')
    os.kill(ident['pid'],sig)


def stop_children(children):
    for ident in reversed(children):
        if alive(ident['pid']) and identity(ident['pid'])==ident:checked_signal(ident,signal.SIGTERM)
    deadline=time.monotonic()+20
    while any(alive(i['pid']) and identity(i['pid'])==i for i in children) and time.monotonic()<deadline:
        time.sleep(.2)
    for ident in reversed(children):
        if alive(ident['pid']) and identity(ident['pid'])==ident:checked_signal(ident,signal.SIGKILL)


def cleanup():
    children=read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []
    stop_children(children)
    return {'UTC':utc(),'owned_R99_workers_cleaned':True,'unrelated_processes_signaled':0}


def adapted_method(name):
    from VLA.postfit800_r98 import owner as tested
    original=getattr(tested.Owner,name)
    namespace={**original.__globals__,'OUT':OUT,'PORT':PORT,'ROOT':ROOT,'ENV':ENV,
        'PY':PY,'SIM':SIM,'audit_result':audit_result,'stop_children':stop_children}
    return FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)


class Owner:
    # These bodies are EXACTLY R98's tested implementations, not source-string edits.
    resource=adapted_method('resource')
    wait=adapted_method('wait')
    evaluate=adapted_method('evaluate')

    def __init__(self):
        self.bound=bindings();self.child=self.server=None;self.stage='native_preflight_gate'
        self.children=read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []

    def status(self,state='running',**extra):
        atomic_json(OUT/'status.json',{'UTC':utc(),'pid':os.getpid(),'status':state,'stage':self.stage,
            'family':'vlanext_s','parent_checkpoint':str(PARENT),'planned_final_trials':800,
            'episodes_per_action':200,'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'original_thresholds_preserved':True,**extra})

    def spawn(self,python,module,args,label):
        verify(self.bound)
        if module=='VLA.postfit800_r98.evaluate':module='VLA.refinement_r99.evaluate'
        if module not in ALLOWED:raise ValueError('Not an authorized R99 worker module')
        if module=='VLA.recovery_quality_r90.train' and ('fit' in args or 'reload' not in args):
            raise ValueError('Legacy trainer may only reload, not fit')
        command=list(map(str,[python,'-u','-m',module,*args]))
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:ident,observations=await_identity(child,command,os.getpid(),identity)
        except BaseException:
            if child.poll() is None:child.terminate();child.wait(timeout=20)
            raise
        self.children.append(ident);atomic_json(OUT/'CHILDREN.json',{'children':self.children})
        atomic_json(OUT/(label+'_IDENTITY.json'),{'identity':ident,'startup_observations':observations})
        return child

    def stop_active(self):
        stop_children([next(i for i in self.children if i['pid']==p.pid)
            for p in (self.child,self.server) if p and p.poll() is None])
        self.child=self.server=None

    def native_reload(self,cp,label):
        qualified(cp);receipt=OUT/(label+'_RELOAD_PASSED.json')
        if receipt.exists():
            previous=read(receipt)
            if sha256_file(cp/'weights.safetensors')!=previous['weights_sha256']:
                raise ValueError('Reloaded checkpoint changed')
            return
        for attempt in range(3):
            self.stage=label+'_native_CUDA_reload';self.resource();name=f'{label}_reload{attempt}_{time.time_ns()}'
            self.child=self.spawn(PY,'VLA.recovery_quality_r90.train',
                ['--family','vlanext_s','reload','--checkpoint',cp],name)
            try:self.wait(self.child,name,seconds=3600)
            except (RuntimeError,TimeoutError) as error:
                if not retryable(tail(OUT/(name+'.log'))+'\n'+str(error),abrupt=self.child.poll() in (-9,-15)):raise
                atomic_json(OUT/(name+'_INTERRUPTION.json'),{'UTC':utc(),'error':str(error),'NOT_model_failure':True})
                continue
            finally:self.stop_active()
            value=read(cp/'FRESH_RELOAD.json')
            if not value['passed'] or value['actual_anchors']!=8 or max(value['max_errors'])>1e-6:
                raise ValueError('Eight-anchor exact native CUDA reload mismatch')
            atomic_json(receipt,{'UTC':utc(),'checkpoint':str(cp),'actual_result':value,
                'weights_sha256':sha256_file(cp/'weights.safetensors')});return
        raise RuntimeError('Native reload retries exhausted')

    def fit(self,parent,label,additional=512,batch=32):
        complete=OUT/(label+'_FIT_COMPLETED.json')
        if complete.exists():
            cp=Path(read(complete)['checkpoint']);qualified(cp);return cp
        origin=qualified(parent)['updates'];target=origin+additional;resume=parent
        for attempt in range(3):
            folder=OUT/f'{label}_fit_attempt{attempt:02d}'
            if folder.exists():
                if (folder/'LATEST_COMMITTED.json').exists():
                    candidate=Path(read(folder/'LATEST_COMMITTED.json')['checkpoint']);meta=qualified(candidate)
                    if not origin<meta['updates']<=target:raise ValueError('Wrong training resume boundary')
                    resume=candidate
                if qualified(resume)['updates']==target:
                    atomic_json(complete,{'checkpoint':str(resume),'updates':target});return resume
                continue
            current=qualified(resume)['updates'];self.resource();self.stage=label+'_finetune'
            name=f'{label}_fit_attempt{attempt:02d}'
            self.child=self.spawn(PY,'VLA.refinement_r99.train',[
                '--output',folder,'--resume',resume,'--additional-updates',target-current,
                '--accumulation',batch,'--phase-origin',origin,'--phase-horizon',additional],name)
            try:
                deadline=time.monotonic()+12*3600;changed=time.monotonic();signature=None;low=0
                while self.child.poll() is None:
                    verify(self.bound)
                    path=folder/'status.json';detail=read(path) if path.exists() else {}
                    now=time.monotonic();actual=(detail.get('updates'),detail.get('status'))
                    if actual!=signature:changed=now;signature=actual
                    if now>deadline or now-changed>900:raise TimeoutError('stalled: native fit heartbeat deadline')
                    mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
                    low=low+1 if mem['MemAvailable']<640*1024 else 0
                    if low>=3:raise RuntimeError('Resource interruption: host RAM below640MiB')
                    self.status(actual_training_progress=detail,target_checkpoint_updates=target,
                        log=str(OUT/(name+'.log')))
                    time.sleep(5)
                if self.child.returncode:raise RuntimeError(f'Native fit exit{self.child.returncode}; not a model SR')
                result=read(folder/'COMPLETED.json');cp=Path(result['checkpoint']);meta=qualified(cp)
                if meta['updates']!=target:raise ValueError('Actual completed optimizer step mismatch')
                atomic_json(complete,result);return cp
            except (RuntimeError,TimeoutError) as error:
                text=tail(OUT/(name+'.log'))+'\n'+str(error)
                atomic_json(OUT/(name+'_INTERRUPTION.json'),{'UTC':utc(),'error':str(error),
                    'complete_optimizer_commits_preserved':True,'NOT_model_SR':True})
                transient=retryable(text,abrupt=self.child.poll() in (-9,-15),stalled='stalled:' in str(error))
                if not transient:raise
                if (folder/'LATEST_COMMITTED.json').exists():resume=Path(read(folder/'LATEST_COMMITTED.json')['checkpoint'])
            finally:self.stop_active()
        raise RuntimeError('Three bounded training attempts exhausted')

    def method(self,cp,label,final=False):
        meta=qualified(cp);path=OUT/(label+'_METHOD.json')
        value={'family':'vlanext_s','checkpoint':str(cp),'actual_checkpoint_optimizer_updates':meta['updates'],
            'variant':'native_full_flow_DCT_adam8_RGB3_privileged74_history8_official_gripper',
            'code':{**meta['code'],**self.bound},'weights_sha256':{str(cp/'weights.safetensors'):meta['weights_sha256']},
            'prefixes':{task:1 for task in TASKS},'timing':'10Hz query; history8; horizon8; execute1',
            'no_Oracle_or_geometric_target_action_residual':True,
            'metric':'original R62 semantic first-hit; NOT strict1cm; put release/settle not required',
            'frozen_for_user_requested_benchmark':final,'R99_locked_before_final':final,
            'frozen_after_confirmation':False,
            'benchmark_authorization':{'user_requested_800_after_refinement':final,
                'CONF_disclosed_not_an_invented_high_SR_gate':True},
            'R98_final_reclassified_as_development_for_R99':True,
            'new_final_seeds_never_used_for_training_or_selection':FINAL}
        if path.exists() and read(path)!=value:raise ValueError('Previously bound method changed')
        atomic_json(path,value);return path

    def run(self):
        metric_guard()
        frozen=OUT/'FROZEN_METHOD.json'
        if not frozen.exists():
            # Actual native gradients/weight changes/reload; not an SR estimate.
            gate=self.fit(PARENT,'gradient_gate16',additional=16,batch=4)
            self.native_reload(gate,'gradient_gate16')
            self.native_reload(PARENT,'baseline')
            baseline_method=self.method(PARENT,'baseline')
            baseline=self.evaluate(baseline_method,'baseline_dev40','development_selection',DEV)
            best_cp,best_progress=PARENT,baseline
            first=self.fit(PARENT,'rehearsal512')
            self.native_reload(first,'rehearsal512')
            candidate=self.evaluate(self.method(first,'rehearsal512'),'rehearsal512_dev40','development_selection',DEV)
            if score(read(candidate))>score(read(best_progress)):
                best_cp,best_progress=first,candidate
                second=self.fit(first,'rehearsal1024')
                self.native_reload(second,'rehearsal1024')
                other=self.evaluate(self.method(second,'rehearsal1024'),'rehearsal1024_dev40','development_selection',DEV)
                if score(read(other))>score(read(best_progress)):best_cp,best_progress=second,other
            selection={'UTC':utc(),'checkpoint':str(best_cp),'DEV_source':str(best_progress),
                'score':score(read(best_progress)),'baseline_score':score(read(baseline)),
                'selection_uses_only_shared_DEV40_not_CONF_or_new_final':True,
                'actual_weights_improved_over_baseline_DEV':best_cp!=PARENT}
            atomic_json(OUT/'SELECTED_BEFORE_CONF.json',selection)
            confirmation=self.evaluate(self.method(best_cp,'selected'), 'confirmation40','development_confirmation',CONF)
            atomic_json(OUT/'CONFIRMATION.json',{'progress':str(confirmation),'per_action':read(confirmation)['per_action'],
                'not_used_for_checkpoint_selection':True,'low_CONF_does_not_invent_pass_or_modify_checkpoint':True})
            finalmethod=self.method(best_cp,'FROZEN',final=True)
            # self.method produced FROZEN_METHOD.json atomically.
            if finalmethod!=frozen:raise ValueError('Frozen path mismatch')
            atomic_json(OUT/'FROZEN_BEFORE_FINAL.json',{'UTC':utc(),'method_sha256':sha256_file(frozen),
                'final_seeds':FINAL,'selection':selection,'new_final_outcomes_seen_at_freeze':0})
        else:
            verify(read(frozen)['code']);verify(read(frozen)['weights_sha256'])
        final=self.evaluate(frozen,'final800','fresh_secondary_final800',FINAL)
        result=audit_result(final);atomic_json(OUT/'CERTIFIED_RESULT.json',result)
        text='# R99 VLANeXt-S: actual800\n\nOriginal R62 metric, NOT strict1cm; PUT release not required.\n\n'
        text+='|Action|Successes / n|SR|Safety aborts (failures)|\n|---|---:|---:|---:|\n'
        for task,v in result['per_action'].items():
            text+=f"|{task}|{v['successes']}/{v['n']}|{v['SR_percent']:.2f}%|{v['safety_aborts']}|\n"
        text+='\nOne measured800 cohort, not five seeds. No post-final adaptation in this workflow.\n'
        text+='R98 is prior development evidence. New final instances were locked before training.\n'
        (OUT/'RESULTS.md').write_text(text)
        self.stage='completed_actual800_certified';self.status('completed',per_action=result['per_action'])


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=read(OUT/'QUALIFICATION.json')
        if not proof['passed'] or proof['code']!=bindings():raise ValueError('Current R99 qualification required')
        if (OUT/'CERTIFIED_RESULT.json').exists():return
        owner=Owner();reservation={'UTC':utc(),'owner':identity(os.getpid())}
        atomic_json(OUT/'RESERVATION.json',reservation)
        with (OUT/f'watchdog_{os.getpid()}.log').open('x') as log:
            subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog','--owner-pid',str(os.getpid())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+20
        while not (OUT/f'WATCHDOG_READY_{os.getpid()}.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('Watchdog not ready')
            time.sleep(.1)
        atomic_json(OUT/'STARTED.json',{'UTC':utc(),'pid':os.getpid(),'code':owner.bound})
        try:owner.run()
        except BaseException as error:
            atomic_json(OUT/'ERROR.json',{'UTC':utc(),'error':str(error),'traceback':traceback.format_exc(),
                'NOT_model_SR_or_completed800':True});owner.status('technical_error_retained',error=str(error));raise
        finally:atomic_json(OUT/f'CLEANUP_{os.getpid()}.json',cleanup())


def watchdog(pid):
    expected=read(OUT/'RESERVATION.json')['owner']
    if expected['pid']!=pid:raise ValueError('Watchdog reservation mismatch')
    atomic_json(OUT/f'WATCHDOG_READY_{pid}.json',{'UTC':utc(),'owner':expected})
    while alive(pid) and identity(pid)==expected:
        if (OUT/f'CLEANUP_{pid}.json').exists():return
        time.sleep(5)
    if (OUT/f'CLEANUP_{pid}.json').exists():return
    atomic_json(OUT/f'WATCHDOG_CLEANUP_{pid}.json',cleanup())
    starts=list(OUT.glob('WATCHDOG_RESTART_*.json'))
    if len(starts)>=2 or (OUT/'ERROR.json').exists() or (OUT/'CERTIFIED_RESULT.json').exists():return
    verify(read(OUT/'QUALIFICATION.json')['code'])
    with (OUT/f'restart_{len(starts)}.log').open('x') as log:
        child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(OUT/f'WATCHDOG_RESTART_{len(starts)}.json',{'UTC':utc(),'pid':child.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true')
    p.add_argument('--watchdog',action='store_true');p.add_argument('--owner-pid',type=int);a=p.parse_args()
    if a.watchdog:watchdog(a.owner_pid)
    elif a.launch:
        OUT.mkdir(exist_ok=True)
        if (OUT/'status.json').exists() and alive(read(OUT/'status.json')['pid']):raise ValueError('Duplicate live R99 owner')
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'status.json')},flush=True)
    else:main()
