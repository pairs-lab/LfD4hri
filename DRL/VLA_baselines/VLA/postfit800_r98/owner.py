"""Adopt only after active fit completes; frozen direct800 + durable resume."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import requests

from .common import (ROOT,OUT,SOURCE,CP,CPU,PY,SIM,PORT,SMOKE,FINAL,TASKS,ENV,
    read,utc,verify,bindings,identity,alive,approved,native_s_callbacks,
    tail,retryable,Heartbeat,progress_signature,atomic_json)
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import qualified
from VLA.benchmark_guardian_r89.common import live_children
from VLA.dita_terminal_recovery_r95.identity import await_identity

MODULE='VLA.postfit800_r98.owner'


def module_name(ident):
    words=ident.get('command',[]) if ident else []
    return words[words.index('-m')+1] if '-m' in words else None


def checked_signal(ident,sig,role):
    if not ident or not alive(ident['pid']):return
    if identity(ident['pid'])!=ident:raise ValueError('PID identity changed; no signal sent')
    command=ident['command'];module=module_name(ident)
    if role=='legacy_scheduler':
        valid=module=='VLA.dita_terminal_recovery_r95.others' and sig in (signal.SIGSTOP,signal.SIGCONT,signal.SIGTERM)
    else:
        valid=(module in ('VLA.postfit800_r98.evaluate','VLA.continuous_r86.vlanext_server') or
            (module=='VLA.recovery_quality_r90.train' and 'reload' in command and 'fit' not in command
                and str(CP) in command)) and sig in (signal.SIGTERM,signal.SIGKILL)
    if not valid:raise ValueError('Signal is not an owned evaluation worker or scheduler; never signal an optimizer')
    os.kill(ident['pid'],sig)


def stop_children(children):
    for ident in reversed(children):
        if alive(ident['pid']) and identity(ident['pid'])==ident:checked_signal(ident,signal.SIGTERM,'owned')
    deadline=time.monotonic()+20
    while any(alive(i['pid']) and identity(i['pid'])==i for i in children) and time.monotonic()<deadline:
        time.sleep(.2)
    for ident in reversed(children):
        if alive(ident['pid']) and identity(ident['pid'])==ident:checked_signal(ident,signal.SIGKILL,'owned')


def cleanup(release_parent=True):
    children=read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []
    stop_children(children)
    reservation=read(OUT/'RESERVATION.json') if (OUT/'RESERVATION.json').exists() else {}
    parent=reservation.get('legacy_scheduler');resumed=False
    if release_parent and parent and alive(parent['pid']) and identity(parent['pid'])==parent:
        checked_signal(parent,signal.SIGCONT,'legacy_scheduler');resumed=True
    return {'UTC':utc(),'recorded_inference_workers_cleaned':True,'legacy_parent_resumed':resumed,
        'active_optimizer_signals_sent':0}


def audit_result(path,require_complete=True):
    from VLA.autorefinement_r97.audit import audit
    with native_s_callbacks():return audit(path,require_complete=require_complete)


class Owner:
    def __init__(self):
        self.bound=bindings();self.child=self.server=None;self.stage='reserve_postfit_handoff'
        self.children=read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []

    def status(self,state='running',**extra):
        atomic_json(OUT/'status.json',{'UTC':utc(),'pid':os.getpid(),'status':state,'stage':self.stage,
            'family':'vlanext_s','target_checkpoint':str(CP),'planned_final_trials':800,
            'episodes_per_action':200,'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'original_thresholds_preserved':True,**extra})

    def reserve(self):
        request=read(OUT/'REQUEST.json');parent=request['legacy_scheduler'];optimizer=request['active_optimizer']
        if request['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            raise ValueError('Boot changed; never signal stale process identities')
        atomic_json(OUT/'RESERVATION.json',{'UTC':utc(),'owner':identity(os.getpid()),
            'legacy_scheduler':parent,'adopted_optimizer_NEVER_signal':optimizer})
        with (OUT/f'watchdog_{os.getpid()}.log').open('x') as log:
            subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog','--owner-pid',str(os.getpid())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ready=OUT/f'WATCHDOG_READY_{os.getpid()}.json';deadline=time.monotonic()+20
        while not ready.exists():
            if time.monotonic()>deadline:raise RuntimeError('Postfit cleanup watchdog not ready')
            time.sleep(.1)
        if not alive(parent['pid']):
            qualified(CP)
            return
        if identity(parent['pid'])!=parent:raise ValueError('Legacy scheduler identity changed')
        children=live_children(parent['pid'])
        if children-set([optimizer['pid']]):raise ValueError('Legacy scheduler acquired an unexpected child')
        if alive(optimizer['pid']) and identity(optimizer['pid'])!=optimizer:
            raise ValueError('Original optimizer identity changed; no scheduling signal sent')
        checked_signal(parent,signal.SIGSTOP,'legacy_scheduler')
        deadline=time.monotonic()+3
        while (Path('/proc')/str(parent['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('T','t'):
            if time.monotonic()>deadline:raise RuntimeError('Scheduler pause not acknowledged')
            time.sleep(.02)
        if live_children(parent['pid'])-set([optimizer['pid']]):
            checked_signal(parent,signal.SIGCONT,'legacy_scheduler')
            raise ValueError('Scheduler launch raced; existing child not interrupted')
        self.stage='waiting_existing_native_fit9216_OPTIMIZER_UNTOUCHED'
        self.status(adopted_optimizer=optimizer)

    def await_fit(self):
        request=read(OUT/'REQUEST.json');optimizer=request['active_optimizer'];parent=request['legacy_scheduler']
        deadline=time.monotonic()+24*3600
        while alive(optimizer['pid']):
            if identity(optimizer['pid'])!=optimizer:raise ValueError('Adopted optimizer PID reused')
            self.status(actual_training_progress=read(CP/'status.json') if (CP/'status.json').exists() else None,
                adopted_optimizer_pid=optimizer['pid'],optimizer_signals_sent=0)
            if time.monotonic()>deadline:raise TimeoutError('Existing fit did not finish; optimizer not terminated')
            time.sleep(10)
        meta=qualified(CP)
        if meta['updates']!=9216 or meta['family']!='vlanext_s':raise ValueError('Wrong actual completed checkpoint')
        if alive(parent['pid']):
            if identity(parent['pid'])!=parent or live_children(parent['pid']):
                raise ValueError('Cannot retire scheduler with active children')
            checked_signal(parent,signal.SIGTERM,'legacy_scheduler')
            checked_signal(parent,signal.SIGCONT,'legacy_scheduler')
            deadline=time.monotonic()+20
            while alive(parent['pid']) and time.monotonic()<deadline:time.sleep(.1)
            if alive(parent['pid']):raise RuntimeError('Legacy scheduler did not exit; no second GPU owner')
        atomic_json(OUT/'HANDOFF_COMPLETED.json',{'UTC':utc(),'checkpoint':str(CP),
            'actual_updates':9216,'native_weight_and_optimizer_hashes_verified':True,
            'active_optimizer_signals_sent':0,'old_scheduler':parent,'new_owner':identity(os.getpid())})
        old=read(SOURCE/'LIVE.json')
        atomic_json(OUT/'PRIOR_SCHEDULER_LIVE.json',old)
        atomic_json(SOURCE/'R98_SUCCESSOR.json',{'UTC':utc(),'status':str(OUT/'status.json'),
            'existing_fit_finished_without_interruption':True})
        self.stage='completed_fit_native_reload_pending';self.status()

    def spawn(self,python,module,args,label):
        verify(self.bound)
        if module not in ('VLA.postfit800_r98.evaluate','VLA.continuous_r86.vlanext_server',
            'VLA.recovery_quality_r90.train'):raise ValueError('Unexpected native worker module')
        if module=='VLA.recovery_quality_r90.train' and ('fit' in args or 'reload' not in args
            or 'vlanext_s' not in args or str(CP) not in list(map(str,args))):
            raise ValueError('This postfit workflow cannot launch or cancel training')
        command=list(map(str,[python,'-u','-m',module,*args]))
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:ident,observations=await_identity(child,command,os.getpid(),identity)
        except BaseException:
            if child.poll() is None:child.terminate();child.wait(timeout=20)
            raise
        self.children.append(ident)
        atomic_json(OUT/'CHILDREN.json',{'children':self.children})
        atomic_json(OUT/(label+'_IDENTITY.json'),{'identity':ident,'startup_observations':observations})
        return child

    def resource(self):
        from VLA.cross_vla800_r62.common import resources
        deadline=time.monotonic()+12*3600
        while True:
            value=resources()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
            if not apps and value['GPU_free_MiB']>=13000 and value['RAM_available_MiB']>=5500 and value['disk_free_GiB']>=25:
                return
            self.status('waiting_safe_inference_resources',resources=value)
            if time.monotonic()>deadline:raise TimeoutError('No safe postfit inference resource window')
            time.sleep(10)

    def wait(self,child,label,folder=None,seconds=3600):
        deadline=time.monotonic()+seconds;heartbeat=Heartbeat(time.monotonic());low=0
        while child.poll() is None:
            verify(self.bound)
            if self.server and self.server.poll() is not None:raise RuntimeError('ConnectionError: native server exited')
            if time.monotonic()>deadline:raise TimeoutError('Finite native worker deadline')
            if folder:
                age=heartbeat.observe(progress_signature(folder),time.monotonic())
                if age>900:raise RuntimeError('stalled: no progress OR native query heartbeat for15min')
            mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
            low=low+1 if mem['MemAvailable']<640*1024 else 0
            if low>=3:raise RuntimeError('Resource interruption: host RAM below640MiB')
            detail=None
            if folder and (folder/'progress.json').exists():
                detail=read(folder/'progress.json');detail={k:v for k,v in detail.items() if k!='rows'}
            self.status(log=str(OUT/(label+'.log')),actual_test_progress=detail,
                neural_query_heartbeat_age_s=age if folder else None)
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{label} exit{child.returncode}; retained evidence, NOT model SR0')

    def reload(self):
        self.stage='fresh_native_S_CUDA_reload'
        receipt=CP/'FRESH_RELOAD.json'
        if (OUT/'NATIVE_RELOAD_PASSED.json').exists():
            record=read(OUT/'NATIVE_RELOAD_PASSED.json')
            if sha256_file(receipt)!=record['receipt_sha256']:raise ValueError('Reload receipt changed')
            return
        for attempt in range(3):
            self.resource();label=f'native_reload_{time.time_ns()}'
            self.child=self.spawn(PY,'VLA.recovery_quality_r90.train',
                ['--family','vlanext_s','reload','--checkpoint',CP],label)
            try:
                self.wait(self.child,label,seconds=3600)
            except (RuntimeError,TimeoutError) as error:
                abrupt=self.child.poll() in (-9,-15)
                if not retryable(tail(OUT/(label+'.log'))+'\n'+str(error),abrupt=abrupt):raise
                atomic_json(OUT/(label+'_INTERRUPTION.json'),{'UTC':utc(),'error':str(error),
                    'technical_retry':attempt+1,'checkpoint_not_changed':True})
                continue
            finally:
                if self.child and self.child.poll() is None:
                    stop_children([next(i for i in self.children if i['pid']==self.child.pid)])
                self.child=None
            result=read(receipt)
            if not result['passed'] or result['actual_anchors']!=8 or max(result['max_errors'])>1e-6:
                raise ValueError('Fresh exact native CUDA reload mismatch')
            atomic_json(OUT/'NATIVE_RELOAD_PASSED.json',{'UTC':utc(),'receipt':str(receipt),
                'receipt_sha256':sha256_file(receipt),'actual_result':result})
            return
        raise RuntimeError('Three bounded reload retries exhausted; no false readiness pass')

    def freeze(self):
        meta=qualified(CP)
        method={'family':'vlanext_s','checkpoint':str(CP),'actual_checkpoint_optimizer_updates':9216,
            'variant':'native_full_adam8_privileged74_map3_OFFICIAL_GRIPPER_SIGN',
            'decoder':'official sign: native gripper >0 -> +1, otherwise -1; pose untouched',
            'code':{**meta['code'],**self.bound},'weights_sha256':{str(CP/'weights.safetensors'):meta['weights_sha256']},
            'prefixes':{t:1 for t in TASKS},'frozen_after_confirmation':False,
            'frozen_for_user_requested_benchmark':True,'R98_frozen_before_any_evaluation':True,
            'benchmark_authorization':{'user_requested_direct_postfit800':True,'no_CONF_pass_claimed':True},
            'timing':'10Hz observation/query; native horizon8; actual causal history8; execute1',
            'no_Oracle_or_geometric_target_action_residual':True,
            'metric':'original R62 semantic first-hit; NOT strict1cm; put release/settle not required'}
        path=OUT/'FROZEN_METHOD.json'
        if path.exists() and read(path)!=method:raise ValueError('Cannot change a previously frozen method')
        atomic_json(path,method)
        atomic_json(OUT/'FROZEN_BEFORE_EVALUATION.json',{'UTC':utc(),'method_sha256':sha256_file(path),
            'actual_checkpoint_updates':9216,'smoke_seeds':SMOKE,'final_seeds':FINAL,
            'no_outcome_based_checkpoint_selection_or_threshold_changes':True})
        return path

    def evaluate(self,method,label,role,seeds):
        from VLA.autorefinement_r97.evaluate import ledger
        finish=OUT/(label+'_COMPLETE.json')
        if finish.exists():
            progress=Path(read(finish)['progress']);audit_result(progress);return progress
        cohort=OUT/(label+'_COHORT.json');value={'role':role,'seeds':seeds}
        if cohort.exists() and read(cohort)!=value:raise ValueError('Locked cohort changed')
        atomic_json(cohort,value)
        previous=None
        for attempt in range(4):
            name=f'{label}_attempt{attempt:02d}';folder=OUT/name;progress=folder/'progress.json'
            if folder.exists():
                if progress.exists():
                    ledger(progress,value)
                    if read(progress)['status'].startswith('completed_actual_'):
                        result=audit_result(progress);atomic_json(folder/'AUDIT.json',result)
                        atomic_json(finish,{'progress':str(progress)});return progress
                    previous=progress
                continue
            self.stage=label;self.resource();self.server=self.spawn(PY,'VLA.continuous_r86.vlanext_server',
                ['--method',method,'--port',PORT],name+'_server')
            session=requests.Session();session.trust_env=False
            try:
                deadline=time.monotonic()+600
                while True:
                    if self.server.poll() is not None:raise RuntimeError('Native S server failed during loading')
                    try:
                        response=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2);response.raise_for_status()
                        meta=response.json()
                        if (meta['method_sha256']!=sha256_file(method) or meta['family']!='vlanext_s'
                            or meta['native_horizon']!=8 or not meta['actual_neural_model']):
                            raise ValueError('Incorrect native server/checkpoint/horizon identity')
                        break
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Native readiness timeout')
                        self.status();time.sleep(2)
                if previous:
                    snapshot={**read(previous),'source_progress':str(previous),'source_sha256':sha256_file(previous)}
                    atomic_json(OUT/(name+'_RESUME_SNAPSHOT.json'),snapshot)
                    if snapshot['completed_trials']:audit_result(previous,require_complete=False)
                args=['--method',method,'--cohort',cohort,'--output',folder,'--port',PORT]
                if previous:args.extend(['--resume-progress',previous])
                self.child=self.spawn(SIM,'VLA.postfit800_r98.evaluate',args,name)
                self.wait(self.child,name,folder,seconds=36*3600 if len(seeds)==200 else 2*3600)
                self.child=None
                result=audit_result(progress);atomic_json(folder/'AUDIT.json',result)
                atomic_json(finish,{'progress':str(progress)});return progress
            except (RuntimeError,TimeoutError) as error:
                text=tail(OUT/(name+'.log'))+'\n'+tail(OUT/(name+'_server.log'))+'\n'+str(error)
                atomic_json(OUT/(name+'_INTERRUPTION.json'),{'UTC':utc(),'error':str(error),
                    'retained_progress':str(progress) if progress.exists() else None,'NOT_model_SR':True})
                transient=retryable(text,abrupt=bool(self.child and self.child.poll() in (-9,-15)),
                    stalled='stalled:' in str(error))
                if not transient:raise
                if progress.exists():previous=progress
            finally:
                session.close()
                owned=[next(i for i in self.children if i['pid']==p.pid) for p in (self.child,self.server)
                    if p and p.poll() is None]
                stop_children(owned);self.child=self.server=None
        raise RuntimeError('Four bounded technical attempts exhausted; no invented actual800 completion')

    def run(self):
        self.reserve();self.await_fit();self.reload();method=self.freeze()
        smoke=self.evaluate(method,'integration4','development_selection',SMOKE)
        atomic_json(OUT/'INTEGRATION4.json',{'progress':str(smoke),'completed_actual_trials':4,
            'purpose':'input/action/plant/receipt/termination lifecycle only; NOT checkpoint selection',
            'policy_failures_do_not_change_checkpoint_or_prevent_requested800':True})
        final=self.evaluate(method,'final800','fresh_secondary_final800',FINAL)
        result=audit_result(final)
        atomic_json(OUT/'CERTIFIED_RESULT.json',result)
        text='# VLANeXt-S — actual800\n\nOriginal R62 semantic metric; NOT strict1cm; PUT release not required.\n\n'
        text+='|Action|Successes / n|SR|\n|---|---:|---:|\n'
        for task,v in result['per_action'].items():text+=f"|{task}|{v['successes']}/{v['n']}|{v['SR_percent']:.2f}%|\n"
        text+='\nOne measured cohort, not five seeds; all task failures and safety aborts retained.\n'
        (OUT/'RESULTS.md').write_text(text)
        self.stage='completed_actual800_certified';self.status('completed',per_action=result['per_action'])


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=read(OUT/'QUALIFICATION.json')
        if not proof['passed'] or proof['code']!=bindings():raise ValueError('Current CPU preflight required')
        if (OUT/'CERTIFIED_RESULT.json').exists():return
        owner=Owner();atomic_json(OUT/'STARTED.json',{'UTC':utc(),'pid':os.getpid(),'code':owner.bound})
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
    starts=list(OUT.glob('WATCHDOG_RESTART_*.json'))
    restart=len(starts)<2 and not (OUT/'ERROR.json').exists() and not (OUT/'CERTIFIED_RESULT.json').exists()
    atomic_json(OUT/f'WATCHDOG_CLEANUP_{pid}.json',cleanup(release_parent=not restart))
    if restart:
        verify(read(OUT/'QUALIFICATION.json')['code'])
        with (OUT/f'restart_{len(starts)}.log').open('x') as log:
            process=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/f'WATCHDOG_RESTART_{len(starts)}.json',{'UTC':utc(),'pid':process.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true')
    p.add_argument('--watchdog',action='store_true');p.add_argument('--owner-pid',type=int);a=p.parse_args()
    if a.watchdog:watchdog(a.owner_pid)
    elif a.launch:
        OUT.mkdir(exist_ok=True)
        if (OUT/'status.json').exists() and alive(read(OUT/'status.json')['pid']):raise ValueError('Duplicate live owner')
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            process=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':process.pid,'status':str(OUT/'status.json')})
    else:main()
