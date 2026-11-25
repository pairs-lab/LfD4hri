"""Durable native admission -> finite fit -> DEV/CONF -> frozen actual800.

R99 is never signalled. Only explicitly owned R100 worker identities may be
stopped/restarted. This is a finite tested scheduler, not an unattended code
generator that changes a model/criterion in the middle of its benchmark.
"""
import argparse,fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
import requests
from .common import ROOT,OUT,CPU,SIM,ENV,R99,read,verify,bindings,atomic_json,utc
from .recipe import FAMILIES,PYTHONS,GATES,HORIZONS,PORTS,RECIPES
from .guard import resources,r99_live,run as guarded,checked_stop,admitted_fraction,same_owned_process
from .native_gate import source_bindings
from .evaluate import FINAL,DEV,CONF,SMOKE,audit_result
from .train import qualified
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import identity,alive
from VLA.postfit800_r98.common import tail,retryable,Heartbeat,progress_signature
from VLA.dita_terminal_recovery_r95.identity import await_identity

MODULE='VLA.admission_r100.workflow'


class WorkerInterrupted(RuntimeError):
    """Restart from a new attempt/committed state, never reuse a partial folder."""


def concurrent_fit_admitted(family):
    """Only the measured low-memory Seer trainer may share the primary GPU."""
    if family!='seer':return False
    root=OUT/family
    if (root/'CONCURRENT_TRAINING_DEFERRED.json').exists():return False
    paths=[root/'NATIVE_ADMISSION_READY.json',root/'TRAINER_CODE_ROUNDTRIP.json']
    if not all(p.exists() for p in paths):return False
    native,trainer=map(read,paths)
    if not native.get('passed') or not trainer.get('passed') or trainer.get('effective_batch')!=16:return False
    meta=read(Path(trainer['checkpoint'])/'TRAINING.json')
    return bool(meta.get('actual_full_native_model') and meta['peak_GPU_MiB']<=3000
        and trainer['fresh_native_reload']['passed'] and native['actual_integration_trials']==4)


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Immutable admission/cohort/method changed: '+str(path))
    else:atomic_json(path,value)
    return path


def foundations(family):
    names={'seer':['seer_DROID','seer_CLIP','seer_MAE'],
        'gr00t_n15':['gr00t_'+n for n in ('config.json','model.safetensors.index.json',
            'model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors')],
        'falcon_esm':[]}[family]
    result={}
    for name in names:
        r=read(OUT/'assets'/(name+'.json'))
        if not r['completed']:raise ValueError('Unfinished authentic asset')
        result[r['path']]=r['sha256']
    if family=='falcon_esm':
        from VLA.falcon_esm_r91.common import WEIGHT,WEIGHT_SHA
        result[str(WEIGHT)]=WEIGHT_SHA
    return result


def method_code(family):
    from VLA.postfit800_r98.common import bindings as inherited
    result={**inherited(),**source_bindings(family)}
    for name in ('server.py','evaluate.py','contracts.py','checkpoint_io.py'):
        p=Path(__file__).parent/name;result[str(p)]=sha256_file(p)
    # Protect actual native implementations and all directly reused loader
    # helpers, not just a small entrypoint that might hide a changed backbone.
    if family=='falcon_esm':
        from VLA.falcon_esm_r91.common import bindings as native_bindings
        result.update(native_bindings())
    elif family=='gr00t_n15':
        folder=ROOT/'VLA/third_party/gr00t_n15/gr00t/model'
        result.update({str(p):sha256_file(p) for p in folder.rglob('*.py')})
    return result


def make_method(family,checkpoint,destination,*,final=False):
    checkpoint=Path(checkpoint);meta=qualified(checkpoint,family)
    method={'family':family,'checkpoint':str(checkpoint.resolve()),'actual_checkpoint_optimizer_updates':meta['updates'],
        'variant':'authentic native foundation +explicit learned causal privileged adaptation',
        'code':{**method_code(family),**meta['code']},
        'weights_sha256':{str(checkpoint.resolve()/'weights.safetensors'):meta['weights_sha256'],**foundations(family)},
        'prefixes':{t:1 for t in ('touch','pick','move','pick_place')},
        'native_horizon':HORIZONS[family],'timing':'10Hz observe/query, execute1; video disabled',
        'no_Oracle_or_geometric_target_action_residual':True,
        'R100_frozen_before_final':final,'frozen_for_user_requested_benchmark':final,
        'privileged_inputs_are_SIMULATION_state_NOT_pure_image_text_baseline':True,
        'metric':'original R62 semantic first-hit, NOT strict1cm, PUT release not required'}
    return immutable(destination,method)


def stop_all(children):
    owned=[r for r in children if same_owned_process(r)]
    for r in reversed(owned):checked_stop(r)
    deadline=time.monotonic()+20
    while any(same_owned_process(r) for r in owned) and time.monotonic()<deadline:time.sleep(.2)
    for r in owned:
        if same_owned_process(r):os.kill(r['pid'],signal.SIGKILL)


class Owner:
    def __init__(self):
        self.bound=read(OUT/'AUTOMATION_QUALIFICATION.json')['code'];verify(self.bound)
        self.family=None;self.stage='native_admission';self.server=self.child=None
        self.children=read(OUT/'WORKFLOW_CHILDREN.json')['children'] if (OUT/'WORKFLOW_CHILDREN.json').exists() else []

    def status(self,status='running',**extra):
        state={'UTC':utc(),'pid':os.getpid(),'status':status,'stage':self.stage,'family':self.family,
            'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'R99_preserved_running':r99_live(),'original_thresholds_unchanged':True,**extra}
        atomic_json(OUT/'WORKFLOW_STATUS.json',state)
        if self.family:atomic_json(OUT/self.family/'WORKFLOW_STATUS.json',state)

    def resource(self,*,after_primary=False):
        deadline=time.monotonic()+48*3600
        while True:
            verify(self.bound);r=resources();active=r99_live()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                text=True,timeout=15).strip()
            if (not after_primary or not active) and (active or not apps) and r['RAM_available_MiB']>=3500 and r['GPU_free_MiB']>=(8000 if active else 13500):
                return admitted_fraction(active,r,['--family',self.family])
            self.status('waiting_safe_resources',resources=r,training_waits_for_primary=after_primary)
            if time.monotonic()>deadline:raise TimeoutError('Finite48h resource admission window exhausted')
            time.sleep(10)

    def spawn(self,python,module,args,label):
        verify(self.bound)
        if module not in ('VLA.admission_r100.server','VLA.admission_r100.evaluate'):raise ValueError('Unexpected live worker')
        command=list(map(str,[python,'-u','-m',module,*args]));root=OUT/self.family
        with (root/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:r,observations=await_identity(child,command,os.getpid(),identity)
        except BaseException:
            if child.poll() is None:child.terminate();child.wait(timeout=20)
            raise
        self.children.append(r);atomic_json(OUT/'WORKFLOW_CHILDREN.json',{'children':self.children})
        atomic_json(root/(label+'_IDENTITY.json'),{'identity':r,'startup_observations':observations});return child

    def wait(self,child,label,folder=None,seconds=3600):
        deadline=time.monotonic()+seconds;heartbeat=Heartbeat(time.monotonic())
        while child.poll() is None:
            verify(self.bound)
            if self.server and self.server.poll() is not None:raise RuntimeError('ConnectionError: owned native server exited')
            if time.monotonic()>deadline:raise TimeoutError('Finite worker deadline')
            age=heartbeat.observe(progress_signature(folder),time.monotonic()) if folder else None
            if folder and age>900:raise RuntimeError('stalled: no trace/query/restore heartbeat15min')
            r=resources()
            if r['RAM_available_MiB']<1400 or r['GPU_free_MiB']<2200:
                raise RuntimeError('Resource interruption: preserve headroom and committed evidence')
            if r99_live():
                q=R99/'final800_attempt00/NEURAL_QUERIES.jsonl'
                if q.exists() and time.time()-q.stat().st_mtime>60:
                    raise RuntimeError('Resource interruption: preserve primary neural query latency')
            progress=read(folder/'progress.json') if folder and (folder/'progress.json').exists() else None
            self.status(log=label,actual_progress={k:v for k,v in progress.items() if k!='rows'} if progress else None,
                query_heartbeat_age_s=age,resources=r);time.sleep(5)
        if child.returncode:raise RuntimeError(f'{label} exit{child.returncode}; NOT model SR0')

    def gpu(self,family,args,label,*,module='VLA.admission_r100.native_gate',after_primary=False,seconds=3600):
        # guard.run itself has a process-identity record. A killed owner cannot
        # leave its worker eligible to run concurrently with a successor.
        self.family=family;self.stage=label;self.resource(after_primary=after_primary);self.status()
        for attempt in range(4):
            tag=f'{family}_{label}_guard{attempt:02d}'
            record=OUT/'guard'/(tag+'.json')
            if record.exists():
                value=read(record)
                if value['status']=='completed_native_worker':return value
                if value['status']=='native_worker_error_requires_diagnosis':
                    if not retryable(tail(value.get('log',''))):raise RuntimeError('Native worker requires code diagnosis: '+str(record))
                if module=='VLA.admission_r100.train' or '--output' in args:
                    raise WorkerInterrupted('Retained interrupted output; require new attempt or committed resume')
                continue
            value=guarded(ROOT/PYTHONS[family],module,args,tag,during_R99=not after_primary,seconds=seconds)
            if value['status']=='completed_native_worker':return value
            if value['status']=='deferred_until_safe_resources':self.resource(after_primary=after_primary);continue
            if value['status']=='stopped_owned_worker_to_preserve_resources' and r99_live():
                # Do not try to fit a lower-fidelity proxy to a tight window.
                if module=='VLA.admission_r100.train' or '--output' in args:
                    raise WorkerInterrupted('Resource-deferred real native job; retain partial folder')
                self.resource(after_primary=True);continue
            if not retryable(tail(value.get('log','')),abrupt=value.get('returncode') in (-9,-15)):
                raise RuntimeError('Native admission/training requires diagnosis: '+str(value))
            if module=='VLA.admission_r100.train' or '--output' in args:
                raise WorkerInterrupted('Technical interruption; new attempt/optimizer checkpoint required')
        raise RuntimeError('Four bounded native worker attempts exhausted')

    def evaluate(self,method,label,role,seeds):
        from VLA.autorefinement_r97.evaluate import ledger
        root=OUT/self.family;finish=root/(label+'_COMPLETE.json')
        if finish.exists():path=Path(read(finish)['progress']);audit_result(path);return path
        cohort=immutable(root/(label+'_COHORT.json'),{'role':role,'seeds':seeds});previous=None
        for attempt in range(4):
            name=f'{label}_attempt{attempt:02d}';folder=root/name;progress=folder/'progress.json'
            if folder.exists():
                if progress.exists():
                    data,_,_=ledger(progress,read(cohort))
                    if data['status'].startswith('completed_actual_'):
                        result=audit_result(progress);atomic_json(folder/'AUDIT.json',result)
                        atomic_json(finish,{'progress':str(progress)});return progress
                    previous=progress
                continue
            self.stage=label;fraction=self.resource();self.status()
            self.server=self.spawn(ROOT/PYTHONS[self.family],'VLA.admission_r100.server',
                ['--method',method,'--port',PORTS[self.family],'--memory-fraction',fraction],name+'_server')
            session=requests.Session();session.trust_env=False
            try:
                deadline=time.monotonic()+600
                while True:
                    if self.server.poll() is not None:raise RuntimeError('Native server failed loading')
                    rsrc=resources()
                    if rsrc['RAM_available_MiB']<1400 or rsrc['GPU_free_MiB']<2200:
                        raise RuntimeError('Resource interruption: native loading headroom guard')
                    try:
                        r=session.get(f'http://127.0.0.1:{PORTS[self.family]}/health',timeout=2);r.raise_for_status();value=r.json()
                        if value['family']!=self.family or value['native_horizon']!=HORIZONS[self.family] or value['method_sha256']!=sha256_file(method):
                            raise ValueError('Wrong native checkpoint/family/horizon identity')
                        break
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Native server readiness timeout')
                        time.sleep(2)
                args=['--method',method,'--cohort',cohort,'--output',folder,'--port',PORTS[self.family]]
                if previous:
                    if read(previous)['completed_trials']:audit_result(previous,require_complete=False)
                    args.extend(['--resume-progress',previous])
                self.child=self.spawn(SIM,'VLA.admission_r100.evaluate',args,name)
                self.wait(self.child,name,folder,seconds=48*3600 if len(seeds)==200 else 4*3600);self.child=None
                result=audit_result(progress);atomic_json(folder/'AUDIT.json',result)
                atomic_json(finish,{'progress':str(progress)});return progress
            except (RuntimeError,TimeoutError) as error:
                text=tail(root/(name+'.log'))+'\n'+tail(root/(name+'_server.log'))+'\n'+str(error)
                atomic_json(root/(name+'_INTERRUPTION.json'),{'UTC':utc(),'error':str(error),
                    'retained_progress':str(progress) if progress.exists() else None,'NOT_model_SR':True})
                if not retryable(text,abrupt=bool(self.child and self.child.poll() in (-9,-15)),stalled='stalled:' in str(error)):raise
                if progress.exists():previous=progress
            finally:
                session.close();stop_all([i for i in self.children if i['pid'] in
                    [p.pid for p in (self.child,self.server) if p and p.poll() is None]])
                self.child=self.server=None
        raise RuntimeError('Four bounded physical-evaluation attempts exhausted; no fabricated completion')

    def admission(self,family):
        self.family=family;root=OUT/family;self.stage='verifying_native_admission';self.status()
        for file in ('DATA_STATS.json','DATA_AUDIT.json'):
            if not (root/file).exists():raise ValueError('Native data audit not yet complete: '+str(root/file))
        chosen=root/'SELECTED_NATIVE_GATE.json'
        gate=Path(read(chosen)['checkpoint']) if chosen.exists() else root/GATES[family]
        if not (gate/'TRAINING.json').exists():
            for attempt in range(4):
                candidate=root/GATES[family] if attempt==0 else root/f'gate2_resource_retry{attempt:02d}'
                if (candidate/'TRAINING.json').exists():
                    try:qualified(candidate,family)
                    except ValueError as error:
                        atomic_json(root/f'GATE_SUPERSEDED_{attempt:02d}.json',{'UTC':utc(),'checkpoint':str(candidate),
                            'reason':str(error),'earlier_evidence_preserved':True,'NOT_model_SR':True})
                        continue
                    gate=candidate;break
                if candidate.exists():continue
                try:self.gpu(family,['--family',family,'--op','gate','--output',candidate],f'native_gate2_{attempt:02d}',
                    after_primary=attempt>0)
                except WorkerInterrupted:continue
                gate=candidate;break
            else:raise RuntimeError('Native full-model two-update admission budget exhausted')
        qualified(gate,family)
        immutable(chosen,{'checkpoint':str(gate.resolve())})
        if not (gate/'FRESH_RELOAD.json').exists():
            self.gpu(family,['--family',family,'--op','reload','--checkpoint',gate],'native_fresh_reload2')
        reload=read(gate/'FRESH_RELOAD.json')
        if not reload['passed'] or len(reload['max_errors'])!=8 or max(reload['max_errors'])>1e-6:raise ValueError('Native fresh reload failed')
        # Also execute the exact LONG-TRAIN code for two real accumulated
        # updates, save its optimizer, and reload independently. Its weights
        # are an integration fixture, NOT selected using test outcomes and
        # NOT silently substituted for the admitted initialization.
        roundtrip=root/'TRAINER_CODE_ROUNDTRIP.json'
        if not roundtrip.exists():
            cp=None
            for attempt in range(4):
                output=root/f'trainer_roundtrip_attempt{attempt:02d}'
                if (output/'COMPLETED.json').exists():cp=Path(read(output/'COMPLETED.json')['checkpoint']);break
                if output.exists():continue
                try:self.gpu(family,['--family',family,'--resume',gate,'--output',output,'--additional-updates',2],
                    f'trainer_code_roundtrip{attempt:02d}',module='VLA.admission_r100.train',after_primary=attempt>0)
                except WorkerInterrupted:continue
                cp=Path(read(output/'COMPLETED.json')['checkpoint']);break
            if cp is None:raise RuntimeError('Actual accumulated trainer roundtrip attempts exhausted')
            qualified(cp,family)
            self.gpu(family,['--family',family,'--op','reload','--checkpoint',cp],'trainer_code_fresh_reload')
            if not read(cp/'FRESH_RELOAD.json')['passed']:raise ValueError('Actual trainer saved adaptation cannot reload')
            atomic_json(roundtrip,{'passed':True,'UTC':utc(),'checkpoint':str(cp),'real_additional_updates':2,
                'effective_batch':RECIPES[family]['effective_batch'],'optimizer_RNG_checkpoint_restored':True,
                'fresh_native_reload':read(cp/'FRESH_RELOAD.json'),'fixture_weights_not_used_for_long_fit_or_SR':True})
        method=make_method(family,gate,root/'INTEGRATION_METHOD.json')
        path=self.evaluate(method,'integration4','development_selection',SMOKE)
        atomic_json(root/'NATIVE_ADMISSION_READY.json',{'passed':True,'UTC':utc(),'gate_checkpoint':str(gate),
            'actual_optimizer_updates':2,'fresh_reload_anchors':8,'actual_integration_trials':4,
            'integration_audit':audit_result(path),'trained_success_rate_or_high_performance_NOT_implied':True,
            'next':'finite target-domain adaptation then real DEV/CONF and actual800'})
        return gate

    def fit(self,family,parent,phase):
        root=OUT/family;completion=root/(phase+'_FINISHED.json')
        if completion.exists():cp=Path(read(completion)['checkpoint']);qualified(cp,family);return cp
        target=qualified(parent,family)['updates']+RECIPES[family]['initial_updates']
        cp=Path(parent);concurrent=concurrent_fit_admitted(family)
        # Retry from the newest FULLY published optimizer/RNG checkpoint only.
        for attempt in range(4):
            output=root/f'{phase}_attempt{attempt:02d}'
            if output.exists():
                latest=output/'LATEST_COMMITTED.json'
                if latest.exists():cp=Path(read(latest)['checkpoint']);qualified(cp,family)
                if (output/'COMPLETED.json').exists():
                    cp=Path(read(output/'COMPLETED.json')['checkpoint']);break
                continue
            remaining=target-qualified(cp,family)['updates']
            if remaining<=0:break
            # Only Seer's measured2.5GiB native batch16 trainer may overlap
            # R99. All others wait; a resource interruption revokes overlap
            # persistently and resumes the exact commit after R99 completes.
            try:self.gpu(family,['--family',family,'--resume',cp,'--output',output,'--additional-updates',remaining],
                phase+f'_attempt{attempt:02d}',module='VLA.admission_r100.train',after_primary=not concurrent,seconds=36*3600)
            except WorkerInterrupted:
                if (output/'LATEST_COMMITTED.json').exists():cp=Path(read(output/'LATEST_COMMITTED.json')['checkpoint'])
                if concurrent and r99_live():
                    atomic_json(root/'CONCURRENT_TRAINING_DEFERRED.json',{'UTC':utc(),
                        'reason':'owned concurrent trainer interrupted; preserve primary and resume after R99',
                        'last_committed_checkpoint':str(cp),'R99_signals_sent':0})
                    concurrent=False
                continue
            cp=Path(read(output/'COMPLETED.json')['checkpoint']);break
        if qualified(cp,family)['updates']!=target:raise RuntimeError('Finite fit did not reach its locked target')
        self.gpu(family,['--family',family,'--op','reload','--checkpoint',cp],phase+'_fresh_reload',after_primary=not concurrent)
        atomic_json(completion,{'UTC':utc(),'checkpoint':str(cp),'actual_updates':target});return cp

    def finish_family(self,family,parent):
        from VLA.autonomy_r83.common import score,ready
        self.family=family;root=OUT/family
        if (root/'CERTIFIED_RESULT.json').exists():audit_result(Path(read(root/'CERTIFIED_RESULT.json')['source']));return
        a=self.fit(family,parent,'adapt2048');method=make_method(family,a,root/'ADAPT2048_METHOD.json')
        first=self.evaluate(method,'dev40_adapt2048','development_selection',DEV)
        selected=a;candidates=[{'checkpoint':str(a),'progress':str(first),'score':list(score(read(first)))}]
        if not ready(read(first)):
            b=self.fit(family,a,'adapt4096');m=make_method(family,b,root/'ADAPT4096_METHOD.json')
            second=self.evaluate(m,'dev40_adapt4096','development_selection',DEV)
            candidates.append({'checkpoint':str(b),'progress':str(second),'score':list(score(read(second)))})
            if score(read(second))>score(read(first)):selected=b
        selection=immutable(root/'DEV_SELECTION.json',{'checkpoint':str(selected),'candidates':candidates,
            'criterion':'completed DEV40: total successes, minimum action successes, fewer safety aborts; ties earlier',
            'CONF_and_final_NOT_used_for_selection':True})
        m=make_method(family,selected,root/'CONFIRMATION_METHOD.json')
        confirm=self.evaluate(m,'confirmation40','development_confirmation',CONF)
        if not ready(read(confirm)):
            atomic_json(root/'PERFORMANCE_NOT_READY.json',{'UTC':utc(),'actual_confirmation':audit_result(confirm),
                'no_more_unbounded_fits':True,'no_high_SR_claim':True,
                'final800_deferred_until_real_integration_or_learning_problem_diagnosed':True})
            return
        frozen=make_method(family,selected,root/'FROZEN_METHOD.json',final=True)
        immutable(root/'FINAL_LOCK.json',{'method_sha256':sha256_file(frozen),'seeds':FINAL,
            'DEV_SELECTION_sha256':sha256_file(selection),'CONF_progress_sha256':sha256_file(confirm)})
        final=self.evaluate(frozen,'final800','fresh_secondary_final800',FINAL)
        result=audit_result(final);atomic_json(root/'CERTIFIED_RESULT.json',result)
        self.stage='completed_actual800_certified';self.status('completed_family',per_action=result['per_action'])

    def run(self):
        # Finish all pending integrations before long fits; existing unready
        # depth work does not block the already admitted families.
        admitted={}
        for family in FAMILIES:
            try:
                if not (OUT/family/'DATA_AUDIT.json').exists():continue
                admitted[family]=self.admission(family)
            except Exception as error:
                atomic_json(OUT/family/'WORKFLOW_ERROR.json',{'UTC':utc(),'error':str(error),
                    'traceback':traceback.format_exc(),'NOT_model_SR':True,'other_families_continue':True})
        if 'falcon_esm' not in admitted and not (OUT/'falcon_esm/WORKFLOW_ERROR.json').exists():
            self.family='falcon_esm';self.stage='waiting_real_TRAIN_depth_admission';deadline=time.monotonic()+8*3600
            while not (OUT/'falcon_esm/DEPTH_ADMISSION.json').exists():
                self.status('waiting_native_depth',actual_depth_progress=read(OUT/'falcon_esm/DEPTH_STATUS.json'))
                if time.monotonic()>deadline:raise TimeoutError('Native depth admission8h deadline')
                time.sleep(10)
            label='falcon_native_input_audit_'+str(time.time_ns())
            with (OUT/(label+'.log')).open('x') as log:
                result=subprocess.run([str(ROOT/PYTHONS['falcon_esm']),'-u','-m','VLA.admission_r100.falcon_audit'],
                    cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            if result.returncode:raise RuntimeError('FALCON native RGB-D processor audit failed, inspect '+label)
            try:admitted['falcon_esm']=self.admission('falcon_esm')
            except Exception as error:
                atomic_json(OUT/'falcon_esm/WORKFLOW_ERROR.json',{'UTC':utc(),'error':str(error),
                    'traceback':traceback.format_exc(),'NOT_model_SR':True})
        for family,parent in admitted.items():
            try:self.finish_family(family,parent)
            except Exception as error:
                atomic_json(OUT/family/'WORKFLOW_ERROR.json',{'UTC':utc(),'error':str(error),
                    'traceback':traceback.format_exc(),'NOT_model_SR':True,'other_families_continue':True})
        self.family=None;self.stage='finite_queue_finished';self.status('completed_queue',
            certified_families=[f for f in FAMILIES if (OUT/f/'CERTIFIED_RESULT.json').exists()],
            diagnosed_families=[f for f in FAMILIES if (OUT/f/'WORKFLOW_ERROR.json').exists()])


def cleanup():
    children=read(OUT/'WORKFLOW_CHILDREN.json')['children'] if (OUT/'WORKFLOW_CHILDREN.json').exists() else []
    # Also account for a guard child whose parent/owner died abruptly.
    for p in (OUT/'guard').glob('*_IDENTITY.json'):
        r=read(p).get('identity')
        owner=read(OUT/'WORKFLOW_OWNER.json') if (OUT/'WORKFLOW_OWNER.json').exists() else None
        if r and owner and r['ppid']==owner['pid'] and r not in children:children.append(r)
    stop_all(children)


def main():
    with (OUT/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owner=Owner()
        atomic_json(OUT/'WORKFLOW_OWNER.json',identity(os.getpid()))
        with (OUT/f'workflow_watchdog_{os.getpid()}.log').open('x') as log:
            subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog','--owner-pid',str(os.getpid())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:owner.run()
        except BaseException as error:
            owner.status('technical_error_retained',error=str(error));atomic_json(OUT/'WORKFLOW_ERROR.json',
                {'UTC':utc(),'error':str(error),'traceback':traceback.format_exc(),'NOT_model_SR':True});raise
        finally:cleanup();atomic_json(OUT/f'WORKFLOW_CLEANUP_{os.getpid()}.json',{'UTC':utc(),'R99_signals_sent':0})


def watchdog(pid):
    expected=read(OUT/'WORKFLOW_OWNER.json')
    if expected['pid']!=pid:raise ValueError('Owner identity mismatch')
    while same_owned_process(expected):
        if (OUT/f'WORKFLOW_CLEANUP_{pid}.json').exists():return
        time.sleep(5)
    if (OUT/f'WORKFLOW_CLEANUP_{pid}.json').exists():return
    cleanup();atomic_json(OUT/f'WORKFLOW_WATCHDOG_CLEANUP_{pid}.json',{'UTC':utc(),'R99_signals_sent':0})
    starts=list(OUT.glob('WORKFLOW_RESTART_*.json'))
    if len(starts)<2 and not (OUT/'WORKFLOW_ERROR.json').exists():
        verify(read(OUT/'AUTOMATION_QUALIFICATION.json')['code'])
        with (OUT/f'workflow_restart{len(starts)}.log').open('x') as log:
            p=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/f'WORKFLOW_RESTART_{len(starts)}.json',{'UTC':utc(),'pid':p.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--watchdog',action='store_true')
    p.add_argument('--owner-pid',type=int);a=p.parse_args()
    if a.watchdog:watchdog(a.owner_pid)
    elif a.launch:
        old=OUT/'WORKFLOW_OWNER.json'
        if old.exists() and same_owned_process(read(old)):raise ValueError('Duplicate native workflow owner')
        verify(read(OUT/'AUTOMATION_QUALIFICATION.json')['code'])
        with (OUT/f'workflow_launch_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'WORKFLOW_STATUS.json')},flush=True)
    else:main()
