"""One final authentic fit, DEV selection, CONF and durable fresh800."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
from .common import (OUT,R100,PARENT,CPU,PYTHON,ENV,ROOT,TASKS,DEV,CONF,SMOKE,FINAL,
    TARGET_UPDATES,MODULE,read,utc,atomic_json,verify,bindings,percentage)
from .evaluate import audit_result
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import identity,alive
from VLA.admission_r100.guard import same_owned_process
from VLA.dita_terminal_recovery_r95.identity import await_identity

ALLOWED = {'VLA.admission_r100.train','VLA.admission_r100.native_gate',
    'VLA.admission_r100.server','VLA.falcon_final_r105.evaluate',
    'VLA.falcon_final_r105.preflight'}


def pending_records(progress,records):
    completed={(r['seed'],r['task']) for r in progress['rows']}
    return [r for r in records if (r['seed'],r['task']) not in completed]


def journal_records(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines(keepends=True) if line.endswith('\n')]


def verify_resume(old,new):
    import numpy as np
    before=read(old);after=read(new)
    pending=pending_records(before,journal_records(Path(old).parent/'NEURAL_QUERIES.jsonl'))
    replay={(r['seed'],r['task'],r['index']):r for r in journal_records(Path(new).parent/'NEURAL_QUERIES.jsonl')}
    errors=[]
    if len(pending)<6 or after.get('reused_completed_trials',0)<before['completed_trials']:
        raise ValueError('Actual completed-outcome and pending-prefix recovery was not exercised')
    for r in pending:
        actual=replay[(r['seed'],r['task'],r['index'])]
        if r['input_sha256']!=actual['input_sha256']:
            raise ValueError('Real resumed RGB/state/depth/token input hash differs')
        error=float(np.max(np.abs(np.asarray(r['actions'])-np.asarray(actual['actions']))))
        if error>1e-6:raise ValueError('Real resumed native action prefix differs')
        errors.append(error)
    return {'passed':True,'actual_reused_completed_trials':after['reused_completed_trials'],
        'actual_pending_native_queries_compared':len(errors),'max_native_action_error':max(errors),
        'all_pending_input_hashes_equal':True,'restored_trials_NOT_counted_twice':True,
        'source_progress':str(old),'resumed_progress':str(new)}


def stop_owned(children):
    owned=[r for r in children if r and same_owned_process(r)]
    for r in reversed(owned):
        cmd=r['command']
        if '-m' not in cmd or cmd[cmd.index('-m')+1] not in ALLOWED:
            raise ValueError('Refuse signal to unrelated/legacy process')
        os.kill(r['pid'],signal.SIGTERM)
    deadline=time.monotonic()+20
    while any(same_owned_process(r) for r in owned) and time.monotonic()<deadline:
        time.sleep(.1)
    for r in owned:
        if same_owned_process(r):os.kill(r['pid'],signal.SIGKILL)


def configure():
    import VLA.admission_r100.workflow as base
    import VLA.admission_r100.guard as guard
    base.OUT=guard.OUT=OUT
    base.audit_result=audit_result
    base.stop_all=stop_owned
    return base


def make_method(checkpoint,destination,final=False):
    base=configure();destination=Path(destination)
    raw=base.make_method('falcon_esm',checkpoint,destination.with_suffix('.base.json'),final=final)
    value={**read(raw),'code':{**read(raw)['code'],**bindings()},
        'R105_new_holdout_frozen_before_final':bool(final),
        'parent2050_old800_diagnostic_NOT_checkpoint_selection':True,
        'single_final_continuation_target_updates':TARGET_UPDATES,
        'CPU_AUDIT_sha256':sha256_file(OUT/'CPU_AUDIT.json'),
        'no_new_policy_geometry_or_action_assist':True}
    return base.immutable(destination,value)


def summary(state):
    progress=state.get('actual_progress') or {}
    final=state.get('certified_result') or {}
    per=progress.get('per_action') or final.get('per_action') or state.get('per_action') or {}
    complete=progress.get('completed_trials',final.get('actual_trials'))
    lines=['# FALCON-ESM R105: final bounded continuation','',
        f"UTC: {state.get('UTC')}",f"Stage: {state.get('stage')}; status: {state.get('status')}.",
        '', 'Actual numerical budget2050→4098; no additional low-score refitting. '
        'DEV chooses only between2050/4098; CONF is separate. '
        'The fresh800 is run after all technical gates, even if the score is low.']
    if complete is not None:
        lines+=['',f"Actual episodes completed: {complete}/{progress.get('planned_trials',800)}."]
    if state.get('actual_trainer_status'):
        t=state['actual_trainer_status']
        lines+=['',f"Actual trainer updates: {t.get('updates')}/{t.get('target_updates')}; loss: {t.get('loss','—')}."]
    if per:
        lines+=['','|Action|Successes/n|SR|','|---|---:|---:|']
        for task in TASKS:
            if task in per:
                v=per[task];lines.append(f"|{task}|{v['successes']}/{v['n']}|{percentage(v.get('SR_percent'))}|")
    lines+=['','Original R62 first-hit metric; NOT strict1cm; PUT release/settle not required. '
        'Same robot, physics, cameras, depth, causal67, native actions and safety. '
        'This explicitly privileged adaptation is not an unmodified image/text-only baseline. '
        'No five-seed results are invented. Original R104800 evidence is preserved.']
    return '\n'.join(lines)+'\n'


def live_snapshot():
    """Read-only worker observer; unlike a phase-start file this stays live."""
    path=OUT/'WORKFLOW_STATUS.json'
    if not path.exists():return None
    state=read(path)
    state['observer_UTC']=utc()
    if state.get('status') not in ('completed','technical_error_retained'):
        actual=[]
        for p in (OUT/'falcon_esm').glob('*/status.json'):
            value=read(p);pid=value.get('pid');record=identity(pid) if pid and alive(pid) else None
            if record and 'VLA.admission_r100.train' in record['command']:
                actual.append((p.stat().st_mtime_ns,str(p),value,record))
        if actual:
            _,path,value,record=max(actual)
            state.update(actual_trainer_status=value,actual_trainer_identity=record,actual_trainer_status_file=path)
    atomic_json(OUT/'LIVE_STATUS.json',state)
    (OUT/'REPORT.md').write_text(summary(state))
    return state


def owner_class():
    base=configure()
    class FinalOwner(base.Owner):
        def status(self,status='running',**extra):
            base.Owner.status(self,status,**extra)
            (OUT/'REPORT.md').write_text(summary(read(OUT/'WORKFLOW_STATUS.json')))

        def spawn(self,python,module,args,label):
            verify(self.bound)
            if module=='VLA.admission_r100.evaluate':module='VLA.falcon_final_r105.evaluate'
            if module not in ALLOWED:raise ValueError('Unapproved new native worker')
            command=list(map(str,[python,'-u','-m',module,*args]))
            with (OUT/self.family/(label+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
            try:r,observations=await_identity(child,command,os.getpid(),identity)
            except BaseException:
                if child.poll() is None:child.terminate();child.wait(timeout=20)
                raise
            self.children.append(r)
            atomic_json(OUT/'WORKFLOW_CHILDREN.json',{'children':self.children})
            atomic_json(OUT/self.family/(label+'_IDENTITY.json'),{'identity':r,'startup_observations':observations})
            return child

        def wait(self,child,label,folder=None,seconds=3600):
            # Deliberately interrupt ONLY our four-episode technical smoke,
            # after a completed outcome and >=6 pending real neural queries.
            # The actual800 never receives this injected signal.
            probe=OUT/'CONTROLLED_SMOKE_INTERRUPTION.json'
            enabled=bool(folder and label.startswith('integration4_attempt') and not probe.exists())
            done=threading.Event();errors=[]
            def inject():
                try:
                    while not done.wait(.15) and child.poll() is None:
                        p=Path(folder)/'progress.json';q=Path(folder)/'NEURAL_QUERIES.jsonl'
                        if not p.exists() or not q.exists():continue
                        value=read(p);pending=pending_records(value,journal_records(q))
                        if 1<=value['completed_trials']<4 and len(pending)>=6:
                            r=next(r for r in self.children if r['pid']==child.pid)
                            if not same_owned_process(r):raise ValueError('Smoke child PID ownership changed')
                            atomic_json(probe,{'UTC':utc(),'identity':r,'progress':str(p),
                                'completed_trials':value['completed_trials'],'pending_queries_at_signal':len(pending),
                                'technical_probe_NOT_model_failure':True,'no_signal_to_actual800':True})
                            os.kill(child.pid,signal.SIGTERM)
                            return
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=inject,daemon=True) if enabled else None
            if thread:thread.start()
            try:return super().wait(child,label,folder,seconds)
            finally:
                done.set()
                if thread:thread.join(timeout=5)
                if errors:raise errors[0]

        def preflight(self,checkpoint,label,inspect=False):
            output=OUT/'falcon_esm'/label
            proof=output/'GPU_PREFLIGHT.json'
            if proof.exists():
                value=read(proof)
                if not value['passed'] or value['weights_sha256']!=sha256_file(Path(checkpoint)/'weights.safetensors'):
                    raise ValueError('Wrong saved CUDA preflight checkpoint')
                return value
            self.stage=label;self.resource(after_primary=True);self.status()
            args=['--checkpoint',checkpoint,'--output',output]
            if inspect:args.append('--inspect')
            child=self.spawn(PYTHON,'VLA.falcon_final_r105.preflight',args,label+'_'+str(time.time_ns()))
            self.child=child
            try:self.wait(child,label,seconds=3600)
            finally:
                stop_owned([r for r in self.children if r['pid']==child.pid]);self.child=None
            value=read(proof)
            if not value['passed']:raise ValueError('Real CUDA/native input gate failed')
            return value

        def fixture(self):
            finish=OUT/'falcon_esm/TRAINER_FIXTURE.json'
            if finish.exists():return read(finish)
            # This runs the exact production trainer/optimizer, but its two
            # fixture updates are discarded. The real fit starts at parent2050.
            from VLA.admission_r100.train import qualified
            for attempt in range(4):
                output=OUT/'falcon_esm'/f'trainer_fixture_attempt{attempt:02d}'
                if (output/'COMPLETED.json').exists():
                    cp=Path(read(output/'COMPLETED.json')['checkpoint']);break
                if output.exists():continue
                try:self.gpu('falcon_esm',['--family','falcon_esm','--resume',PARENT,
                    '--output',output,'--additional-updates',2],f'trainer_fixture{attempt:02d}',
                    module='VLA.admission_r100.train',after_primary=True)
                except base.WorkerInterrupted:continue
                cp=Path(read(output/'COMPLETED.json')['checkpoint']);break
            else:raise RuntimeError('Bounded trainer-fixture attempts exhausted')
            if qualified(cp,'falcon_esm')['updates']!=2052:
                raise ValueError('Wrong optimizer-resume fixture budget')
            proof=self.preflight(cp,'fixture_fresh_reload')
            value={'passed':True,'checkpoint':str(cp),'additional_fixture_updates':2,
                'effective_batch':16,'actual_optimizer_RNG_resume_and_fresh_reload':proof,
                'fixture_NOT_selected_and_NOT_parent_of_final_fit':True}
            atomic_json(finish,value);return value

        def fit(self,family,parent,phase):
            # Reuse the committed optimizer/RNG recovery logic, but expose
            # the actual child trainer progress, not only its phase label.
            return super().fit(family,parent,phase)

        def run_final(self):
            from VLA.admission_r100.train import qualified
            from VLA.autonomy_r83.common import score,ready
            self.family='falcon_esm';(OUT/self.family).mkdir(exist_ok=True)
            if not read(OUT/'CPU_AUDIT.json')['passed']:
                raise ValueError('All-frame numeric admission has not passed')
            self.preflight(PARENT,'parent_cuda_input_preflight',inspect=True)
            self.fixture()
            smoke_method=make_method(PARENT,OUT/self.family/'SMOKE_METHOD.json')
            smoke=self.evaluate(smoke_method,'integration4','development_selection',SMOKE)
            interruption=read(OUT/'CONTROLLED_SMOKE_INTERRUPTION.json')
            atomic_json(OUT/self.family/'ACTUAL_RESUME_PROOF.json',verify_resume(Path(interruption['progress']),smoke))
            self.stage='awaiting_actual_image_review'
            deadline=time.monotonic()+3600
            while not (OUT/'VISUAL_REVIEW.json').exists():
                self.status('waiting_actual_image_review',image_directory=str(OUT/self.family/'parent_cuda_input_preflight'))
                if time.monotonic()>deadline:raise TimeoutError('Real native image review required before the final fit')
                time.sleep(2)
            visual=read(OUT/'VISUAL_REVIEW.json')
            if not visual.get('passed') or visual.get('GPU_PREFLIGHT_sha256')!=sha256_file(
                OUT/self.family/'parent_cuda_input_preflight/GPU_PREFLIGHT.json'):
                raise ValueError('Actual image review missing or belongs to another preprocessing run')
            atomic_json(OUT/self.family/'TECHNICAL_READY.json',{'passed':True,'UTC':utc(),
                'CPU_AUDIT_sha256':sha256_file(OUT/'CPU_AUDIT.json'),
                'VISUAL_REVIEW_sha256':sha256_file(OUT/'VISUAL_REVIEW.json'),
                'real_four_action_integration':audit_result(smoke),
                'actual_resume_fixture':read(OUT/self.family/'TRAINER_FIXTURE.json'),
                'high_SR_NOT_implied':True})
            cp=self.fit('falcon_esm',PARENT,'final_continuation2048')
            if qualified(cp,'falcon_esm')['updates']!=TARGET_UPDATES:
                raise ValueError('The single final continuation exceeded or missed4098')
            candidates=[]
            for checkpoint,tag in [(PARENT,'2050'),(cp,'4098')]:
                method=make_method(checkpoint,OUT/self.family/f'DEV_{tag}_METHOD.json')
                progress=self.evaluate(method,'dev40_'+tag,'development_selection',DEV)
                candidates.append({'checkpoint':str(checkpoint),'progress':str(progress),'score':list(score(read(progress)))})
            chosen=max(candidates,key=lambda x:tuple(x['score']))
            selection=base.immutable(OUT/self.family/'DEV_SELECTION.json',{'checkpoint':chosen['checkpoint'],
                'candidates':candidates,'criterion':'DEV40 total, worst action, fewer safety aborts; tie earlier2050',
                'old_final800_new_CONF_and_new_final_NOT_used_for_checkpoint_selection':True})
            selected=Path(chosen['checkpoint'])
            conf_method=make_method(selected,OUT/self.family/'CONF_METHOD.json')
            confirm=self.evaluate(conf_method,'confirmation40','development_confirmation',CONF)
            final_method=make_method(selected,OUT/self.family/'FROZEN_METHOD.json',final=True)
            base.immutable(OUT/self.family/'FINAL_LOCK.json',{'method_sha256':sha256_file(final_method),
                'seeds':FINAL,'DEV_SELECTION_sha256':sha256_file(selection),
                'CONF_sha256':sha256_file(confirm),'CONF_readiness_passed':ready(read(confirm)),
                'low_CONF_score_does_NOT_trigger_another_fit_or_discard_final':True})
            progress=self.evaluate(final_method,'final800','fresh_secondary_final800',FINAL)
            result=audit_result(progress)
            atomic_json(OUT/self.family/'CERTIFIED_RESULT.json',result)
            self.stage='completed_actual800_certified'
            self.status('completed',certified_result=result)
    return FinalOwner


def cleanup():configure().cleanup()


def main():
    with (OUT/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner=owner_class()();atomic_json(OUT/'WORKFLOW_OWNER.json',identity(os.getpid()))
        with (OUT/f'watchdog_{os.getpid()}.log').open('x') as log:
            subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog','--owner-pid',str(os.getpid())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:owner.run_final()
        except BaseException as error:
            owner.status('technical_error_retained',error=str(error))
            atomic_json(OUT/'WORKFLOW_ERROR.json',{'UTC':utc(),'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True})
            raise
        finally:
            cleanup();atomic_json(OUT/f'CLEANUP_{os.getpid()}.json',{'UTC':utc(),'unrelated_processes_signaled':0})


def watchdog(pid):
    expected=read(OUT/'WORKFLOW_OWNER.json')
    if expected['pid']!=pid:raise ValueError('Wrong owner/watchdog identity')
    while same_owned_process(expected):
        live_snapshot()
        if (OUT/f'CLEANUP_{pid}.json').exists():return
        time.sleep(5)
    live_snapshot()
    if (OUT/f'CLEANUP_{pid}.json').exists():return
    cleanup()
    restarts=list(OUT.glob('RESTART_*.json'))
    if len(restarts)<2 and not (OUT/'WORKFLOW_ERROR.json').exists():
        with (OUT/f'restart{len(restarts)}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/f'RESTART_{len(restarts)}.json',{'UTC':utc(),'pid':child.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--watchdog',action='store_true')
    p.add_argument('--owner-pid',type=int);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.watchdog:watchdog(a.owner_pid)
    elif a.launch:
        qualification=read(OUT/'AUTOMATION_QUALIFICATION.json');verify(qualification['code'])
        if not read(OUT/'CPU_AUDIT.json')['passed']:raise ValueError('CPU audit incomplete')
        if (OUT/'WORKFLOW_OWNER.json').exists() and same_owned_process(read(OUT/'WORKFLOW_OWNER.json')):
            raise ValueError('Duplicate GPU workflow')
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'WORKFLOW_STATUS.json')},flush=True)
    else:main()
