"""Finish the active native fit, benchmark the terminal checkpoint and close Dita."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import qualified
from VLA.dita_transfer_r84.common import PY
from VLA.cross_vla800_r62.common import resources
from .common import (ROOT, OUT, R86, R90, CP, CPU, SIM, PORT, SEEDS, bindings,
    read, verify, utc, identity, same, alive, live_children, parent_pid,
    proc_state, authorize_signal, reload_only, validate_checkpoint_metadata)

MODULE='VLA.dita_close800_r94.run'
ENV={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
    'TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled'}


def checked_signal(expected,sig,role):
    authorize_signal(expected,sig,role)
    if not same(expected) or not alive(expected['pid']): return False
    os.kill(expected['pid'],sig)
    return True


def wait_exit(expected,seconds=20):
    deadline=time.monotonic()+seconds
    while same(expected) and alive(expected['pid']) and time.monotonic()<deadline: time.sleep(.1)
    return not (same(expected) and alive(expected['pid']))


def log_tail(path,limit=16000):
    with Path(path).open('rb') as stream:
        stream.seek(max(0,Path(path).stat().st_size-limit))
        return stream.read().decode(errors='replace')


def cleanup(reservation,children):
    stopped=[]
    for child in reversed(children):
        expected=child['identity']
        if not same(expected) or not alive(expected['pid']): continue
        checked_signal(expected,signal.SIGTERM,child['role'])
        if not wait_exit(expected):
            checked_signal(expected,signal.SIGKILL,child['role'])
        if not wait_exit(expected,10): raise RuntimeError('Owned child still alive; no competing queue release')
        stopped.append(expected['pid'])
    parent=reservation.get('parent')
    resumed=False
    if parent and same(parent) and proc_state(parent['pid']) in ('T','t'):
        # Only an unfinished reservation can reach this path. The active
        # optimizer was never signaled; this restores its original scheduler.
        resumed=checked_signal(parent,signal.SIGCONT,'queue_parent')
    return {'UTC':utc(),'own_reload_or_benchmark_children_stopped':stopped,
        'original_parent_resumed_on_error':resumed,'active_optimizer_signals_sent':0,
        'cancelled_R90_Dita_continuation_not_restarted':True}


def watchdog():
    with (OUT/'.watchdog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reservation=read(OUT/'RESERVATION.json')
        atomic_json(OUT/'WATCHDOG_READY.json',{'UTC':utc(),'pid':os.getpid(),'owner':reservation['owner']})
        while True:
            if (OUT/'CLEANUP_DONE.json').exists(): return
            if not same(reservation['owner']) or not alive(reservation['owner']['pid']):
                children=read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []
                result=cleanup(reservation,children)
                atomic_json(OUT/'WATCHDOG_RECOVERY.json',{**result,'owner_exited':True})
                return
            time.sleep(10)


class Owner:
    def __init__(self):
        self.stage='preflight'; self.code=bindings(); self.children=[]
        self.server=None; self.evaluator=None; self.held={}; self.watch=None

    def status(self,status='running',**extra):
        atomic_json(OUT/'status.json',{'UTC':utc(),'pid':os.getpid(),'status':status,'stage':self.stage,
            'terminal_checkpoint':str(CP),'target_updates':10000,'planned_trials':800,
            'episodes_per_action':200,'further_Dita_finetune_authorized':False,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'evaluator_pid':self.evaluator.pid if self.evaluator and self.evaluator.poll() is None else None,
            **extra})

    def child(self,cmd,label,role):
        verify(self.code)
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        expected=identity(child.pid)
        if expected is None or expected['ppid']!=os.getpid(): raise RuntimeError('Unregistered own child')
        authorize_signal(expected,signal.SIGTERM,role)
        self.children.append({'identity':expected,'role':role})
        atomic_json(OUT/'CHILDREN.json',{'children':self.children})
        return child

    def reserve(self):
        request=read(OUT/'REQUEST.json'); verify(request['code'])
        parent,optimizer,recovery=request['parent'],request['optimizer'],request.get('idle_recovery')
        state=read(R86/'LIVE.json')
        if (not same(parent) or not same(optimizer) or parent_pid(optimizer['pid'])!=parent['pid']
            or module_name_safe(optimizer)!='VLA.recovery_r85.dita_train'
            or str(CP) not in optimizer['command'] or 'fit' not in optimizer['command']
            or state.get('family')!='dita' or state.get('stage')!='fit10000'
            or state.get('child_pid')!=optimizer['pid'] or state.get('server_pid')):
            raise ValueError('Active native fit identity/stage changed before reservation; no blind signals')
        if recovery and (not same(recovery) or live_children(recovery['pid']) or
            read(R90/'LIVE.json').get('stage')!='waiting_for_current_R86_native_FT_and_DEV_to_finish_NO_preemption'):
            raise ValueError('R90 is not the recorded idle waiting coordinator')
        self.held={'UTC':utc(),'owner':identity(os.getpid()),'parent':parent,'optimizer':optimizer,
            'idle_recovery':recovery,'signal_scope':'scheduler parent only; never optimizer'}
        atomic_json(OUT/'RESERVATION.json',self.held)
        with (OUT/'watchdog.log').open('x') as log:
            self.watch=subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+30
        while not (OUT/'WATCHDOG_READY.json').exists():
            if self.watch.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Cleanup watchdog not ready; no scheduling reservation')
            time.sleep(.2)
        if not same(parent) or not same(optimizer): raise ValueError('Process identity changed')
        checked_signal(parent,signal.SIGSTOP,'queue_parent')
        deadline=time.monotonic()+3
        while proc_state(parent['pid']) not in ('T','t'):
            if time.monotonic()>deadline: raise RuntimeError('Scheduling parent did not acknowledge reservation')
            time.sleep(.02)
        current=read(R86/'LIVE.json')
        if current.get('child_pid')!=optimizer['pid'] or current.get('stage')!='fit10000':
            raise ValueError('Parent stage raced; refuse to adopt')
        if not same(optimizer) or proc_state(optimizer['pid']) in ('T','t'):
            raise ValueError('Existing optimizer must continue untouched')
        if recovery:
            if live_children(recovery['pid']): raise ValueError('Idle recovery acquired a live child')
            checked_signal(recovery,signal.SIGTERM,'idle_recovery')
            if not wait_exit(recovery): raise RuntimeError('Idle R90 did not exit; no forced interruption')
            receipt={'UTC':utc(),'old_owner':recovery,'reason':'user closes Dita after current10k and800',
                'Dita_recovery_fit_cancelled':True,'other_family_recipe_retained':'vlanext_s_only_after_R94',
                'qualified_data_preserved':str(R90/'qualified_recovery'),'successor':str(OUT/'status.json')}
            atomic_json(R90/'SUPERSEDED_DITA_CLOSE_REQUEST.json',receipt)
            atomic_json(R90/'LIVE.json',{'stage':'superseded_user_closed_Dita_other_recipe_preserved',**receipt})
        atomic_json(OUT/'PARENT_HELD_OPTIMIZER_CONTINUES.json',{'UTC':utc(),
            'parent_state':proc_state(parent['pid']),'optimizer_state':proc_state(optimizer['pid']),
            'optimizer_identity':optimizer,'active_optimizer_signals_sent':0})
        atomic_json(ROOT/'VLA/frozen_models/dita_close_request.json',{
            'UTC':utc(),'checkpoint_to_finish':str(CP),'further_Dita_finetune_authorized':False,
            'after_finish':'native_reload_then_exact800_then_close','controller':str(OUT/'status.json')})

    def await_checkpoint(self):
        self.stage='waiting_existing_native_fit10000_no_new_Dita_training'; deadline=time.monotonic()+24*3600
        optimizer=self.held['optimizer']
        while same(optimizer) and alive(optimizer['pid']):
            progress=read(CP/'status.json') if (CP/'status.json').exists() else None
            self.status(actual_training_progress=progress,original_scheduling_parent_only_held=True)
            if time.monotonic()>deadline: raise TimeoutError('Existing fit exceeded finite24h wait; optimizer not killed')
            time.sleep(15)
        meta=qualified(CP); validate_checkpoint_metadata(meta)
        parent=self.held['parent']
        if live_children(parent['pid']): raise ValueError('Scheduler still has a live child; no retirement')
        if same(parent) and alive(parent['pid']):
            checked_signal(parent,signal.SIGTERM,'queue_parent')
            checked_signal(parent,signal.SIGCONT,'queue_parent')
            if not wait_exit(parent): raise RuntimeError('Completed-stage scheduler did not exit')
        receipt={'UTC':utc(),'parent':parent,'optimizer':optimizer,'checkpoint':str(CP),
            'actual_optimizer_updates':meta['updates'],'optimizer_signals_sent':0,
            'native_checkpoint_hashes_verified':True,'remaining_Dita_fit_cancelled':True,
            'successor':str(OUT/'status.json'),'not_a_fictitious_R86_workflow_complete':True}
        atomic_json(OUT/'HANDOFF_COMPLETED.json',receipt)
        atomic_json(R86/'USER_SUPERSEDED.json',receipt)
        atomic_json(R86/'LIVE.json',{'status':'superseded_after_completed_fit_by_user_terminal800',**receipt})
        atomic_json(R86/'dita/owner_status.json',{'status':'superseded_user_requested_terminal800',**receipt})
        return meta

    def resource(self):
        deadline=time.monotonic()+12*3600
        while True:
            result=resources()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
            if not apps and result['GPU_free_MiB']>=13000 and result['RAM_available_MiB']>=5500 and result['disk_free_GiB']>=25:
                return result
            self.status(resource_wait=result)
            if time.monotonic()>deadline: raise TimeoutError('No safe native reload/inference margin')
            time.sleep(15)

    def freeze(self,meta):
        self.stage='native_fresh_CUDA_reload_no_optimizer_updates'; self.status()
        self.resource()
        command=['VLA.recovery_r85.dita_train','reload','--checkpoint',str(CP)]
        reload_only(command)
        child=self.child([PY,'-u','-m',*command],'native_reload','own_reload')
        deadline=time.monotonic()+3600
        while child.poll() is None:
            if time.monotonic()>deadline: raise TimeoutError('Finite fresh native reload limit')
            self.status(reload_pid=child.pid); time.sleep(10)
        if child.returncode or not read(CP/'FRESH_RELOAD_cuda.json')['passed']:
            raise RuntimeError('Fresh native reload failed; checkpoint retained, no reported SR')
        validate_checkpoint_metadata(qualified(CP))
        method=read(R86/'dita/method4096.json')
        verify(method['code']); verify(method['weights_sha256'])
        weights={path:digest for path,digest in method['weights_sha256'].items()
                 if Path(path).resolve()!=R86/'dita/fit4096/weights.safetensors'}
        weights[str(CP/'weights.safetensors')]=meta['weights_sha256']
        method.update(checkpoint=str(CP),weights_sha256=weights,
            code={**method['code'],**meta['code'],**self.code},actual_checkpoint_optimizer_updates=10000,
            frozen_after_confirmation=False,frozen_for_user_requested_benchmark=True,
            terminal_checkpoint_no_further_Dita_finetune=True,
            benchmark_authorization={'user_requested_800_despite_failed_pilot_gate':True,
                'request':'finetune xong thì test800 luôn, dừng không finetune nữa, đóng model',
                'no_new_DEV_or_CONF_promotion_claim':True,'fixed_checkpoint_chosen_before_final_outcomes':True},
            metric='R62 semantic first-hit; NOT strict1cm; PUT release/settle not required',
            paired_scene_block_reused_from_earlier_Dita_NOT_new_independent_block=True)
        from .evaluate import validate
        cohort={'role':'fresh_secondary_final800','seeds':SEEDS,
            'user_requested_benchmark_override':True,'scene_block_reused_for_checkpoint_comparison':True,
            'not_five_measured_seeds':True}
        validate(method,cohort); verify(method['code']); verify(weights)
        atomic_json(OUT/'FROZEN_METHOD.json',method); atomic_json(OUT/'COHORT.json',cohort)
        atomic_json(ROOT/'VLA/frozen_models/dita.json',{'status':'terminal_checkpoint_frozen_benchmark_pending',
            'UTC':utc(),'checkpoint':str(CP),'updates':10000,'weights_sha256':meta['weights_sha256'],
            'method':str(OUT/'FROZEN_METHOD.json'),'method_sha256':sha256_file(OUT/'FROZEN_METHOD.json'),
            'further_finetune_authorized':False,'result_is_not_available_yet':True,
            'controller':str(OUT/'status.json')})

    def benchmark(self):
        from VLA.benchmark_guardian_r89.audit import certify, audit_chain
        from VLA.benchmark_guardian_r89.common import complete_rows
        from VLA.benchmark_guardian_r89.guardian import retryable
        resume=None; deadline=time.monotonic()+36*3600
        for attempt in range(4):
            folder=OUT/f'attempt{attempt:02d}'; folder.mkdir(exist_ok=False)
            self.stage='loading_terminal_native_model'; self.status(attempt=attempt)
            try:
                self.resource()
                self.server=self.child([PY,'-u','-m','VLA.dita_transfer_r84.server',
                    '--method',OUT/'FROZEN_METHOD.json','--port',PORT],f'server_{attempt:02d}','own_benchmark')
                session=requests.Session(); session.trust_env=False
                try:
                    startup=time.monotonic()+1200
                    while True:
                        if self.server.poll() is not None: raise RuntimeError('Native server startup failed')
                        try:
                            response=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2)
                            response.raise_for_status(); health=response.json()
                            if health['method_sha256']!=sha256_file(OUT/'FROZEN_METHOD.json') or health['checkpoint']!=str(CP):
                                raise ValueError('Wrong native method/checkpoint at endpoint')
                            atomic_json(folder/'NATIVE_HEALTH.json',health); break
                        except requests.RequestException:
                            if time.monotonic()>startup: raise TimeoutError('Actual native server readiness')
                            time.sleep(2)
                    args=[SIM,'-u','-m','VLA.dita_close800_r94.evaluate','--method',OUT/'FROZEN_METHOD.json',
                        '--cohort',OUT/'COHORT.json','--output',folder/'final800','--port',PORT]
                    if resume: args+=['--resume-progress',resume]
                    self.evaluator=self.child(args,f'evaluate_{attempt:02d}','own_benchmark')
                    self.stage='actual_terminal_Dita800_no_video'
                    while self.evaluator.poll() is None:
                        if self.server.poll() is not None: raise RuntimeError('Native server exited during actual cohort')
                        if time.monotonic()>deadline: raise TimeoutError('Finite36h total benchmark budget')
                        path=folder/'final800/progress.json'
                        progress=read(path) if path.exists() else None
                        self.status(attempt=attempt,actual_progress={k:v for k,v in progress.items() if k!='rows'} if progress else None)
                        time.sleep(15)
                    if self.evaluator.returncode: raise RuntimeError('Technical evaluator interruption; inspect retained evidence')
                finally:
                    session.close()
            except Exception as error:
                atomic_json(folder/'ERROR.json',{'UTC':utc(),'error':str(error),'traceback':traceback.format_exc(),
                    'NOT_task_failure_or_SR0':True})
                cleanup({},self.children)
                # Shared retry classifier checks these retained log locations.
                logs=[]
                for name in (f'server_{attempt:02d}.log',f'evaluate_{attempt:02d}.log'):
                    path=OUT/name
                    if path.exists(): logs.append(log_tail(path))
                (folder/'evaluate.log').write_text('\n'.join(logs))
                fatal_markers=any((folder/name).exists() for name in ('RECEIPT_MISMATCH.json','PREFIX_DIVERGENCE.json'))
                if (attempt==3 or fatal_markers or time.monotonic()>deadline
                    or not retryable(folder,owner_missing=False,stalled=False)):
                    raise
                path=folder/'final800/progress.json'
                if path.exists():
                    data,rows,keys=complete_rows(path); audit_chain(path)
                    snapshot={**data,'original_progress_path':str(path),'original_progress_sha256':sha256_file(path)}
                    resume=OUT/f'resume_snapshot_{attempt:02d}.json'; atomic_json(resume,snapshot)
                self.stage='bounded_technical_retry_unchanged_model_and_completed_prefix'; self.status(attempt=attempt)
                continue
            finally:
                cleanup({},self.children)
            path=folder/'final800/progress.json'; result=certify(path)
            result.update(checkpoint=str(CP),updates=10000,terminal_model_closed=True,
                further_finetune_authorized=False,scene_block_reused_from_earlier_Dita=True)
            atomic_json(OUT/'CERTIFIED_RESULT.json',result)
            manifest=read(ROOT/'VLA/frozen_models/dita.json')
            manifest.update(status='closed_after_actual800',UTC=utc(),result_is_not_available_yet=False,
                result=str(OUT/'CERTIFIED_RESULT.json'),result_sha256=sha256_file(OUT/'CERTIFIED_RESULT.json'),
                actual_trials=800,per_action=result['per_action'])
            atomic_json(ROOT/'VLA/frozen_models/dita.json',manifest)
            report='# Dita10,000 — terminal checkpoint, actual800\n\n| Action | Successes / n | SR |\n|---|---:|---:|\n'
            for task,value in result['per_action'].items():
                report+=f"| {task} | {value['successes']}/{value['n']} | {value['SR_percent']:.2f}% |\n"
            report+='\nNo further Dita fine-tuning. One actual800 cohort,200/action; same paired scene block as earlier Dita checkpoints, not five measured seeds.\n'
            report+='R62 semantic first-hit, NOT strict1cm; PUT release/settle not required. No independent CONF promotion is claimed.\n'
            (OUT/'RESULTS.md').write_text(report)
            self.stage='terminal_Dita_closed_actual800_certified'; self.status('completed',per_action=result['per_action'])
            return
        raise RuntimeError('Technical attempt budget exhausted; no invented completion')


def module_name_safe(expected):
    from .common import module_name
    return module_name(expected)


def main():
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'STARTED.json').exists(): raise FileExistsError('No duplicate terminal model controller')
        qualification=read(OUT/'QUALIFICATION.json')
        if not qualification['passed'] or qualification['code']!=bindings():
            raise ValueError('Current terminal-controller CPU qualification required')
        owner=Owner(); atomic_json(OUT/'STARTED.json',{'UTC':utc(),'pid':os.getpid(),'code':owner.code})
        success=False
        try:
            owner.reserve(); meta=owner.await_checkpoint(); owner.freeze(meta); owner.benchmark(); success=True
        except BaseException as error:
            atomic_json(OUT/'ERROR.json',{'UTC':utc(),'stage':owner.stage,'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('technical_error_evidence_retained_NOT_completed800',error=str(error)); raise
        finally:
            atomic_json(OUT/'CLEANUP_DONE.json',cleanup(owner.held,owner.children))
        if success:
            # Preserve the already authorized other-model queue without ever
            # restarting R90's cancelled Dita recipe.
            with (OUT/'other_models_launch.log').open('x') as log:
                process=subprocess.Popen([str(CPU),'-u','-m','VLA.dita_close800_r94.other_models'],
                    cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(OUT/'OTHER_MODELS_QUEUE.json',{'UTC':utc(),'pid':process.pid,
                'families':['vlanext_s'],'Dita_excluded':True,'no_new_Dita_fit':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--launch',action='store_true'); p.add_argument('--watchdog',action='store_true')
    args=p.parse_args()
    if args.watchdog: watchdog()
    elif args.launch:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'status.json'),'no_further_Dita_fit':True})
    else: main()
