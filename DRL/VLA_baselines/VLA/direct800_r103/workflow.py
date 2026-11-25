"""Adopt the finishing real trainer, then direct neural800; no SR gate."""
import argparse,fcntl,os,signal,subprocess,time,traceback
from pathlib import Path

from VLA.admission_r100.common import ROOT,OUT as R100,CPU,ENV,read,utc,atomic_json,verify
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import identity,alive
from VLA.admission_r100.guard import same_owned_process

OUT=ROOT/'VLA/audits/direct800_r103_20261025'
MODULE='VLA.direct800_r103.workflow'


def configure():
    # Process-local output routing only. No live source/module on disk changes.
    import VLA.admission_r100.workflow as base
    import VLA.admission_r100.guard as guard
    base.OUT=guard.OUT=OUT
    return base


def terminal(expected):
    if not same_owned_process(expected):return True
    path=Path('/proc')/str(expected['pid'])/'stat'
    try:return path.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
    except FileNotFoundError:return True


def retire_scheduler(expected,watchdog=False):
    if terminal(expected):return
    cmd=expected['command']
    if '-m' not in cmd or cmd[cmd.index('-m')+1]!='VLA.admission_r100.workflow':
        raise ValueError('Never signal a native trainer/server or unrelated process during handoff')
    if ('--watchdog' in cmd)!=watchdog:raise ValueError('Wrong scheduler/watchdog role')
    os.kill(expected['pid'],signal.SIGTERM)
    if not watchdog and same_owned_process(expected):os.kill(expected['pid'],signal.SIGCONT)
    deadline=time.monotonic()+15
    while not terminal(expected) and time.monotonic()<deadline:time.sleep(.1)
    if not terminal(expected):raise RuntimeError('Old coordinator retirement did not complete')


def handoff():
    if (OUT/'COORDINATOR_HANDOFF_COMPLETE.json').exists():return
    record=read(OUT/'HANDOFF.json')
    # Retire its watchdog FIRST, so retirement cannot restart the old workflow
    # or clean up the still-running trainer.
    retire_scheduler(record['old_watchdog'],watchdog=True)
    retire_scheduler(record['old_owner'])
    atomic_json(OUT/'COORDINATOR_HANDOFF_COMPLETE.json',{'UTC':utc(),
        'actual_trainer_identity':record['adopted_trainer'],'trainer_signals_sent':0,
        'old_watchdog_and_scheduler_retired':True})
    atomic_json(R100/'DIRECT800_HANDOFF.json',{'UTC':utc(),'family':'falcon_esm',
        'new_status':str(OUT/'WORKFLOW_STATUS.json'),'new_report':str(OUT/'REPORT.md'),
        'new_owner_pid':os.getpid(),'trainer_signals_sent':0})


def actual_checkpoint():
    from VLA.admission_r100.train import qualified
    root=Path(read(OUT/'HANDOFF.json')['training_output'])
    complete=root/'COMPLETED.json'
    if complete.exists():
        cp=Path(read(complete)['checkpoint'])
        if qualified(cp,'falcon_esm')['updates']!=2050:raise ValueError('Direct run requires the actual completed2050 checkpoint')
        return cp
    latest=root/'LATEST_COMMITTED.json'
    return Path(read(latest)['checkpoint']) if latest.exists() else None


def direct_lock(checkpoint):
    base=configure();folder=OUT/'falcon_esm';folder.mkdir(exist_ok=True)
    a=read(OUT/'DIRECT_AUTHORIZATION.json')
    if a['family']!='falcon_esm' or a['episodes_per_action']!=200 or a['additional_performance_finetuning']:
        raise ValueError('Wrong direct benchmark authorization')
    temp=base.make_method('falcon_esm',checkpoint,folder/'BASE_FROZEN_METHOD.json',final=True)
    method={**read(temp),'direct800_user_authorized':True,
        'performance_readiness_gate_bypassed_by_user':True,'DEV_CONF_performance_NOT_claimed_passed':True,
        'checkpoint_selection':'last completed current2050 fit; no final outcome used',
        'code':{**read(temp)['code'],**read(OUT/'AUTOMATION_QUALIFICATION.json')['direct_code']}}
    frozen=base.immutable(folder/'FROZEN_METHOD.json',method)
    from VLA.admission_r100.evaluate import FINAL
    base.immutable(folder/'FINAL_LOCK.json',{'method_sha256':sha256_file(frozen),'seeds':FINAL,
        'user_direct_authorization_sha256':sha256_file(OUT/'DIRECT_AUTHORIZATION.json'),
        'no_low_SR_failure_filtering':True,'original_thresholds_unchanged':True})
    return frozen


def owner_class():
    base=configure()
    class DirectOwner(base.Owner):
        def status(self,status='running',**extra):
            super().status(status,**extra)
            state=read(OUT/'WORKFLOW_STATUS.json');progress=state.get('actual_progress') or {}
            lines=['# R103: FALCON direct800 after current fit','',f'UTC: {utc()}',
                f"Stage: {self.stage}; status: {status}.",'',
                'User authorized direct800 without a DEV/CONF performance gate. '
                'Technical reload, original safety guards and independent outcome/query audits remain required.','']
            if extra.get('actual_trainer_status'):
                t=extra['actual_trainer_status'];lines.append(f"Actual existing trainer: {t.get('updates')}/{t.get('target_updates')} updates.")
            per=progress.get('per_action',extra.get('per_action',{}))
            if per:
                lines+=['','|Action|Successes/n|SR|','|---|---:|---:|']
                for task,v in per.items():lines.append(f"|{task}|{v['successes']}/{v['n']}|{v['SR_percent']:.2f}%|")
            lines+=['','800 actual episodes =200/action; same frozen R100 final cohort. No video. '
                'Original semantic metric is NOT strict1cm; PUT release/settle not required. '
                'Low-SR outcomes are retained. These are not five measured seeds.']
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

        def wait_for_fit(self):
            self.family='falcon_esm';self.stage='adopting_existing_finetune'
            rec=read(OUT/'HANDOFF.json');expected=rec['adopted_trainer']
            root=Path(rec['training_output']);deadline=time.monotonic()+36*3600
            while not (root/'COMPLETED.json').exists() and not terminal(expected):
                verify(self.bound)
                if time.monotonic()>deadline:raise TimeoutError('Actual trainer completion deadline')
                t=read(root/'status.json');self.status(actual_trainer_status=t,trainer_signals_sent=0)
                if time.time()-(root/'status.json').stat().st_mtime>900:
                    raise RuntimeError('Existing actual trainer heartbeat stale15min; diagnosis required')
                time.sleep(2)
            cp=actual_checkpoint()
            from VLA.admission_r100.train import qualified
            # Only resume an unfinished numerical budget, never add another fit
            # chosen because a DEV score was low.
            for attempt in range(3):
                if cp is not None and qualified(cp,'falcon_esm')['updates']==2050:return cp
                if cp is None:raise RuntimeError('No complete optimizer/RNG checkpoint to resume')
                output=OUT/'falcon_esm'/f'finish_current_fit_attempt{attempt:02d}'
                if output.exists():
                    p=output/'LATEST_COMMITTED.json'
                    if p.exists():cp=Path(read(p)['checkpoint'])
                    continue
                remaining=2050-qualified(cp,'falcon_esm')['updates']
                try:self.gpu('falcon_esm',['--family','falcon_esm','--resume',cp,'--output',output,
                    '--additional-updates',remaining],f'finish_current_fit{attempt:02d}',
                    module='VLA.admission_r100.train',after_primary=True,seconds=36*3600)
                except base.WorkerInterrupted:
                    if (output/'LATEST_COMMITTED.json').exists():cp=Path(read(output/'LATEST_COMMITTED.json')['checkpoint'])
                    continue
                cp=Path(read(output/'COMPLETED.json')['checkpoint'])
            if cp is None or qualified(cp,'falcon_esm')['updates']!=2050:raise RuntimeError('Actual fit completion failed')
            return cp

        def run_direct(self):
            self.family='falcon_esm';(OUT/self.family).mkdir(exist_ok=True)
            final=OUT/self.family/'CERTIFIED_RESULT.json'
            if final.exists():base.audit_result(Path(read(final)['source']));return
            cp=self.wait_for_fit()
            self.gpu(self.family,['--family',self.family,'--op','reload','--checkpoint',cp],
                'completed_fit_fresh_reload',after_primary=True)
            reload=read(cp/'FRESH_RELOAD.json')
            if not reload['passed'] or len(reload['max_errors'])!=8 or max(reload['max_errors'])>1e-6:
                raise ValueError('Do not bypass actual checkpoint reload integrity')
            frozen=direct_lock(cp)
            from VLA.admission_r100.evaluate import FINAL
            progress=self.evaluate(frozen,'final800','fresh_secondary_final800',FINAL)
            result=base.audit_result(progress);atomic_json(final,result)
            self.stage='completed_actual800_certified';self.status('completed',per_action=result['per_action'])
    return DirectOwner


def main():
    with (OUT/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner=owner_class()();atomic_json(OUT/'WORKFLOW_OWNER.json',identity(os.getpid()))
        handoff()
        with (OUT/f'watchdog_{os.getpid()}.log').open('x') as log:
            subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog','--owner-pid',str(os.getpid())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:owner.run_direct()
        except BaseException as error:
            owner.status('technical_error_retained',error=str(error))
            atomic_json(OUT/'WORKFLOW_ERROR.json',{'UTC':utc(),'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True});raise
        finally:
            configure().cleanup();atomic_json(OUT/f'CLEANUP_{os.getpid()}.json',{'UTC':utc(),'adopted_trainer_signals_sent':0})


def watchdog(pid):
    expected=read(OUT/'WORKFLOW_OWNER.json')
    if expected['pid']!=pid:raise ValueError('Wrong direct owner identity')
    while same_owned_process(expected):
        if (OUT/f'CLEANUP_{pid}.json').exists():return
        time.sleep(5)
    if (OUT/f'CLEANUP_{pid}.json').exists():return
    configure().cleanup()
    starts=list(OUT.glob('RESTART_*.json'))
    if len(starts)<2 and not (OUT/'WORKFLOW_ERROR.json').exists():
        with (OUT/f'restart{len(starts)}.log').open('x') as log:
            p=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/f'RESTART_{len(starts)}.json',{'UTC':utc(),'pid':p.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--watchdog',action='store_true')
    p.add_argument('--owner-pid',type=int);a=p.parse_args()
    if a.watchdog:watchdog(a.owner_pid)
    elif a.launch:
        verify(read(OUT/'AUTOMATION_QUALIFICATION.json')['code'])
        if (OUT/'WORKFLOW_OWNER.json').exists() and same_owned_process(read(OUT/'WORKFLOW_OWNER.json')):
            raise ValueError('Duplicate direct owner')
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'WORKFLOW_STATUS.json')},flush=True)
    else:main()
