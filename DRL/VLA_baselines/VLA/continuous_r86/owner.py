"""Adopt at an episode-cohort boundary; interleave already qualified models.

No live source edits, no speculative model adapter and no unbounded self-patching.
Actual neural runners/native recipes remain R84/R85. Audit every completed stage.
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
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.recovery_r85.common import ROOT, R84, OUT as R85, read, utc, verify
import VLA.recovery_r85.owner as prior
from .audit import cohort_audit

OUT = ROOT/'VLA/audits/continuous_r86_20261021'
TASKS = ('touch','pick','move','pick_place')


def bindings():
    return {str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}


def identity(pid):
    """PID reuse guard. Only explicit, validated identities may receive signals."""
    p = Path('/proc')/str(int(pid))
    try:
        words = (p/'stat').read_text().rsplit(')',1)[1].split()
        command = (p/'cmdline').read_bytes().split(b'\0')
        return {'pid':int(pid),'ppid':int(words[1]),'start_ticks':words[19],
                'command':[s.decode() for s in command if s]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(pid):
    if not pid: return False
    try:
        state=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()[0]
        return state not in ('Z','X')
    except (FileNotFoundError, ProcessLookupError):
        return False


def checked_signal(expected, sig):
    if expected is None or identity(expected['pid']) != expected:
        raise ValueError('Process identity changed; no signal sent')
    allowed = {'VLA.recovery_r85.owner','VLA.replacement_r84.server','VLA.dita_transfer_r84.server'}
    if not any(m in expected['command'] for m in allowed):
        raise ValueError('Not one of the specifically adopted study processes')
    os.kill(expected['pid'],sig)


def snapshot_old():
    result={}
    for family,folder in (('vlanext_s',R85),('dita',R85/'dita')):
        state=folder/'owner_status.json'
        if state.exists(): result[family]=read(state)
    for family,path in (('vlanext_s',R85/'vlanext_s/fit2048/status.json'),
                        ('dita',R85/'dita/fit1024/status.json')):
        if path.exists():result.setdefault(family,{})['actual_child_progress']=read(path)
    return result


def handoff(expected):
    """Do not cancel a fit or partial cohort. Pause parent only at DEV stage.

    The child keeps testing while the parent is paused, so it cannot dispatch
    the following long fit between the cohort completing and our handoff.
    """
    deadline=time.monotonic()+72*3600
    paused=False
    try:
        while alive(expected['pid']):
            if identity(expected['pid'])!=expected:raise ValueError('Prior owner PID reused')
            states=[('vlanext_s',2048,read(R85/'owner_status.json'),R85/'vlanext_s')]
            if (R85/'dita/owner_status.json').exists():
                states.append(('dita',1024,read(R85/'dita/owner_status.json'),R85/'dita'))
            atomic_json(OUT/'LIVE.json',{'status':'monitoring_existing_actual_fit_before_boundary_handoff',
                'UTC':utc(),'pid':os.getpid(),'old_owner':snapshot_old(),
                'no_training_or_incomplete_episode_cancelled':True})
            candidate=next(((family,step,state,folder) for family,step,state,folder in states
                if state['status']=='running' and
                    ((state['stage']==f'dev40_fit{step}' and state.get('child_pid')) or
                     (state['stage']==f'dev40_fit{step}_server' and state.get('server_pid')))),None)
            if candidate:
                family,step,state,folder=candidate
                checked_signal(expected,signal.SIGSTOP);paused=True
                fresh=read(R85/'owner_status.json' if family=='vlanext_s' else R85/'dita/owner_status.json')
                if (fresh['stage'],fresh.get('child_pid'))!=(state['stage'],state['child_pid']):
                    checked_signal(expected,signal.SIGCONT);paused=False
                    continue
                server=identity(state['server_pid']) if state.get('server_pid') else None
                if server and server['ppid']!=expected['pid']:
                    raise ValueError('Recorded policy server is not a child of the adopted owner')
                child=int(state['child_pid']) if state.get('child_pid') else None
                target=folder/f'dev40_fit{step}/progress.json'
                while alive(child):
                    atomic_json(OUT/'LIVE.json',{'status':'parent_held_at_boundary_child_still_evaluating',
                        'UTC':utc(),'pid':os.getpid(),'prior_child_pid':child,
                        'actual_progress':read(target) if target.exists() else None})
                    if time.monotonic()>deadline:raise TimeoutError('Finite handoff budget')
                    time.sleep(15)
                # A failed child is explicitly retained as integration evidence;
                # it must never be accepted as a completed zero-SR cohort.
                outcome=read(target) if target.exists() else {'status':'cohort_not_started_NOT_model_SR'}
                if outcome.get('status')=='completed_actual_development':
                    atomic_json(OUT/f'{family}_{step}_PRE_HANDOFF_AUDIT.json',cohort_audit(target,require_complete=True))
                elif child:
                    atomic_json(OUT/'PRIOR_EVALUATION_ERROR.json',{'progress':outcome,'NOT_model_SR':True})
                if server and alive(server['pid']):checked_signal(server,signal.SIGTERM)
                checked_signal(expected,signal.SIGTERM)
                checked_signal(expected,signal.SIGCONT);paused=False
                for _ in range(60):
                    if not alive(expected['pid']) and (not server or not alive(server['pid'])):break
                    time.sleep(1)
                if alive(expected['pid']) or (server and alive(server['pid'])):
                    raise RuntimeError('Old processes did not terminate; no second GPU owner launched')
                receipt={'UTC':utc(),'old_owner_identity':expected,'old_server_identity':server,
                    'completed_or_error_evidence':str(target),'partial_training_killed':False,
                    'new_owner_pid':os.getpid(),'family_boundary':family,
                    'next':'Dita1024 before another large S fit, reusing any completed stages'}
                atomic_json(OUT/'HANDOFF_RECEIPT.json',receipt)
                for status in (R85/'owner_status.json',R85/'dita/owner_status.json'):
                    old=read(status);old.update(status='superseded_at_completed_cohort_boundary',UTC=utc(),
                        successor_status=str(OUT/'LIVE.json'),handoff_receipt=str(OUT/'HANDOFF_RECEIPT.json'))
                    atomic_json(status,old)
                return
            if time.monotonic()>deadline:raise TimeoutError('Finite handoff budget; did not cancel existing study')
            time.sleep(20)
        atomic_json(OUT/'HANDOFF_RECEIPT.json',{'UTC':utc(),'old_owner_identity':expected,
            'old_owner_already_exited':True,'next':'adopt only integrity-qualified completed checkpoints'})
    finally:
        if paused and alive(expected['pid']):checked_signal(expected,signal.SIGCONT)


class Observed:
    def __init__(self):
        super().__init__();self.bound.update(bindings())
    def phase(self,label):
        self.stage=label
        with (OUT/'EVENTS.jsonl').open('a') as f:
            f.write(json.dumps({'UTC':utc(),'family':self.family,'stage':label})+'\n')
        self.status()
    def status(self,status='running',**extra):
        state={'status':status,'family':self.family,'stage':self.stage,'UTC':utc(),'pid':os.getpid(),
            'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,**extra}
        atomic_json(OUT/self.family/'owner_status.json',state)
        atomic_json(OUT/'LIVE.json',state)
    def run(self,python,module,args,label,*,hours=12):
        if module=='VLA.autonomy_r83.evaluate':module='VLA.continuous_r86.evaluate'
        elif module=='VLA.dita_transfer_r84.train':module='VLA.recovery_r85.dita_train'
        verify(self.bound);self.phase(label)
        logpath=OUT/self.family/(label+'.log')
        with logpath.open('x') as log:
            self.child=subprocess.Popen([str(python),'-u','-m',module,*map(str,args)],cwd=ROOT,
                env=prior.flow.ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+hours*3600
            while self.child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Finite phase budget '+label)
                detail=None
                if '--output' in args:
                    folder=Path(args[args.index('--output')+1])
                    for name in ('status.json','progress.json'):
                        p=folder/name
                        if p.exists():
                            detail=read(p)
                            # Keep the live index small; all raw outcomes stay on disk.
                            if 'rows' in detail:detail={k:v for k,v in detail.items() if k!='rows'}
                self.status(log=str(logpath),actual_child_progress=detail)
                time.sleep(10)
            code=self.child.returncode;self.child=None
        if code:raise RuntimeError(f'{label} exit{code}; inspect retained {logpath}; NOT model SR0')
    def evaluate(self,method,label,role,seeds):
        progress=super().evaluate(method,label,role,seeds)
        path=OUT/self.family/label/'progress.json'
        audit=cohort_audit(path,require_complete=True)
        atomic_json(path.parent/'AUDIT.json',audit)
        return progress


class Flow(Observed,prior.Flow):
    family='vlanext_s'
    def method(self,checkpoint,name,*,frozen=False):
        path=super().method(checkpoint,name,frozen=frozen)
        method=read(path)
        method.update(variant='native_full_adam8_privileged74_map3_OFFICIAL_GRIPPER_SIGN',
            decoder='official VLANeXt signed gripper: >0 -> +1, else -1; common plant unchanged',
            previous_continuous_deadband_variant_results_NOT_poolable=True)
        atomic_json(path,method)
        return path
    def evaluate(self,method,label,role,seeds):
        # Same evaluator/plant/metric/seeds, but explicitly version the fixed
        # output decoder. Do not relabel an old continuous-gripper DEV outcome.
        self.resource();self.phase(label+'_server')
        folder=OUT/self.family;cohort=folder/(label+'_COHORT.json')
        atomic_json(cohort,{'role':role,'seeds':seeds})
        self.server_log=(folder/(label+'_server.log')).open('x')
        self.server=subprocess.Popen([str(prior.flow.PY),'-u','-m','VLA.continuous_r86.vlanext_server',
            '--method',str(method),'--port',str(prior.flow.PORT)],cwd=ROOT,env=prior.flow.ENV,
            stdin=subprocess.DEVNULL,stdout=self.server_log,stderr=subprocess.STDOUT)
        session=requests.Session();session.trust_env=False
        try:
            for _ in range(300):
                if self.server.poll() is not None:raise RuntimeError('Corrected native server exited; retained log')
                try:
                    response=session.get(f'http://127.0.0.1:{prior.flow.PORT}/health',timeout=2)
                    response.raise_for_status()
                    if response.json()['method_sha256']!=sha256_file(method):raise ValueError('Another method occupies port')
                    break
                except requests.RequestException:self.status();time.sleep(2)
            else:raise TimeoutError('Native corrected server readiness')
            self.run(prior.flow.SIM,'VLA.autonomy_r83.evaluate',['--method',method,'--cohort',cohort,
                '--output',folder/label,'--port',prior.flow.PORT],label,
                hours=36 if role=='fresh_secondary_final800' else 4)
            path=folder/label/'progress.json';audit=cohort_audit(path,require_complete=True)
            atomic_json(path.parent/'AUDIT.json',audit)
            return read(path)
        finally:session.close();self.stop()

class Dita(Observed,prior.Dita):family='dita'


def qualified(cp):
    m=read(cp/'TRAINING.json');verify(m['code'])
    if not m['completed']:raise ValueError('Only completed actual optimizer stages may resume')
    for file,key in (('weights.safetensors','weights_sha256'),('optimizer.pt','optimizer_sha256')):
        if sha256_file(cp/file)!=m[key]:raise ValueError('Resume artifact integrity mismatch: '+str(cp/file))
    return m


def fit_and_dev(owner,step,parent):
    family=owner.family;cp=OUT/family/f'fit{step}'
    parent_meta=qualified(parent)
    owner.resource()
    args=['fit','--output',cp,'--updates',step,'--accumulation',16 if family=='vlanext_s' else 32,'--resume',parent]
    module='VLA.replacement_r84.train' if family=='vlanext_s' else 'VLA.dita_transfer_r84.train'
    py=prior.flow.PY if family=='vlanext_s' else prior.dita.PY
    if family=='vlanext_s':args+=['--optimizer','adamw8']
    if int(parent_meta['updates'])==step:
        cp=parent
        reload_file=cp/('FRESH_RELOAD.json' if family=='vlanext_s' else 'FRESH_RELOAD_cuda.json')
        if not reload_file.exists() or not read(reload_file)['passed']:
            owner.run(py,module,['reload','--checkpoint',cp],f'fit{step}_reload',hours=1)
        atomic_json(OUT/family/f'REUSED_FIT{step}.json',{'actual_checkpoint':str(cp),
            'no_optimizer_updates_repeated':True,'prior_incomplete_evaluation_NOT_pooled':True})
    else:
        owner.run(py,module,args,f'fit{step}',hours=24)
        owner.run(py,module,['reload','--checkpoint',cp],f'fit{step}_reload',hours=1)
    owner.run(prior.flow.PY,'VLA.continuous_r86.probe',['--checkpoint',cp,
        '--output',cp/'TRAIN_PREDICTION_AUDIT.json'],f'fit{step}_train_prediction_audit',hours=1)
    method=owner.method(cp,f'method{step}')
    progress=owner.evaluate(method,f'dev40_fit{step}','development_selection',list(range(20283001,20283011)))
    return cp,progress


def ready(progress):
    if (progress['status']!='completed_actual_development' or progress['completed_trials']!=40
            or len({(r['seed'],r['task']) for r in progress['rows']})!=40
            or any(v['n']!=10 for v in progress['per_action'].values())):
        raise ValueError('Selection requires completed40 with10 each action')
    return sum(r['success'] for r in progress['rows'])>=24 and min(v['successes'] for v in progress['per_action'].values())>=4


def confirm_and_final(owner,cp):
    method=owner.method(cp,'selected_method')
    conf=owner.evaluate(method,'confirmation40','development_confirmation',list(range(20283101,20283111)))
    if not ready(conf):
        owner.stage='confirmation_failed_NO_final800';owner.status('not_promoted')
        return
    method=owner.method(cp,'FROZEN_FINAL_METHOD',frozen=True)
    atomic_json(OUT/owner.family/'FROZEN_FINAL.json',{'checkpoint':str(cp),'method':str(method),
        'method_sha256':sha256_file(method),'code':owner.bound,'no_final_tuning':True})
    final=owner.evaluate(method,'final800','fresh_secondary_final800',list(range(20284001,20284201)))
    if final['completed_trials']!=800 or any(v['n']!=200 for v in final['per_action'].values()):
        raise ValueError('Final800 coverage mismatch')
    atomic_json(OUT/owner.family/'FINAL_AUDIT.json',{'passed':True,'actual_trials':800,
        'per_action':final['per_action'],'actual_cohorts_of800':1,'not_five_measured_seeds':True})
    owner.stage='frozen_final800_completed';owner.status('completed',per_action=final['per_action'])


def main(a):
    OUT.mkdir(parents=True,exist_ok=True)
    for family in ('vlanext_s','dita'):(OUT/family).mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'OWNER_STARTED.json').exists():raise FileExistsError('No duplicate controller')
        verify(read(OUT/'QUALIFICATION.json')['code'])
        if not read(OUT/'QUALIFICATION.json')['passed']:raise ValueError('Offline qualification missing')
        atomic_json(OUT/'OWNER_STARTED.json',{'pid':os.getpid(),'UTC':utc(),'code':bindings()})
        handoff(read(a.adopt_identity))
        prior.flow.OUT=OUT;prior.dita.OUT=OUT/'dita';prior.flow.PORT=5988;prior.dita.PORT=5989
        atomic_json(OUT/'dita/TEXT_FOUNDATION.json',read(R85/'dita/TEXT_FOUNDATION.json'))
        owners={'vlanext_s':Flow(),'dita':Dita()}
        parents={'vlanext_s':R84/'vlanext_s/fit512','dita':R85/'dita/gpu_gate4'}
        # The existing S2048 fit and actual DEV40 are reused, never rerun or
        # silently pooled. If it qualified, confirm it before further tuning.
        done=set();selections={family:[] for family in owners}
        adopted={}
        for family,step,folder in (('vlanext_s',2048,R85/'vlanext_s'),('dita',1024,R85/'dita')):
            cp=folder/f'fit{step}';dev=folder/f'dev40_fit{step}/progress.json'
            try:
                if (cp/'TRAINING.json').exists():
                    qualified(cp);parents[family]=cp
                    if dev.exists() and read(dev)['status']=='completed_actual_development':
                        inherited=cohort_audit(dev,require_complete=True)
                        atomic_json(OUT/family/f'ADOPTED_DEV{step}.json',inherited)
                        # S's decoder now differs from R85: reuse its trained
                        # weights, NEVER reuse its old-gripper performance gate.
                        if family=='dita':adopted[family]=(cp,read(dev))
                    else:
                        atomic_json(OUT/family/'PRIOR_INCOMPLETE_NOT_POOLED.json',{
                            'checkpoint_reused':str(cp),'new_DEV40_will_be_separate':True})
            except BaseException as e:
                done.add(family)
                atomic_json(OUT/family/'ADOPTION_ERROR.json',{'error':str(e),'NOT_model_SR':True})
                owners[family].stage='adoption_integrity_failed';owners[family].status('quarantined_other_model_continues')
        schedule=[('dita',1024),('vlanext_s',2048),('dita',4096),('vlanext_s',8192),('dita',10000)]
        for family,step in schedule:
            if family in done:continue
            owner=owners[family]
            try:
                if family in adopted and int(read(adopted[family][0]/'TRAINING.json')['updates'])==step:
                    cp,progress=adopted[family]
                    owner.run(prior.flow.PY,'VLA.continuous_r86.probe',['--checkpoint',cp,
                        '--output',OUT/family/'ADOPTED_TRAIN_PREDICTIONS.json'],f'adopted{step}_train_prediction_audit',hours=1)
                else:cp,progress=fit_and_dev(owner,step,parents[family])
                selections[family].append({'updates':step,'checkpoint':str(cp),'per_action':progress['per_action']})
                atomic_json(OUT/family/'DEV_SELECTION.json',{'actual_options':selections[family],
                    'rule':'24/40 total AND4/10/action, then same rule on independent CONF40',
                    'no_confirmation_or_final_tuning':True})
                parents[family]=cp
                if ready(progress):
                    done.add(family);confirm_and_final(owner,cp)
                else:
                    owner.stage='awaiting_next_declared_training_stage';owner.status('queued_more_adaptation',next_parent=str(cp))
            except BaseException as e:
                done.add(family)
                atomic_json(OUT/family/'INTEGRATION_ERROR.json',{'UTC':utc(),'stage':owner.stage,
                    'error':str(e),'traceback':traceback.format_exc(),'NOT_model_SR_or_intrinsic_weakness':True})
                owner.status('quarantined_integration_error_next_qualified_model_continues',error=str(e))
            finally:owner.stop()
        for family,owner in owners.items():
            if family not in done:
                owner.stage='finite_budget_complete_not_promoted';owner.status('needs_further_audit_NOT_intrinsic_model_limit')
        atomic_json(OUT/'WORKFLOW_COMPLETE.json',{'UTC':utc(),'actual_selections':selections,
            'guaranteed_SR':False,'all_outcomes_and_failures_retained':True})
        atomic_json(OUT/'LIVE.json',{'UTC':utc(),'status':'declared_workflow_finished_read_per_model_status',
            'families':{family:read(OUT/family/'owner_status.json') for family in owners}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true')
    p.add_argument('--adopt-identity',type=Path,default=OUT/'ADOPT_IDENTITY.json');a=p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(prior.flow.PY),'-u','-m','VLA.continuous_r86.owner',
                '--adopt-identity',str(a.adopt_identity)],cwd=ROOT,env=prior.flow.ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'actual_progress':str(OUT/'LIVE.json')})
    else:
        try:main(a)
        except BaseException as e:
            atomic_json(OUT/'WORKFLOW_ERROR.json',{'UTC':utc(),'error':str(e),'traceback':traceback.format_exc(),
                'NOT_model_SR':True})
            atomic_json(OUT/'LIVE.json',{'UTC':utc(),'status':'orchestration_error_no_model_SR_claim','error':str(e)})
            raise
