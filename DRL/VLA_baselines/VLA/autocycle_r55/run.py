"""Automatic development triage -> qualified continuation -> confirmation.

At most two evidenced continuation rounds per weak family; never an infinite
best-test-score search. All four selected methods freeze BEFORE new final120.
Prior active source/checkpoints/processes remain untouched.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT,BASE,DATA,PRIOR,R54,R54_BASE,SIM,PI,MODELS,read,verify,hashes,optimizer_updates
from .decision import qualified_progress,score,diagnose,stable,candidate_key,continue_after_round


def alive(pid):
    if not pid:return False
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def initial_candidates(model):
    """Read frozen DEVELOPMENT choices only, never prior final progress/results."""
    if model in ('xvla','vla_adapter'):
        folder=PRIOR/model;chosen=read(folder/'frozen_final_method.json')
        base=read(folder/'baseline_DEVELOPMENT_ONLY_reference.json')
        candidates=chosen['map_candidates']
        if chosen['visual_mode']=='map3':
            item=next(c for c in candidates if c['checkpoint']==chosen['checkpoint']);progress=item['progress']
        else:progress=base['progress']
        baseline={'checkpoint':chosen['checkpoint'],'schedule':chosen['schedule'],'mode':chosen['visual_mode'],
            'development_progress':progress,'preserved_baseline':True}
        parent=max(candidates,key=lambda c:score(c['progress']))['checkpoint'] if candidates else None
    else:
        chosen=read(R54/(model+'_frozen_final_method.json'))
        baseline={**chosen,'mode':'baseline' if chosen['baseline'] else 'map3','preserved_baseline':True}
        candidates=[c for c in chosen['candidates'] if not c['baseline']]
        parent=max(candidates,key=lambda c:score(c['development_progress']))['checkpoint'] if candidates else None
    p=qualified_progress(baseline['development_progress'])
    if p['checkpoint']!=baseline['checkpoint']:raise ValueError('Selected DEVELOPMENT receipt belongs to different weights')
    return baseline,None if parent is None else Path(parent)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    child=server=server_log=None;stage='preflight_and_predeclaration';selected={};errors={};history={}
    frozen=hashes()
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
            'active_child_pid':None if child is None else child.pid,'active_server_pid':None if server is None else server.pid,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'selected_methods':selected,'errors':errors,**kw})
        try:
            from .report import report
            report(a.output)
        except Exception as error:
            # A report problem must not orphan a healthy native optimizer.
            atomic_json(a.output/'report_error.json',{'error':f'{type(error).__name__}: {error}',
                'active_model_job_not_interrupted':True})
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_MEM_FRACTION':'.82'}
    def gpu_slot():
        from VLA.new_vlas.train import wait_gpu
        folder=a.output/'GPU_admission';folder.mkdir(exist_ok=True);wait_gpu(folder)
    def execute(python,module,options,label,gpu=False):
        nonlocal child
        verify(frozen)
        if gpu:gpu_slot()
        with (a.output/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(python),'-u','-m',module,*map(str,options)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);status()
            while child.poll() is None:time.sleep(10);status()
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label}: integration/gate exit {code}; NOT zero model SR')
    def stop():
        nonlocal server,server_log
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        server=None
        if server_log is not None:server_log.close();server_log=None
    def python(model):
        return ROOT/'VLA/.venvs'/('xvla' if model=='xvla' else 'openvla')/'bin/python' if model in ('xvla','vla_adapter') else SIM if model=='smolvla' else PI
    def serve(model,method,port,label):
        nonlocal server,server_log
        gpu_slot();cp=Path(method['checkpoint']);py=python(model)
        if model in ('xvla','vla_adapter'):
            module='VLA.privileged_visual_r53.server' if method['mode']=='map3' else 'VLA.stable_native_r44.server'
            opts=['--checkpoint',cp,'--port',port]
        else:
            module='VLA.family_transfer_r54.server';opts=['--model',model,'--checkpoint',cp,'--port',port,
                '--output',a.output/(label+'_runtime')]
            if method['mode']=='baseline':opts+=['--baseline']
        server_log=(a.output/(label+'_server.log')).open('x')
        server=subprocess.Popen([str(py),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=server_log,stderr=subprocess.STDOUT);status()
        until=time.monotonic()+1800
        while True:
            if server.poll() is not None:raise RuntimeError('Own native server failed before health')
            try:
                reply=requests.get(f'http://127.0.0.1:{port}/health',timeout=5);reply.raise_for_status()
                if reply.json()['checkpoint']!=str(cp.resolve()):raise ValueError('Actual server owns different checkpoint')
                return
            except requests.RequestException:
                if time.monotonic()>until:raise TimeoutError('Own native server not ready')
                time.sleep(5);status()
    def evaluate(model,method,port,cohort,folder,label):
        opts=['--model',model,'--port',port,'--mode',method['mode'],'--cohort',cohort,'--output',folder]
        opts+=['--schedule',method['schedule']] if model in ('xvla','vla_adapter') else ['--prefix',method['prefix']]
        execute(SIM,'VLA.autocycle_r55.evaluate',opts,label)
        execute(SIM,'VLA.autocycle_r55.audit_video',['--model',model,'--progress',folder/'progress.json',
            '--output',folder/'video_audit'],label+'_decode_video')
    def confirm(model,method,port,label):
        serve(model,method,port,label)
        folder=a.output/(label+'_confirmation40')
        evaluate(model,method,port,a.output/'confirmation40.json',folder,label+'_confirmation40')
        stop();return {**method,'confirmation_progress':str(folder/'progress.json')}
    def assess(model,cp,port,label):
        method={'checkpoint':str(cp),'mode':'map3','preserved_baseline':False}
        serve(model,method,port,label);paths=[]
        for prefix in ((4,6,30 if model=='xvla' else 8) if model in ('xvla','vla_adapter') else (2,4,5)):
            folder=a.output/f'{label}_prefix{prefix}_pilot8'
            opts=['--model',model,'--port',port,'--prefix',prefix,'--mode','map3','--output',folder]
            if model not in ('xvla','vla_adapter'):opts+=['--cohort',a.output/'pilot8.json']
            execute(SIM,'VLA.autocycle_r55.evaluate',opts,f'{label}_prefix{prefix}')
            paths.append(folder/'progress.json')
        if model in ('xvla','vla_adapter'):
            from VLA.privileged_visual_r53.run import global_schedule
            schedule=a.output/(label+'_global_schedule.json');atomic_json(schedule,global_schedule(paths,model))
            method['schedule']=str(schedule)
        else:
            from VLA.family_transfer_r54.run import choose_prefix
            method['prefix']=choose_prefix(paths)
        folder=a.output/(label+'_development40')
        evaluate(model,method,port,a.output/'development40.json',folder,label+'_development40')
        method['development_progress']=str(folder/'progress.json')
        folder=a.output/(label+'_confirmation40')
        evaluate(model,method,port,a.output/'confirmation40.json',folder,label+'_confirmation40')
        method['confirmation_progress']=str(folder/'progress.json');stop();return method
    def fit_round(model,parent,profile,number):
        folder=a.output/f'{model}_round{number}';folder.mkdir();py=python(model)
        if model in ('xvla','vla_adapter'):
            origin=read(parent/'config.json')['actual_optimizer_updates']
            proof=ROOT/'VLA/audits/privileged_visual_r53_20261016'/('xvla_input_v3/audit.json' if model=='xvla' else 'adapter_input_v3/audit.json')
            common=['--family',model,'--data',DATA,'--sampling-profile',profile,'--input-proof',proof,'--total-updates',origin+1500]
            gate=folder/'gate2'
            execute(py,'VLA.autocycle_r55.fit_native',[*common,'--mode','gate','--resume',parent,
                '--updates',origin+2,'--output',gate],f'{model}_r{number}_gate2',True)
            cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);reload=folder/'reload2'
            execute(py,'VLA.privileged_visual_r53.reload_gate',['--checkpoint',cp,'--output',reload],f'{model}_r{number}_reload2',True)
            fit=folder/'fit1500'
            execute(py,'VLA.autocycle_r55.fit_native',[*common,'--mode','long','--resume',cp,'--updates',origin+1500,
                '--reload-proof',reload/'audit.json','--output',fit],f'{model}_r{number}_fit1500',True)
            cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
            execute(py,'VLA.privileged_visual_r53.reload_gate',['--checkpoint',cp,'--completed-fit',
                '--output',folder/'reload1500'],f'{model}_r{number}_reload1500',True)
        elif model=='smolvla':
            origin=read(parent/'r54_contract.json')['added_optimizer_updates'];gate=folder/'gate2'
            common=['--sampling-profile',profile]
            execute(py,'VLA.autocycle_r55.fit_smol',[*common,'--mode','gate','--resume',parent,'--updates',origin+2,
                '--output',gate],f'{model}_r{number}_gate2',True)
            cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);proof=folder/'reload2.json'
            execute(py,'VLA.family_transfer_r54.reload_smol',['--checkpoint',cp,'--output',proof],f'{model}_r{number}_reload2',True)
            fit=folder/'fit2000'
            execute(py,'VLA.autocycle_r55.fit_smol',[*common,'--mode','long','--resume',cp,'--updates',origin+2000,
                '--reload-proof',proof,'--output',fit],f'{model}_r{number}_fit2000',True)
            cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
            execute(py,'VLA.family_transfer_r54.reload_smol',['--checkpoint',cp,'--output',folder/'reload2000.json'],f'{model}_r{number}_reload2000',True)
        else:
            common=['--sampling-profile',profile];gate=folder/'gate2'
            execute(py,'VLA.autocycle_r55.fit_pi0',[*common,'--parent',parent,'--updates',2,'--output',gate],f'{model}_r{number}_gate2',True)
            cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);proof=folder/'reload2.json'
            execute(py,'VLA.family_transfer_r54.reload_pi0',['--checkpoint',cp,'--output',proof],f'{model}_r{number}_reload2',True)
            fit=folder/'fit1998'
            execute(py,'VLA.autocycle_r55.fit_pi0',[*common,'--parent',cp,'--qualification',proof,'--updates',1998,
                '--output',fit],f'{model}_r{number}_fit1998',True)
            cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
            execute(py,'VLA.family_transfer_r54.reload_pi0',['--checkpoint',cp,'--output',folder/'reload1998.json'],f'{model}_r{number}_reload1998',True)
        return cp
    try:
        preflight=read(BASE/'preflight/audit.json')
        if not preflight.get('passed') or preflight['data_audit_sha256']!=sha256_file(DATA/'audit.json'):
            raise ValueError('Own unchanged-observation and actual sampler preflight required')
        verify(preflight['code'])
        from VLA.run_feedback_refinement_r52 import qualify_final_seeds
        new_final=list(range(20269701,20269731));confirmation=list(range(20268101,20268111))
        exclusions=qualify_final_seeds(new_final);confirm_exclusions=qualify_final_seeds(confirmation)
        old=set()
        for p in (PRIOR/'final_geometry_exclusion.json',
                  ROOT/'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4/final_cohort_exclusion_audit.json'):
            old.update(c['physical_scene_sha256'] for c in read(p)['cases'])
        new={c['physical_scene_sha256'] for c in exclusions['cases']};conf={c['physical_scene_sha256'] for c in confirm_exclusions['cases']}
        if new&old or conf&old or new&conf:raise ValueError('New final/confirmation geometry overlaps prior final or each other')
        atomic_json(a.output/'geometry_exclusion.json',{'final':exclusions,'confirmation':confirm_exclusions,'old_final_overlap':0})
        atomic_json(a.output/'pilot8.json',{'role':'development_prefix_pilot','scope':'R55_DEVELOPMENT_prefix_selection_NOT_final','seeds':[20268001,20268002]})
        atomic_json(a.output/'development40.json',{'role':'development_selection','scope':'R55_DEVELOPMENT40_NOT_final','seeds':list(range(20268011,20268021))})
        atomic_json(a.output/'confirmation40.json',{'role':'development_confirmation','scope':'R55_DEVELOPMENT_confirmation40_USED_for_selection_NOT_final','seeds':confirmation})
        atomic_json(a.output/'final120.json',{'role':'untouched_final','scope':'R55_UNTOUCHED_common_final120_NO_subsequent_tuning','seeds':new_final})
        atomic_json(a.output/'plan.json',{'models':MODELS,'code':frozen,'prior_owner':str(R54),
            'never_preempt_prior_jobs':True,'prior_final_outcomes_or_scores_consumed_for_selection':False,
            'max_continuation_rounds_each_weak_family':2,'max_actual_added_updates_native_each':3000,
            'max_actual_added_updates_Smol_pi0_each':4000,'target_STABLE_DEVELOPMENT':'atleast24/40 in BOTH cohorts with<=4 aborts each AND each task>=3/10 in BOTH',
            'round2_requires_actual_pooled80_gain_atleast2_and_not_already_stable':True,
            'sampler_profiles':'predeclared_acquisition_or_transport_from_development_trace_symptoms_only',
            'new_fit_start_window_after_prior12h':True,'all_methods_frozen_BEFORE_any_new_final':True,
            'max_prior_wait_hours':36,'guaranteed_high_SR_or_global_maximum':False})
        stage='waiting_R54_owner_NO_GPU_NO_preemption_NO_final_outcome_reads';status()
        until=time.monotonic()+36*3600
        while True:
            prior=read(R54/'status.json')
            if prior['status']=='completed' and not any(prior.get(k) for k in ('active_child_pid','active_server_pid')):break
            if prior['status'].startswith('failed') and not any(alive(prior.get(k)) for k in ('pid','active_child_pid','active_server_pid')):break
            if time.monotonic()>until:raise TimeoutError('Finite predecessor wait36h exceeded')
            time.sleep(30);status()
        fit_deadline=time.monotonic()+12*3600
        for i,model in enumerate(MODELS):
            port=5885+i;best=None
            try:
                verify(frozen);stage=model+'_read_frozen_DEVELOPMENT_only';status()
                baseline,parent=initial_candidates(model)
                stage=model+'_independent_DEVELOPMENT_confirmation40';status()
                best=confirm(model,baseline,port,model+'_preserved');history[model]=[best]
                for number in (1,2):
                    if stable(best) or parent is None or time.monotonic()>fit_deadline:break
                    diagnostic=diagnose(best['development_progress']);atomic_json(a.output/f'{model}_r{number}_diagnosis.json',diagnostic)
                    profile=diagnostic['selected_predeclared_TRAIN_sampler_profile']
                    stage=model+f'_round{number}_qualified_native_fit_{profile}';status()
                    cp=fit_round(model,parent,profile,number)
                    stage=model+f'_round{number}_actual_development80';status()
                    candidate=assess(model,cp,port,f'{model}_new_r{number}');history[model].append(candidate)
                    previous=best
                    if candidate_key(candidate)>candidate_key(best):best=candidate
                    atomic_json(a.output/(model+'_DEVELOPMENT_history.json'),history[model])
                    if not continue_after_round(previous,candidate,number):break
                    parent=cp
                selected[model]={**best,'stable_development_target_met':stable(best),
                    'selection_uses_pooled80_development_NOT_final':True}
                atomic_json(a.output/(model+'_frozen_method.json'),selected[model]);status()
            except Exception as e:
                if child is not None and child.poll() is None:raise
                stop();errors[model]=f'{type(e).__name__}: {e}'
                # A failed extra fit cannot erase a successfully confirmed,
                # already-qualified native baseline. Preserve it explicitly.
                if best is not None:
                    selected[model]={**best,'stable_development_target_met':stable(best),
                        'extra_fit_failed_preserved_confirmed_baseline':True}
                    atomic_json(a.output/(model+'_frozen_method.json'),selected[model])
                status()
        atomic_json(a.output/'ALL_METHODS_FROZEN_BEFORE_FINAL.json',{'methods':selected,'errors':errors,'subsequent_training_allowed':False})
        stage='all_methods_frozen_begin_common_final120_NO_further_training';status()
        for i,model in enumerate(MODELS):
            if model not in selected:continue
            try:
                method=selected[model];port=5885+i;stage=model+'_untouched_FINAL120';status()
                serve(model,method,port,model+'_final')
                evaluate(model,method,port,a.output/'final120.json',a.output/(model+'_final120'),model+'_final120')
                stop();status()
            except Exception as e:
                if child is not None and child.poll() is None:raise
                stop();errors[model+'_final']=f'{type(e).__name__}: {e}';status()
        from .report import report
        report(a.output)
        stage='finite_autocycle_finished_REVIEW_actual_success_NOT_guaranteed_maximum';status('completed')
    except Exception as e:status('failed_preserve_all_evidence_NO_fake_success',error=f'{type(e).__name__}: {e}');raise
    finally:stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_suffix('.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','VLA.autocycle_r55.run','--output',str(a.output.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'output':str(a.output.resolve())})
    else:main(a)
