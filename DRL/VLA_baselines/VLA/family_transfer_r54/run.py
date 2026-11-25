"""Finite Smol/pi0 transfer, gated fits, development selection, common final120.

Waits for R53, never preempts actual jobs, never consumes R52/R53 final scores.
Four-family paper cohort is the already-predeclared R53 120 (30/action), not
four different favorable cohorts. RDT remains a fifth candidate, not fake ready.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT,DATA,PRIOR,BASE,SIM,PI,SMOL,SCHEMA,hashes,read,verify


def alive(pid):
    if not pid:return False
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def score(path):
    p=read(path)
    if p['completed_trials']!=p['planned_trials']:raise ValueError('Only complete cohorts select')
    return p['successes'],-sum(bool(r.get('safety_violation')) for r in p['rows'])


def choose_prefix(paths):
    options=[];identity=None;cohort=None
    for p in paths:
        x=read(p)
        if x['completed_trials']!=8 or x['planned_trials']!=8:raise ValueError('Complete matched8 required')
        if identity is not None and identity!=x['checkpoint']:raise ValueError('Do not pool model checkpoints')
        if cohort is not None and cohort!=x['cohort']:raise ValueError('Do not pool different seed cohorts')
        identity,cohort=x['checkpoint'],x['cohort'];options.append((score(p),x['prefix']))
    return max(options,key=lambda v:(*v[0],v[1]==5,v[1]))[1]


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    child=server=log=None;stage='predeclaring_common_four_family_cohort';results=[]
    frozen=hashes()
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
            'active_child_pid':None if child is None else child.pid,'active_server_pid':None if server is None else server.pid,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'results':results,**kw})
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XLA_PYTHON_CLIENT_MEM_FRACTION':'.82',
        'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    def execute(python,module,options,label):
        nonlocal child
        verify(frozen)
        if module.rsplit('.',1)[-1] in ('fit_smol','fit_pi0','reload_smol','reload_pi0'):
            from VLA.new_vlas.train import wait_gpu
            admission=a.output/'GPU_admission';admission.mkdir(exist_ok=True)
            wait_gpu(admission)
        with (a.output/(label+'.log')).open('x') as f:
            child=subprocess.Popen([str(python),'-u','-m',module,*map(str,options)],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status()
            while child.poll() is None:time.sleep(10);status()
            code=child.returncode
        child=None
        if code:raise RuntimeError(label+f' integration/gate exit {code}; NOT zero model SR')
    def stop():
        nonlocal server,log
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        server=None
        if log is not None:log.close();log=None
    def serve(model,cp,port,label,baseline=False):
        nonlocal server,log
        from VLA.new_vlas.train import wait_gpu
        admission=a.output/'GPU_admission';admission.mkdir(exist_ok=True)
        wait_gpu(admission)
        py=SIM if model=='smolvla' else PI
        log=(a.output/(label+'_server.log')).open('x')
        cmd=[str(py),'-u','-m','VLA.family_transfer_r54.server','--model',model,'--checkpoint',str(cp),
             '--port',str(port),'--output',str(a.output/(label+'_runtime'))]
        if baseline:cmd.append('--baseline')
        server=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+1800
        while True:
            if server.poll() is not None:raise RuntimeError('Own native server exited; inspect retained log')
            try:
                reply=requests.get(f'http://127.0.0.1:{port}/health',timeout=5);reply.raise_for_status()
                if reply.json()['checkpoint']!=str(cp.resolve()):raise ValueError('Wrong native checkpoint server')
                return
            except requests.RequestException:
                if time.monotonic()>deadline:raise TimeoutError('Native model server not ready')
                time.sleep(5);status()
    def assess(model,cp,port,label,baseline=False):
        nonlocal stage
        serve(model,cp,port,label,baseline)
        paths=[]
        for prefix in (2,4,5):
            folder=a.output/f'{label}_prefix{prefix}_pilot8'
            execute(SIM,'VLA.family_transfer_r54.evaluate',['--model',model,'--port',port,'--prefix',prefix,
                '--cohort',a.output/'pilot8.json','--output',folder],f'{label}_prefix{prefix}')
            paths.append(folder/'progress.json')
        prefix=choose_prefix(paths)
        folder=a.output/f'{label}_development40'
        stage=label+'_complete_DEVELOPMENT40';status()
        execute(SIM,'VLA.family_transfer_r54.evaluate',['--model',model,'--port',port,'--prefix',prefix,
            '--cohort',a.output/'development40.json','--output',folder],label+'_development40')
        stop()
        return {'checkpoint':str(cp),'prefix':prefix,'baseline':baseline,'score':score(folder/'progress.json'),
                'development_progress':str(folder/'progress.json')}
    try:
        for model in ('smolvla','pi0'):
            q=read(BASE/(model+'_input_final/audit.json'))
            if not q['passed'] or q['model']!=model or q['schema']!=SCHEMA:raise ValueError('Own final actual native CPU proof missing')
            verify(q['code'])
        shared=read(PRIOR/'final_cohort.json')
        if shared['seeds']!=list(range(20269801,20269831)):raise ValueError('Common untouched four-family final changed')
        atomic_json(a.output/'pilot8.json',{'scope':'R54_DEVELOPMENT_pilot8_NOT_independent_SR','seeds':[20268001,20268002]})
        atomic_json(a.output/'development40.json',{'scope':'R54_DEVELOPMENT40_for_selection_NOT_final','seeds':list(range(20268011,20268021))})
        atomic_json(a.output/'final120.json',{**shared,'scope':'R54_FINAL120_SAME_predeclared_R53_cases_NO_tuning'})
        atomic_json(a.output/'plan.json',{'models':['smolvla','pi0'],'common_comparison_models':['xvla','vla_adapter','smolvla','pi0'],
            'fifth_candidate_not_admitted':'rdt','schema':SCHEMA,'qualified_TRAIN_data':str(DATA),
            'data_audit_sha256':sha256_file(DATA/'audit.json'),'code':frozen,'prior_owner':str(PRIOR),
            'never_preempt_actual_jobs':True,'never_read_R52_R53_final_outcomes_for_selection':True,
            'common_final120_definition_sha256':sha256_file(PRIOR/'final_cohort.json'),
            'Smol_budget':'2 real gate + fresh8 reload then2500 total added native updates; compare preserved baseline',
            'pi0_budget':'2 real own gate; independent1500 EEF-LoRA fit from official foundation; ifdev40<24 one3000 fit, not optimizer resume',
            'prefix_selection':'ONE_global_2_4_5_from_complete_same8_then_full40',
            'selection':'best_complete40_success_then_fewer_aborts_then_preserve_baseline_on_ties',
            'same_physics_robot_gripper_objects_cameras_success_horizon':True,'new_action_repair_Oracle_or_IK_policy':False,
            'GT3_CPU_ready_NOT_full_GPU_qualified':True,'guaranteed_SR':None})
        stage='waiting_R53_finite_owner_NO_preemption_NO_final_outcome_reads';status()
        until=time.monotonic()+36*3600
        while True:
            previous=read(PRIOR/'status.json')
            if previous['status']=='completed' and not previous['active_child_pid'] and not previous['active_server_pid']:break
            if previous['status'].startswith('failed') and not any(alive(previous.get(k)) for k in ('pid','active_child_pid','active_server_pid')):
                atomic_json(a.output/'prior_failed_inactive_receipt.json',{'status':previous['status'],'pid':previous['pid'],
                    'all_prior_processes_exited':True,'final_outcomes_read':False});break
            if time.monotonic()>until:raise TimeoutError('Finite R53 wait budget exceeded36h')
            time.sleep(20);status()
        for model,port in [('smolvla',5875),('pi0',5876)]:
            try:
                candidates=[];stage=model+'_own_GPU_save_reload_gate2';status()
                if model=='smolvla':
                    candidates.append(assess(model,SMOL,port,'smol_preserved_baseline',True))
                    gate=a.output/'smol_gate2'
                    execute(SIM,'VLA.family_transfer_r54.fit_smol',['--mode','gate','--updates',2,'--output',gate],model+'_gate2')
                    cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);proof=a.output/'smol_reload2.json'
                    execute(SIM,'VLA.family_transfer_r54.reload_smol',['--checkpoint',cp,'--output',proof],model+'_reload2')
                    fit=a.output/'smol_fit2500';stage=model+'_actual2500_added_updates';status()
                    execute(SIM,'VLA.family_transfer_r54.fit_smol',['--mode','long','--updates',2500,'--resume',cp,
                        '--reload-proof',proof,'--output',fit],model+'_fit2500')
                    cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
                    execute(SIM,'VLA.family_transfer_r54.reload_smol',['--checkpoint',cp,'--output',a.output/'smol_reload2500.json'],model+'_reload2500')
                    candidates.append(assess(model,cp,port,'smol_new2500'))
                else:
                    common=['--input-proof',BASE/'pi0_input_final/audit.json']
                    gate=a.output/'pi0_gate2'
                    execute(PI,'VLA.family_transfer_r54.fit_pi0',[*common,'--updates',2,'--output',gate],model+'_gate2')
                    cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);proof=a.output/'pi0_reload2.json'
                    execute(PI,'VLA.family_transfer_r54.reload_pi0',['--checkpoint',cp,'--output',proof],model+'_reload2')
                    for budget in (1500,3000):
                        fit=a.output/f'pi0_fit{budget}';stage=model+f'_actual{budget}_EEF_LoRA_updates';status()
                        execute(PI,'VLA.family_transfer_r54.fit_pi0',[*common,'--updates',budget,'--qualification',proof,
                            '--output',fit],model+f'_fit{budget}')
                        cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
                        execute(PI,'VLA.family_transfer_r54.reload_pi0',['--checkpoint',cp,'--output',a.output/f'pi0_reload{budget}.json'],model+f'_reload{budget}')
                        candidate=assess(model,cp,port,f'pi0_new{budget}');candidates.append(candidate)
                        if candidate['score'][0]>=24:break
                best=max(candidates,key=lambda c:(*c['score'],c['baseline']))
                atomic_json(a.output/(model+'_frozen_final_method.json'),{**best,'candidates':candidates,'final_outcomes_used_for_selection':False})
                stage=model+'_LOCKED_common_final120_NO_more_tuning';status()
                serve(model,Path(best['checkpoint']),port,model+'_selected_final',best['baseline'])
                final=a.output/(model+'_final120')
                execute(SIM,'VLA.family_transfer_r54.evaluate',['--model',model,'--port',port,'--prefix',best['prefix'],
                    '--cohort',a.output/'final120.json','--output',final],model+'_final120')
                stop();results.append({'model':model,'status':'completed','progress':str(final/'progress.json')});status()
            except Exception as e:
                if child is not None and child.poll() is None:raise
                stop();results.append({'model':model,'status':'failed_integration_NOT_zero_SR','error':f'{type(e).__name__}: {e}'});status()
        stage='finite_four_family_extension_attempts_finished_REVIEW_each_result';status('completed')
    except Exception as e:status('failed_preserve_all_evidence',error=f'{type(e).__name__}: {e}');raise
    finally:stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_suffix('.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-u','-m','VLA.family_transfer_r54.run','--output',str(a.output.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'owner_pid':child.pid,'output':str(a.output.resolve())}))
    else:main(a)
