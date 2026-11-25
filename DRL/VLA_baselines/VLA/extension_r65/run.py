"""Detached serialized extension, after all existing R62 model jobs release GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import (ROOT,OUT,R62,R56,DATA,BASE,PY,SIM,PI,ENV,JEPA_ENV,SCHEMA,
                     sources,verify,read,utc,predecessor_busy,resource_ready,model_python,model_env)
from .results import refresh


def choose_prefix(pilots):
    def key(item):
        prefix,p=item;rows=p['rows']
        if p['completed_trials']!=8 or len(rows)!=8:raise ValueError('Complete fixed pilot8 required')
        return (sum(r['success'] for r in rows),-sum(bool(r['safety_violation']) for r in rows),prefix)
    return max(pilots,key=key)[0]


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise ValueError('Refuse duplicate extension/evaluations')
        bound=sources();verify(bound);atomic_json(out/'BOUND_CODE_AND_ENVIRONMENT.json',bound)
        atomic_json(out/'PREDECLARATION.json',{'UTC':utc(),'models':['vla_jepa','pi0_fast'],
            'existing4model_owner_not_modified':str(R62),'wait_all_R62_actual_children_servers':True,
            'same800cohort':str(R62/'cohort.json'),'per_action':200,'per_model':800,
            'JEPA_total_added_updates':2500,'FAST_new_optimizer_updates':1500,
            'selection_only_fixed_development8_then40_plus40':True,
            'shared_robot_gripper_camera_physics_safety_success':True,
            'PUT_goal_region_NOT_release':True,'no_new_Oracle_or_geometric_action_repair':True,
            'forecast_four_additional_UNRUN_evaluation_cohorts_NOT_training_seed_variance':True})
        child=server=log=None;stage='CPU_preparation_then_wait_existing_R62_no_preemption';models={}
        def status(state='running',**kw):
            value={'status':state,'stage':stage,'pid':os.getpid(),'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,'models':models,'UTC':utc(),**kw}
            atomic_json(out/'owner_status.json',value);refresh(out,value)
        def execute(py,module,opts,folder,label,env=None):
            nonlocal child
            verify(bound);folder.mkdir(parents=True,exist_ok=True)
            with (folder/(label+'.log')).open('x') as stream:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env or ENV,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:status();time.sleep(10)
                code=child.returncode
            child=None
            if code:raise RuntimeError(f'{label} exited{code}; inspect preserved log, not a model SR')
        def wait_slot():
            until=time.monotonic()+a.wait_hours*3600
            while True:
                busy=predecessor_busy();ok,r=resource_ready()
                if not busy and ok:break
                if time.monotonic()>until:raise TimeoutError('Finite resource wait expired; no foreign process stopped')
                status(waiting_R62=busy,resources=r);time.sleep(15)
        def stop():
            nonlocal server,log
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)
            server=None
            if log is not None:log.close();log=None
        def serve(model,cp,folder,label):
            nonlocal server,log
            wait_slot();verify(bound);log=(folder/(label+'_server.log')).open('x')
            server=subprocess.Popen([str(model_python(model)),'-u','-m','VLA.extension_r65.server',
                '--model',model,'--checkpoint',str(cp),'--port','5925','--output',str(folder/(label+'_runtime'))],
                cwd=ROOT,env=model_env(model),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            until=time.monotonic()+1800
            while True:
                status()
                if server.poll() is not None:raise RuntimeError('Own actual neural server exited')
                try:
                    response=requests.get('http://127.0.0.1:5925/health',timeout=5);response.raise_for_status()
                    meta=response.json()
                    if meta['model']!=model or meta['checkpoint']!=str(cp):raise ValueError('Wrong native model/checkpoint')
                    return
                except requests.RequestException:
                    if time.monotonic()>until:raise TimeoutError('Native model startup exceeded30minutes')
                    time.sleep(5)
        def evaluate(model,cohort,dest,prefix,smoke=False):
            opts=['--model',model,'--cohort',cohort,'--output',dest,'--port',5925,'--prefix',prefix]
            if smoke:opts+=['--smoke']
            execute(SIM,'VLA.extension_r65.evaluate',opts,dest.parent,dest.name)
        try:
            status()
            # Preparation has no CUDA context. Do not import models in this owner.
            preparation=out/'preparation'
            if not (preparation/'DOWNLOADS.json').exists():
                execute(sys.executable,'VLA.extension_r65.fetch',[],preparation,'fetch',
                    {**ENV,'HF_HUB_OFFLINE':'0','TRANSFORMERS_OFFLINE':'0','HF_HUB_DISABLE_XET':'1'})
            verify(read(preparation/'DOWNLOADS.json')['files_sha256'])
            for model in ('vla_jepa','pi0_fast'):
                folder=out/model;folder.mkdir(exist_ok=False);models[model]={'status':'CPU_audit_pending','completed800':0}
                try:
                    stage=model+'_own_native_CPU_input_audit';status()
                    input_dir=preparation/(model+'_native_input')
                    if not input_dir.exists():
                        execute(model_python(model),'VLA.extension_r65.audit_inputs',['--model',model,'--output',input_dir],
                            preparation,model+'_input_audit',{**model_env(model),'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})
                    proof=read(input_dir/'audit.json')
                    if not proof.get('passed') or not proof.get('all4_tasks'):raise ValueError('Own native input audit incomplete')
                    verify(proof['code'])
                    stage=model+'_wait_all_R62_and_exclusive_GPU';status();wait_slot()
                    train_module='VLA.extension_r65.train_jepa' if model=='vla_jepa' else 'VLA.extension_r65.train_fast'
                    stage=model+'_actual_two_Adam_updates';status()
                    gate=folder/'gate2'
                    execute(model_python(model),train_module,['fit','--output',gate,'--updates',2,
                        '--input-proof',input_dir/'audit.json'],folder,'gate2',model_env(model))
                    cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);reload_proof=folder/'fresh_reload2.json'
                    stage=model+'_fresh_process_exact_saved_native_reload8';status();wait_slot()
                    execute(model_python(model),train_module,['reload','--checkpoint',cp,'--output',reload_proof],
                        folder,'fresh_reload2',model_env(model))
                    if not read(reload_proof).get('passed'):raise ValueError('Own fresh reload gate failed')
                    smoke_cohort=folder/'smoke_cohort.json'
                    atomic_json(smoke_cohort,{'seeds':[20268001],'role':'development_integration_NOT_SR'})
                    stage=model+'_actual_neural8_simulator_controls_NOT_SR';status();serve(model,cp,folder,'gate')
                    smoke=folder/'native8_NOT_SR';evaluate(model,smoke_cohort,smoke,1,True);stop()
                    if not read(smoke/'SMOKE_NOT_SR.json')['passed']:raise ValueError('Native robot8 integration failed')
                    stage=model+'_real_native_finetune';status();wait_slot()
                    fit=folder/('fit2500' if model=='vla_jepa' else 'fit1500')
                    opts=['fit','--output',fit,'--updates',2500 if model=='vla_jepa' else 1500,
                        '--input-proof',input_dir/'audit.json','--reload-proof',reload_proof,'--robot-proof',smoke/'SMOKE_NOT_SR.json']
                    if model=='vla_jepa':opts+=['--resume',cp]
                    execute(model_python(model),train_module,opts,folder,'long_fit',model_env(model))
                    cp=Path(read(fit/'latest_checkpoint.json')['checkpoint']);fresh=folder/'fresh_reload_final.json'
                    stage=model+'_fresh_reload_completed_fit';status();wait_slot()
                    execute(model_python(model),train_module,['reload','--checkpoint',cp,'--output',fresh],
                        folder,'fresh_reload_final',model_env(model))
                    if not read(fresh)['passed']:raise ValueError('Completed checkpoint fresh reload failed')
                    stage=model+'_native_development_selection';status();serve(model,cp,folder,'development')
                    pilots=[]
                    for prefix in ((3,5,7) if model=='vla_jepa' else (8,16,32)):
                        dest=folder/f'prefix{prefix}_pilot8';evaluate(model,R56/'pilot8.json',dest,prefix)
                        pilots.append((prefix,read(dest/'progress.json')))
                    prefix=choose_prefix(pilots)
                    for name in ('development40','confirmation40'):
                        evaluate(model,R56/(name+'.json'),folder/name,prefix)
                    stop()
                    stage=model+'_freeze_before_any800outcome';status()
                    cp_files=[p for p in cp.rglob('*') if p.is_file()]
                    freeze={'model':model,'checkpoint':str(cp),'prefix':prefix,'UTC':utc(),
                        'checkpoint_sha256':{str(p):sha256_file(p) for p in cp_files},
                        'selection_only_development':True,'native_schema':SCHEMA,
                        'development_progress':str(folder/'development40/progress.json'),
                        'confirmation_progress':str(folder/'confirmation40/progress.json'),
                        'new800_outcomes_used':False}
                    atomic_json(folder/'FROZEN_BEFORE_800.json',freeze)
                    stage=model+'_actual_locked800_no_video';models[model]['status']='running800';status()
                    serve(model,cp,folder,'frozen800');evaluate(model,R62/'cohort.json',folder/'episodes',prefix);stop()
                    stage=model+'_native_receipts_and_full800_audit';status()
                    from .results import audit
                    audit(folder);models[model]={'status':'completed800_and_audit','completed800':800};status()
                except Exception as error:
                    if child is not None and child.poll() is None:
                        child.terminate();child.wait(timeout=30)
                    child=None
                    stop();models[model]={'status':'integration_error_NOT_zero_SR','error':f'{type(error).__name__}: {error}'}
                    atomic_json(folder/'INTEGRATION_ERROR_NOT_MODEL_SR.json',{'error':str(error),'traceback':traceback.format_exc(),'UTC':utc()})
                    status()
            stage='finished_extension_review_errors_if_any';status('completed' if all(v['status']=='completed800_and_audit' for v in models.values()) else 'finished_with_integration_errors_NOT_fake_SR')
        except Exception as error:
            status('owner_error_preserve_all',error=f'{type(error).__name__}: {error}');raise
        finally:
            if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
            stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--wait-hours',type=float,default=72);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'owner.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','VLA.extension_r65.run','--output',str(a.output.resolve()),
                '--wait-hours',str(a.wait_hours)],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'owner_pid':child.pid,'output':str(a.output.resolve())}))
    else:main(a)
