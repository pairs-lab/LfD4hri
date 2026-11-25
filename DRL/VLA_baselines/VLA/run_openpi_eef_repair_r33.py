"""Automatic BOTH OpenPI action-space/recovery adaptations and actual120.

Preserves the joint-output0/40 result. Uses native model/loss with already
measured shared EEF labels, own146 mixed batches and actual2->reload8->2000
updates. No unsafe competition with registered all-family predecessors.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import gpu_state
from VLA.wait_render_then_resume_queues import render_jobs

ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
PI=ROOT/'VLA/third_party/openpi/.venv/bin/python';SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
ADMISSION=AUDIT/'all_family_privileged_r33_v4/native_view_admission.json'
SOURCE=ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'
PREDECESSORS=[AUDIT/'openpi_native32_actual_fit_r32',AUDIT/'scene67_resume_r33',AUDIT/'all_family_privileged_r33_v4']


def alive(path):
    state=json.loads((path/'status.json').read_text());proc=Path('/proc')/str(state.get('pid',0))
    try:return (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (FileNotFoundError,IndexError):return False


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);stage='waiting_native_recovery_admission';models={};history=[];active=server=None
    files=[Path(__file__),ROOT/'VLA/pi0/eef_scene32_training.py',ROOT/'VLA/pi0/eef_scene32_server.py',
        ROOT/'VLA/pi0/fast_decoder_integrity.py',ROOT/'VLA/privileged_native_server.py',ROOT/'VLA/scene64_r33_client.py',
        ROOT/'VLA/pi0/audit_scene32_saved.py',ROOT/'VLA/evaluate_privileged_r33.py',ROOT/'VLA/native_corrective_dataset.py',
        ROOT/'VLA/pi0/scene32_training.py',ROOT/'VLA/pi0/scene32_inputs.py',ROOT/'VLA/scene64_dataset.py']
    frozen={str(p):sha256_file(p) for p in files};deadline=time.monotonic()+a.wait_hours*3600
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OPENPI_DATA_HOME':str(ROOT/'VLA/checkpoints/openpi'),'PYTHONUNBUFFERED':'1'}
    env.pop('JAX_PLATFORMS',None)
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'error':error,'models':models,'history':history,
            'frozen_files':frozen,'updated_at_utc':datetime.now(timezone.utc).isoformat(),'old_joint_output_results_preserved':True,
            'same_robot_scene_scoring_safety':True,'high_level_actions_neural_only':True,'new_success_not_established':True})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite remaining-deadline budget; preserve in-flight action process')
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen EEF repair code changed')
    def execute(label,command,cpu=False):
        nonlocal stage,active
        stage=label;check()
        while render_jobs():save();time.sleep(30);check()
        save()
        with (output/(label+'.log')).open('x') as log:
            active=subprocess.Popen(list(map(str,command)),cwd=ROOT,env={**env,**({'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'} if cpu else {})},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'pid':active.pid,'stage':label,'command':list(map(str,command))})
            while active.poll() is None:save();time.sleep(15);check()
        history.append({'stage':label,'exit_code':active.returncode})
        if active.returncode:raise RuntimeError(label+' failed; actual artifacts preserved')
    def wait_gpu():
        nonlocal stage
        stage='waiting_all_registered_cohorts_and_exclusive_GPU'
        while True:
            check();gpu=gpu_state()
            if not any(alive(p) for p in PREDECESSORS) and not gpu['compute_pids'] and gpu['free_mib']>=14500 and not render_jobs():return
            save();time.sleep(30)
    def close():
        nonlocal server
        if server is not None and server.poll() is None and (active is None or active.poll() is not None):server.terminate();server.wait(timeout=30)
        server=None
    def serve(model,checkpoint,label,port):
        nonlocal server,stage
        stage=label;check();save()
        with (output/(label+'.log')).open('x') as log:
            server=subprocess.Popen(list(map(str,[PI,'-m','VLA.pi0.eef_scene32_server','--model',model,'--checkpoint',checkpoint,
                '--output',output/(label+'_runtime'),'--port',port])),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        import requests
        until=time.monotonic()+1800
        while True:
            if server.poll() is not None:raise RuntimeError('Own native EEF server failed; not task SR')
            try:
                r=requests.get(f'http://127.0.0.1:{port}/metadata',timeout=2)
                if r.ok and r.json().get('checkpoint')==str(checkpoint):return
            except requests.RequestException:pass
            if time.monotonic()>until:raise TimeoutError('Own native EEF startup deadline')
            save();time.sleep(3)
    try:
        while not ADMISSION.exists():check();save();time.sleep(30)
        # Avoid overlapping the other repair's large CPU token/lineage gates.
        while json.loads((PREDECESSORS[-1]/'status.json').read_text())['stage'] in (
            'native_recovery_admission','rdt_native_corrective_batches','pi0_native_corrective_batches','pi0_fast_native_corrective_batches'):
            check();save();time.sleep(30)
        for model in ('pi0','pi0_fast'):
            execute(model+'_actual_EEF_mixed_batches',[PI,'-m','VLA.pi0.eef_scene32_training','audit','--model',model,
                '--projection',PROJECTION,'--admission',ADMISSION,'--output',output/(model+'_actual_EEF_mixed_batches')],True)
        for number,model in enumerate(('pi0','pi0_fast')):
            try:
                wait_gpu();port=5821+number;qualification=output/(model+'_EEF_qualification.json')
                for updates in (2,2000):
                    requested=output/(model+'_EEF_'+str(updates))
                    command=[PI,'-m','VLA.pi0.eef_scene32_training','train','--model',model,'--projection',PROJECTION,
                        '--admission',ADMISSION,'--output',requested,'--updates',updates]
                    if updates>2:command+=['--qualification',qualification]
                    execute(model+'_EEF_fit_'+str(updates),command)
                    experiment=output/(model+'_scene32_EEF_recovery')/requested.name;precision=output/(model+'_EEF_precision_'+str(updates)+'.json')
                    execute(model+'_EEF_precision_'+str(updates),[PI,'-m','VLA.pi0.audit_scene32_saved','--experiment',experiment,
                        '--updates',updates,'--output',precision],True)
                    proof=json.loads(precision.read_text());cp=Path(proof['checkpoint']);label=model+'_EEF_server_'+str(updates)
                    serve(model,cp,label,port);name=model+('_EEF_native8' if updates==2 else '_EEF_development120')
                    evaluate=[SIM,'-m','VLA.evaluate_privileged_r33','--model',model,'--checkpoint',cp,'--port',port,'--output',output/name]
                    if updates==2:evaluate+=['--smoke-source',SOURCE]
                    execute(name,evaluate);close();result=json.loads((output/name/'progress.json').read_text())
                    if updates==2:
                        if result['status']!='completed8_integration_NOT_SR' or result['neural_controls']!=8:raise ValueError('Own EEF actual8 controls incomplete')
                        reload=output/(label+'_runtime/exact_reload.json')
                        if not json.loads(reload.read_text())['passed']:raise ValueError('Own EEF exact reload failed')
                        from VLA.pi0.eef_scene32_training import ACTION_SCHEMA
                        paths=[precision,reload,output/name/'progress.json',ADMISSION]
                        atomic_json(qualification,{'passed':True,'model':model,'action_schema':ACTION_SCHEMA,
                            'actual_optimizer_updates':2,'actual_neural_controls':8,'projection_audit_sha256':sha256_file(PROJECTION/'audit.json'),
                            'recovery_admission_sha256':sha256_file(ADMISSION),'bound_files':{**frozen,**{str(p):sha256_file(p) for p in paths}},
                            'model_SR_established':False})
                if result['completed_trials']!=120:raise ValueError('Incomplete actual EEF learned cohort')
                models[model]={'status':result['status'],'completed_trials':120,'successes':result['successes'],'development_SR120':result['successes']/120,
                    'variant':'native_OpenPI_EEF_delta_with_simstate32_and_recovery20pct','final_test_used':False};save()
            except Exception as error:
                if active is not None and active.poll() is None:raise
                models[model]={'status':'failed_preserve_actual_artifacts_NOT_zero_SR','error':f'{type(error).__name__}: {error}'};close();save()
        stage='completed_two_actual_EEF_adaptation_attempts';save('completed')
    except Exception as error:save('failed_preserve_all_artifacts',f'{type(error).__name__}: {error}');raise
    finally:close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=54);p.add_argument('--launch',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    if not 0<a.wait_hours<=58:raise ValueError('Finite deadline budget required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate action-space adaptation')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_openpi_eef_repair_r33','--output',str(a.output),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'no_existing_process_interrupted':True});print(child.pid)
