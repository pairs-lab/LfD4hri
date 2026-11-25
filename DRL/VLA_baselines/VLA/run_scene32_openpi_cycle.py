"""Automatic BOTH OpenPI families: real32 gates ->2 ->reload8 ->1k ->dev40.

GPU stages wait for the registered RDT, old native queue and Smol67 cycle,
including idle gaps; no competing trainer/episode is signalled. Each family
can fail independently without silently converting missing trials to0%.
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
from VLA.run_cross_vla120 import ROOT,gpu_state

SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python';PI=ROOT/'VLA/third_party/openpi/.venv/bin/python'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
SOURCE=ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'
PREDECESSORS=[ROOT/'VLA/audits/rdt_privileged_postfit_r25_20261014',
    ROOT/'VLA/audits/other_vla_privileged_r29_20261014/gpu_queue_rdt_serialized_20261015',
    ROOT/'VLA/audits/scene67_deadline_r31_20261015/automatic_full_gate_fit_v2']


def predecessor_pending(state):
    if not state:return True
    proc=Path('/proc')/str(state['pid'])
    alive=proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    return alive or state.get('status') not in ('completed','failed','failed_preserve_all_artifacts','failed_preserve_all_actual_artifacts')


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'VLA/pi0/scene32_training.py',ROOT/'VLA/pi0/scene32_inputs.py',ROOT/'VLA/pi0/scene32_server.py',
        ROOT/'VLA/pi0/audit_scene32_saved.py',ROOT/'VLA/evaluate_scene64_native.py',ROOT/'VLA/scene64_native_client.py',
        ROOT/'VLA/pi0/scene64_training.py',ROOT/'VLA/scene64_dataset.py',ROOT/'VLA/scene64_conditioning.py',
        ROOT/'VLA/pi0/training.py',ROOT/'VLA/pi0/augmentation.py']
    frozen={str(p):sha256_file(p) for p in files};history=[];models={};stage='waiting_complete_physical_extraction';active=server=None
    deadline=time.monotonic()+a.wait_hours*3600
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','OPENPI_DATA_HOME':str(ROOT/'VLA/checkpoints/openpi'),'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    env.pop('JAX_PLATFORMS',None)
    def save(status,error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'models':models,'history':history,
            'error':error,'frozen_files':frozen,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'deadline_remaining_seconds':max(0,deadline-time.monotonic()),'no_existing_job_signalled':True,
            'native_VLA_is_only_action_producer':True,'all_models_success_rate_known':False})
    def check():
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen native32 pipeline changed')
    def execute(name,command,extra=None):
        nonlocal active,stage
        stage=name;check();save('running')
        with (output/(name+'.log')).open('x') as log:
            active=subprocess.Popen(list(map(str,command)),cwd=ROOT,env={**env,**(extra or {})},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'pid':active.pid,'stage':name,'command':list(map(str,command))})
            while active.poll() is None:
                save('running');time.sleep(15)
                if time.monotonic()>deadline:raise TimeoutError('Owned in-flight stage retained; no action process killed')
        history.append({'stage':name,'exit_code':active.returncode})
        if active.returncode:raise RuntimeError(name+' failed; preserve all actual artifacts')
    def wait_gpu():
        nonlocal stage
        stage='waiting_registered_predecessors_and_quiet_GPU'
        while True:
            check();pending=any(predecessor_pending(json.loads((p/'status.json').read_text())) for p in PREDECESSORS)
            gpu=gpu_state()
            if not pending and not gpu['compute_pids'] and gpu['free_mib']>=14500:return
            save('running')
            if time.monotonic()>deadline:raise TimeoutError('Finite reserved-GPU budget expired; no previous job signalled')
            time.sleep(30)
    def stop_idle_server():
        nonlocal server
        if server is not None and server.poll() is None:
            if active is not None and active.poll() is None:return
            server.terminate();server.wait(timeout=30)
        server=None
    def start_server(model,cp,label,port):
        nonlocal server,stage
        stage=label;check();save('running')
        command=[PI,'-m','VLA.pi0.scene32_server','--model',model,'--checkpoint',cp,'--port',port,'--output',output/(label+'_runtime')]
        with (output/(label+'.log')).open('x') as log:
            server=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        import requests
        until=time.monotonic()+1800
        while True:
            if server.poll() is not None:raise RuntimeError('Actual native32 server failed; not scored task failure')
            try:
                reply=requests.get(f'http://127.0.0.1:{port}/metadata',timeout=2)
                if reply.ok and reply.json().get('checkpoint')==str(cp):return
            except requests.RequestException:pass
            if time.monotonic()>until:raise TimeoutError('Native32 startup failed before actions')
            save('running');time.sleep(3)
    try:
        while not (PROJECTION/'audit.json').exists():
            check();save('running')
            if json.loads((PROJECTION/'progress.json').read_text())['status'].startswith('failed'):raise ValueError('Full physical source extraction failed')
            if time.monotonic()>deadline:raise TimeoutError('Finite full-source deadline')
            time.sleep(30)
        for model in ('pi0','pi0_fast'):
            folder=output/(model+'_full_native32_batches')
            execute(model+'_full_native32_batches',[PI,'-m','VLA.pi0.scene32_training','audit','--model',model,'--projection',PROJECTION,'--output',folder],{'JAX_PLATFORMS':'cpu'})
            actual=json.loads((folder/'native_batches.json').read_text())
            if not actual['passed'] or actual['pilot_only']:raise ValueError('Actual FULL native32 loader gate failed')
            models[model]={'status':'full_CPU_input_gate_passed_CUDA_pending','task_SR':None}
        for index,model in enumerate(('pi0','pi0_fast')):
            try:
                wait_gpu();port=5761+index;qualification=output/(model+'_actual_smoke_qualification.json')
                for updates,label in ((2,'smoke'),(1000,'train')):
                    requested=output/(model+'_'+label)
                    command=[PI,'-m','VLA.pi0.scene32_training','train','--model',model,'--projection',PROJECTION,'--output',requested,'--updates',updates]
                    if updates>2:command+=['--qualification',qualification]
                    execute(model+'_'+label,command)
                    experiment=output/(model+'_scene32_and_scene64_language')/requested.name
                    precision=output/(model+'_'+label+'_saved_precision.json')
                    execute(model+'_'+label+'_precision',[PI,'-m','VLA.pi0.audit_scene32_saved','--experiment',experiment,'--updates',updates,'--output',precision],{'JAX_PLATFORMS':'cpu'})
                    proof=json.loads(precision.read_text());cp=Path(proof['checkpoint'])
                    if updates==2:
                        native_label=model+'_smoke_reload';start_server(model,cp,native_label,port)
                        integration=output/(model+'_actual_native8')
                        execute(model+'_actual_native8',[SIM,'-m','VLA.evaluate_scene64_native','--model',model,'--checkpoint',cp,'--port',port,
                            '--smoke-source',SOURCE,'--output',integration])
                        stop_idle_server();reloaded=output/(native_label+'_runtime/exact_reload.json')
                        actual=json.loads((integration/'progress.json').read_text())
                        if actual['status']!='completed8_integration_NOT_SR' or actual['neural_controls']!=8:raise ValueError('Actual32 native probe failed')
                        dependencies={**frozen,str(precision):sha256_file(precision),str(reloaded):sha256_file(reloaded),
                            str(integration/'progress.json'):sha256_file(integration/'progress.json')}
                        atomic_json(qualification,{'passed':True,'model':model,'state_schema':proof['state_schema'],'actual_optimizer_updates':2,
                            'actual_neural_controls':8,'exact_saved_values_reload':True,'saved_masters_and_moments_FP32':True,
                            'projection_audit_sha256':sha256_file(PROJECTION/'audit.json'),'dependencies':dependencies,'model_SR_established':False})
                start_server(model,cp,model+'_postfit',port)
                evaluation=output/(model+'_development40')
                execute(model+'_development40',[SIM,'-m','VLA.evaluate_scene64_native','--model',model,'--checkpoint',cp,'--port',port,
                    '--output',evaluation,'--trials-per-action','10'])
                stop_idle_server();result=json.loads((evaluation/'progress.json').read_text())
                if result['completed_trials']!=40:raise ValueError('Incomplete actual learned development40')
                models[model]={'status':result['status'],'completed_trials':40,'successes':result['successes'],'development_SR40':result['successes']/40,
                    'checkpoint':str(cp),'untouched_final_test_used':False,'adaptation_regime':'native32_numeric_plus_full64_language_LoRA1000'}
            except Exception as error:
                if active is not None and active.poll() is None:raise
                models[model]={'status':'failed_preserve_actual_artifacts_NOT_zero_SR','error':f'{type(error).__name__}: {error}','task_SR':None}
                stop_idle_server();save('running')
        stage='completed_two_family_attempts_review_actual_results';save('completed')
    except Exception as error:save('failed_preserve_all_artifacts',f'{type(error).__name__}: {error}');raise
    finally:stop_idle_server()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=60);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=66:raise ValueError('Finite deadline required')
    if not a.launch:return run(a)
    marker=a.output.with_name(a.output.name+'_launch.json')
    if a.output.exists() or marker.exists():raise FileExistsError('No duplicate model queue')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
        child=subprocess.Popen([str(SIM),'-m','VLA.run_scene32_openpi_cycle','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(marker,{'pid':child.pid,'no_existing_job_signalled':True});print(child.pid)


if __name__=='__main__':main()
