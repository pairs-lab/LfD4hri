"""Finite automatic all-family repair -> own qualification -> fit -> test.

Existing OpenPI runs and the evidence-preserving Smol continuation finish
first. CPU data gates can proceed meanwhile. Failures are isolated by family,
never scored as0% or silently retried. No task planner replaces neural actions.
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
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python';OV=ROOT/'VLA/.venvs/openvla/bin/python'
RDT=ROOT/'VLA/.venvs/rdt/bin/python';PI=ROOT/'VLA/third_party/openpi/.venv/bin/python'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
SOURCE=ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'
NEW_NATIVE=AUDIT/'native_corrective_full38_r32/fresh38';NEW_REPLAY=AUDIT/'native_corrective_history_rounding_r33'
OVDATA=AUDIT/'other_native_full_input_gates/openvla_scene64_source'
OVBASE=ROOT/'VLA/checkpoints/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f'
RDT_AUDIT=ROOT/'VLA/audits/development40_rdt_r14_20261013'
RDT_SOURCE=ROOT/'VLA/audits/development40_smolvla_r14_20261013/rdt_qualified_train'
RDT_BASE=ROOT/'VLA/rdt_simstate/rdt_privileged_postfit_r25_20261014/train/checkpoints/step-005000'
VISION=ROOT/'VLA/checkpoints/siglip-so400m-patch14-384'
PREDECESSORS=[AUDIT/'openpi_native32_actual_fit_r32',AUDIT/'scene67_resume_r33']


def alive(state):
    try:return (Path('/proc')/str(state.get('pid',0))/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (FileNotFoundError,IndexError):return False


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'VLA/evaluate_privileged_r33.py',ROOT/'VLA/scene64_r33_client.py',ROOT/'VLA/qualify_native_recovery_view.py',
        ROOT/'VLA/privileged_native_server.py',ROOT/'VLA/pi0/fast_decoder_integrity.py',
        ROOT/'VLA/replay_native_corrective_history.py',ROOT/'VLA/native_corrective_dataset.py',ROOT/'VLA/audit_native_corrective_batches.py',
        ROOT/'VLA/openvla/privileged_loading.py',ROOT/'VLA/openvla/scene64_server.py',ROOT/'VLA/openvla/train_scene64_prompt.py',
        ROOT/'VLA/openvla/train_privileged_prompt.py',ROOT/'VLA/openvla/finetune.py',
        ROOT/'VLA/rdt/train_scene67.py',ROOT/'VLA/rdt/scene67_native.py',ROOT/'VLA/rdt/scene67_server.py',ROOT/'VLA/rdt/audit_scene67_mixed.py',
        ROOT/'VLA/cogact/scene67_learning.py',ROOT/'VLA/cogact/scene67_server.py',ROOT/'VLA/cogact/scene67_conditioning.py',
        ROOT/'VLA/common/environment.py',ROOT/'VLA/common/body_gripper.py',ROOT/'VLA/common/physics_safety.py',
        ROOT/'VLA/common/observation_profile.py',ROOT/'asset/params/reward.xml']
    frozen={str(p):sha256_file(p) for p in files};models={};history=[];stage='waiting_complete_native_recovery_replay';active=server=None
    deadline=time.monotonic()+a.wait_hours*3600
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'PYTHONUNBUFFERED':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    env.pop('JAX_PLATFORMS',None)
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'error':error,'models':models,
            'history':history,'frozen_files':frozen,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'deadline_remaining_seconds':max(0,deadline-time.monotonic()),'same_robot_objects_success_safety':True,
            'high_level_actions_neural_only':True,'all_models_high_success_established':False})
    def check():
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen all-family repair code changed')
        if time.monotonic()>deadline:raise TimeoutError('Finite deadline; preserve owned in-flight child if any')
    def execute(name,command,cpu=False):
        nonlocal stage,active
        stage=name;check()
        while render_jobs():save();time.sleep(30);check()
        save();child_env={**env,**({'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'} if cpu else {})}
        with (output/(name+'.log')).open('x') as log:
            active=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'pid':active.pid,'stage':name,'command':list(map(str,command)),'CPU_only':cpu})
            while active.poll() is None:save();time.sleep(15);check()
        history.append({'stage':name,'exit_code':active.returncode})
        if active.returncode:raise RuntimeError(name+' failed; actual artifacts retained, not a task SR')
    def wait_gpu():
        nonlocal stage
        stage='waiting_registered_predecessors_and_exclusive_GPU'
        while True:
            check();busy=any(alive(json.loads((p/'status.json').read_text())) for p in PREDECESSORS)
            gpu=gpu_state()
            if not busy and not gpu['compute_pids'] and gpu['free_mib']>=14500 and not render_jobs():return
            save();time.sleep(30)
    def start_server(name,command,port,checkpoint):
        nonlocal stage,server
        stage=name;check();save()
        with (output/(name+'.log')).open('x') as log:
            server=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(output/'owned_server.json',{'pid':server.pid,'stage':name,'command':list(map(str,command))})
        import requests
        until=time.monotonic()+2400
        while True:
            if server.poll() is not None:raise RuntimeError(name+' actual server load failed; not a task failure')
            try:
                reply=requests.get(f'http://127.0.0.1:{port}/metadata',timeout=2)
                if reply.ok and reply.json().get('checkpoint')==str(checkpoint):return
            except requests.RequestException:pass
            if time.monotonic()>until:raise TimeoutError('Native server startup deadline')
            save();time.sleep(3)
    def close_server():
        nonlocal server
        if server is not None and server.poll() is None:
            if active is not None and active.poll() is None:return
            server.terminate();server.wait(timeout=30)
        server=None
    def evaluate(model,cp,port,name,smoke=False):
        cmd=[SIM,'-m','VLA.evaluate_privileged_r33','--model',model,'--checkpoint',cp,'--port',port,'--output',output/name]
        if smoke:cmd+=['--smoke-source',SOURCE]
        execute(name,cmd)
        result=json.loads((output/name/'progress.json').read_text())
        if smoke and (result['status']!='completed8_integration_NOT_SR' or result['neural_controls']!=8):raise ValueError('Actual8 controls incomplete')
        if not smoke and result['completed_trials']!=120:raise ValueError('Actual120 learned episodes incomplete')
        return result
    admission=output/'native_view_admission.json'
    def run_openvla():
        wait_gpu();fit=output/'openvla_fit';port=5811;qualification=output/'openvla_actual_qualification.json'
        for updates in (2,1000):
            cmd=[OV,'-m','VLA.openvla.train_scene64_prompt','--dataset',OVDATA,'--output',fit,
                '--base-model',OVBASE,'--steps',updates,'--grad-accumulation','4','--save-freq','250']
            if updates>2:cmd+=['--resume-from',fit/'checkpoints/000002','--smoke-qualification',qualification]
            execute(f'openvla_fit_{updates}',cmd);cp=fit/'checkpoints'/f'{updates:06d}'
            precision=output/f'openvla_precision_{updates}.json'
            execute(f'openvla_precision_{updates}',[OV,'-m','VLA.openvla.audit_privileged_saved_precision','--checkpoint',cp,'--updates',updates,'--output',precision],True)
            label=f'openvla_server_{updates}';runtime=output/(label+'_runtime')
            start_server(label,[OV,'-m','VLA.openvla.scene64_server','--checkpoint',cp,'--output',runtime,'--port',port,'--unnorm-key',OVDATA.name],port,cp)
            result=evaluate('openvla',cp,port,'openvla_native8' if updates==2 else 'openvla_development120',updates==2)
            close_server()
            if updates==2:
                paths=[precision,runtime/'adapter_reload.json',output/'openvla_native8/progress.json',cp/'checkpoint_manifest.json']
                if not json.loads((runtime/'adapter_reload.json').read_text())['passed']:raise ValueError('OpenVLA exact reload failed')
                atomic_json(qualification,{'passed':True,'optimizer_updates':2,'dataset_contract_sha256':sha256_file(OVDATA/'privileged_dataset.json'),
                    'trainer_sha256':sha256_file(ROOT/'VLA/openvla/train_privileged_prompt.py'),'grad_accumulation':4,'learning_rate':.0002,
                    'seed':20261014,'actual_neural_robot_controls':8,'task_SR_certified':False,
                    'dependencies':[{'path':str(p),'sha256':sha256_file(p)} for p in paths]})
        return result
    def run_rdt():
        gate=output/'rdt_mixed_native128.json'
        execute('rdt_mixed_native128',[RDT,'-m','VLA.rdt.audit_scene67_mixed','--source',RDT_SOURCE,'--projection',PROJECTION,
            '--recovery-admission',admission,'--embeddings',RDT_AUDIT/'qualified_embeddings.pt','--vision',VISION,'--output',gate],True)
        wait_gpu();port=5812;qual=output/'rdt_scene67_qualification.json';signal=output/'rdt_signal_qualification.json'
        for updates in (2,2000):
            fit=output/f'rdt_fit_{updates}'
            command=[RDT,'-m','VLA.rdt.train_scene67','--projection',PROJECTION,'--recovery-admission',admission,'--input-gate',gate,
                'train','--source',RDT_SOURCE,'--audit-report',RDT_AUDIT/'qualified_source_audit/audit.json','--action-contract-report',RDT_AUDIT/'qualified_fk.json',
                '--language-embeddings',RDT_AUDIT/'qualified_embeddings.pt','--base-checkpoint',RDT_BASE,'--vision-checkpoint',VISION,
                '--output',fit,'--steps',updates,'--action-horizon','10','--episode-batch-size','1','--samples-per-episode','4','--gradient-accumulation','4',
                '--learning-rate','.0001','--weight-decay','.01','--optimizer','adamw','--num-workers','0','--save-interval','250',
                '--seed','20261015','--signal-precision','fp32_state_action_readout_latent_v1']
            if updates>2:command+=['--scene67-qualification',qual,'--signal-qualification',signal]
            execute(f'rdt_fit_{updates}',command);cp=fit/'checkpoints'/f'step-{updates:06d}';precision=output/f'rdt_precision_{updates}.json'
            execute(f'rdt_precision_{updates}',[RDT,'-m','VLA.rdt.audit_training_precision','--checkpoint',cp,'--updates',updates,'--output',precision],True)
            execute(f'rdt_fingerprint_{updates}',[RDT,'-m','VLA.fingerprint_checkpoint',cp],True)
            label=f'rdt_server_{updates}';runtime=output/(label+'_runtime')
            start_server(label,[RDT,'-m','VLA.rdt.scene67_server','--checkpoint',cp,'--vision',VISION,'--embeddings',RDT_AUDIT/'qualified_embeddings.pt',
                '--output',runtime,'--port',port],port,cp)
            result=evaluate('rdt',cp,port,'rdt_native8' if updates==2 else 'rdt_development120',updates==2);close_server()
            if updates==2:
                from VLA.rdt.scene67_native import SCHEMA
                from VLA.fingerprint_checkpoint import verify_fingerprint
                paths=[precision,runtime/'exact_reload.json',output/'rdt_native8/progress.json',gate,admission,cp/'vla_checkpoint_manifest.json']
                common={'passed':True,'schema':SCHEMA,'actual_optimizer_updates':2,'actual_neural_controls':8,
                    'smoke_checkpoint':str(cp),'checkpoint_content_sha256':verify_fingerprint(cp)['content_tree_sha256'],
                    'signal_precision':'fp32_state_action_readout_latent_v1','neural_success_claimed':False,
                    'source_index_sha256':sha256_file(RDT_SOURCE/'index.json'),'source_manifest_sha256':sha256_file(RDT_SOURCE/'dataset_manifest.json'),
                    'bound_files':{**frozen,**{str(p):sha256_file(p) for p in paths}}}
                atomic_json(qual,common);atomic_json(signal,{**common,'scope':'RDT_FP32_signal_two_update_integration_qualification_NOT_task_success'})
        return result
    def run_cogact():
        wait_gpu();port=5813;qualification=output/'cogact_own_qualification.json'
        for pilot,updates in ((True,2),(False,2000)):
            cache=output/('cogact_pilot_cognition' if pilot else 'cogact_complete_cognition')
            command=[OV,'-m','VLA.cogact.scene67_learning','cache','--projection',PROJECTION,'--admission',admission,'--output',cache]
            if pilot:command+=['--pilot']
            execute('cogact_cache_'+str(updates),command)
            fit=output/f'cogact_fit_{updates}'
            cmd=[OV,'-m','VLA.cogact.scene67_learning','train','--cache',cache,'--output',fit,'--updates',updates]
            if not pilot:cmd+=['--qualification',qualification]
            execute('cogact_fit_'+str(updates),cmd)
            label=f'cogact_server_{updates}';runtime=output/(label+'_runtime')
            start_server(label,[OV,'-m','VLA.cogact.scene67_server','--checkpoint',fit,'--output',runtime,'--port',port],port,fit)
            result=evaluate('cogact',fit,port,'cogact_native8' if pilot else 'cogact_development120',pilot);close_server()
            if pilot:
                cfg=json.loads((fit/'scene67_contract.json').read_text())
                paths=[fit/'saved_precision.json',runtime/'exact_reload.json',output/'cogact_native8/progress.json',cache/'audit.json']
                atomic_json(qualification,{'passed':True,'schema':cfg['schema'],'actual_neural_controls':8,'actual_optimizer_updates':2,
                    'conditioning_stats_sha256':cfg['conditioning_stats_sha256'],'model_SR_established':False,
                    'bound_files':{**frozen,**{str(p):sha256_file(p) for p in paths}}})
        return result
    try:
        save()
        while not (NEW_REPLAY/'audit.json').exists():check();save();time.sleep(30)
        if not json.loads((NEW_REPLAY/'audit.json').read_text())['passed']:raise ValueError('Full native physical replay failed; no training admission')
        execute('native_recovery_admission',[SIM,'-m','VLA.qualify_native_recovery_view','--candidate',NEW_NATIVE,
            '--replay',NEW_REPLAY/'audit.json','--output',admission],True)
        for model,python in (('rdt',RDT),('pi0',PI),('pi0_fast',PI)):
            try:execute(model+'_native_corrective_batches',[python,'-m','VLA.audit_native_corrective_batches','--model',model,
                '--admission',admission,'--output',output/(model+'_native_corrective_batches')],True)
            except RuntimeError as error:models[model]={'input_gate_error':str(error),'actual_task_SR':None};save()
        for model,worker in (('openvla',run_openvla),('rdt',run_rdt),('cogact',run_cogact)):
            try:
                models.setdefault(model,{})['status']='qualified_fitting_and_testing_in_progress';save()
                result=worker();models[model]={'status':result['status'],'completed_trials':120,'successes':result['successes'],
                    'development_SR120':result['successes']/120,'final_test_used':False}
            except Exception as error:
                if active is not None and active.poll() is None:raise
                models.setdefault(model,{}).update(status='failed_preserve_actual_artifacts_NOT_zero_SR',error=f'{type(error).__name__}: {error}')
                close_server();save()
        stage='completed_finite_all_family_attempts_review_actual_SR';save('completed')
    except Exception as error:save('failed_preserve_all_artifacts',f'{type(error).__name__}: {error}');raise
    finally:close_server()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=48);p.add_argument('--launch',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    if not 0<a.wait_hours<=54:raise ValueError('Finite deadline required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('Do not duplicate all-family repair queue')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_all_privileged_r33','--output',str(a.output),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'no_existing_job_interrupted':True});print(child.pid)
