"""Finite automatic OpenVLA corrective refinement AFTER other VLA families.

Never interrupts the registered six-family cohorts. CPU gates run meanwhile;
own CUDA2/reload8 qualify a1000-update warm refinement and40 diagnostics.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import gpu_state
from VLA.wait_render_then_resume_queues import render_jobs

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
PYTHON=ROOT/'VLA/.venvs/openvla/bin/python'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
CURRENT=BASE/'all_family_privileged_r33_v4'
WARM=CURRENT/'openvla_fit/checkpoints/001000'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
ADMISSION=CURRENT/'native_view_admission.json'
DATA=BASE/'other_native_full_input_gates/openvla_scene64_source'
SOURCE=ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'
PARENTS=[CURRENT,BASE/'openpi_EEF_repair_r33_v4',BASE/'scene67_resume_r33_v2']


def ordered_parents():
    pointer=BASE/'Smol_GTcrop_motion_r36_active.json'
    if not pointer.exists():raise ValueError('Explicit earlier target-crop fit owner required')
    crop=Path(json.loads(pointer.read_text())['cycle'])
    if crop.parent!=BASE or not crop.name.startswith('Smol_GTcrop_motion_r36_'):
        raise ValueError('Unexpected target-view fit priority owner')
    return [*PARENTS,crop]


def live(root):
    try:
        state=json.loads((root/'status.json').read_text())
        proc=Path('/proc')/str(state['pid'])
        return (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z' and str(root).encode() in (proc/'cmdline').read_bytes().split(b'\0')
    except (OSError,KeyError,IndexError):
        return False


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(BASE/'OpenVLA_balanced_recovery_active.json',{'cycle':str(output),'pid':os.getpid(),'purpose':'latest untrained motion/sampling refinement generation'})
    stage='preparing_actual_native_balanced_recovery_batches';active=server=None;history=[]
    deadline=time.monotonic()+a.wait_hours*3600
    files=[Path(__file__),ROOT/'VLA/openvla/recovery_balanced.py',ROOT/'VLA/openvla/privileged_loading.py',
        ROOT/'VLA/openvla/scene64_server.py',ROOT/'VLA/openvla/train_privileged_prompt.py',ROOT/'VLA/openvla/finetune.py',
        ROOT/'VLA/scene64_dataset.py',ROOT/'VLA/native_corrective_dataset.py',ROOT/'VLA/evaluate_privileged_r33.py',
        ROOT/'VLA/scene64_r33_client.py',ROOT/'VLA/scene64_conditioning.py',ROOT/'asset/params/reward.xml']
    frozen={str(p):sha256_file(p) for p in files}
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'error':error,'history':history,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_files':frozen,
            'other_VLA_families_have_GPU_priority':True,'runtime_actions_native_VLA_only':True,'new_SR_established':False})
    def check():
        if time.monotonic()>deadline:
            raise TimeoutError('Finite refinement budget; in-flight owned child is preserved')
        if any(sha256_file(p)!=digest for p,digest in frozen.items()):
            raise ValueError('Immutable refinement recipe changed')
    def execute(label,command,cpu=False):
        nonlocal stage,active
        stage=label;check()
        while render_jobs():save();time.sleep(30);check()
        with (output/(label+'.log')).open('x') as log:
            active=subprocess.Popen(list(map(str,command)),cwd=ROOT,env={**env,**({'CUDA_VISIBLE_DEVICES':''} if cpu else {})},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'pid':active.pid,'stage':stage,'command':list(map(str,command))})
            while active.poll() is None:save();time.sleep(15);check()
        history.append({'stage':label,'exit_code':active.returncode});save()
        if active.returncode:raise RuntimeError('Actual refinement stage failed: '+label)
    def close_server():
        nonlocal server
        if server is not None and server.poll() is None and (active is None or active.poll() is not None):
            server.terminate();server.wait(timeout=30)
        server=None
    def serve(cp,label):
        nonlocal stage,server
        stage=label;save();check()
        runtime=output/(label+'_runtime')
        command=[PYTHON,'-m','VLA.openvla.scene64_server','--checkpoint',cp,'--output',runtime,'--port','5831','--unnorm-key',DATA.name]
        with (output/(label+'.log')).open('x') as log:
            server=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(output/'owned_server.json',{'pid':server.pid,'command':list(map(str,command))})
        until=time.monotonic()+2400
        while True:
            if server.poll() is not None:raise RuntimeError('Actual refinement server failed')
            try:
                response=requests.get('http://127.0.0.1:5831/metadata',timeout=3)
                if response.ok and response.json().get('checkpoint')==str(cp):return runtime
            except requests.RequestException:pass
            if time.monotonic()>until:raise TimeoutError('Finite native server startup')
            save();time.sleep(3)
    try:
        # The existing1000-update parent is verified, never relabelled a new
        # optimizer resume. New Adam2->1000 has its own exact saved state.
        base_model=json.loads((WARM/'train_config.json').read_text())['base_model']
        for batch in (2,1):
            execute('actual_native128_batch'+str(batch),[PYTHON,'-m','VLA.openvla.recovery_balanced','audit',
                '--dataset',DATA,'--projection',PROJECTION,'--admission',ADMISSION,'--warmstart',WARM,
                '--batch-size',batch,'--output',output/('native128_batch'+str(batch))],True)
        stage='waiting_all_registered_VLA_families_first';save()
        while True:
            check();gpu=gpu_state()
            if not any(live(root) for root in ordered_parents()) and not gpu['compute_pids'] and gpu['free_mib']>=14500 and not render_jobs():break
            save();time.sleep(30)
        for batch in (2,1):
            fit=output/('fit_batch'+str(batch));gate=output/('native128_batch'+str(batch)+'/audit.json')
            command=[PYTHON,'-m','VLA.openvla.recovery_balanced','train','--dataset',DATA,'--projection',PROJECTION,
                '--admission',ADMISSION,'--warmstart',WARM,'--base-model',base_model,'--batch-size',batch,'--gate',gate,'--output',fit]
            try:execute('CUDA2_batch'+str(batch),command+['--updates','2'])
            except RuntimeError:
                log=(output/('CUDA2_batch'+str(batch)+'.log')).read_text()
                if batch==2 and ('OutOfMemoryError' in log or 'CUDA out of memory' in log):
                    # One explicitly recorded batch-size fallback, never a
                    # model/SR failure or unlimited retry of broken code.
                    continue
                raise
            break
        from VLA.openvla.recovery_balanced import SCHEMA
        qualification=output/'own_cuda_reload8_qualification.json'
        for updates in (2,1000):
            if updates>2:execute('warm_refinement1000',command+['--updates','1000','--qualification',qualification])
            cp=fit/'checkpoints'/f'{updates:06d}';precision=output/f'precision{updates}.json'
            execute('saved_precision'+str(updates),[PYTHON,'-m','VLA.openvla.audit_privileged_saved_precision',
                '--checkpoint',cp,'--updates',updates,'--output',precision],True)
            runtime=serve(cp,'server'+str(updates))
            name='native8' if updates==2 else 'development40'
            evaluate=[SIM,'-m','VLA.evaluate_privileged_r33','--model','openvla','--checkpoint',cp,'--port','5831','--output',output/name]
            if updates==2:evaluate+=['--smoke-source',SOURCE]
            else:evaluate+=['--trials-per-action','10']
            execute(name,evaluate);close_server()
            result=json.loads((output/name/'progress.json').read_text())
            if updates==2:
                if result['neural_controls']!=8 or not json.loads((runtime/'adapter_reload.json').read_text())['passed']:
                    raise ValueError('Actual own native8/exact reload required')
                dependencies=[precision,runtime/'adapter_reload.json',output/name/'progress.json',gate,fit/'exact_warmstart.json',cp/'checkpoint_manifest.json']
                bound={**frozen,**{str(p):sha256_file(p) for p in dependencies}}
                atomic_json(qualification,{'passed':True,'schema':SCHEMA,'batch_size':batch,
                    'warmstart_manifest_sha256':sha256_file(WARM/'checkpoint_manifest.json'),'optimizer_updates':2,
                    'actual_neural_robot_controls':8,'dataset_contract_sha256':sha256_file(DATA/'privileged_dataset.json'),
                    'trainer_sha256':sha256_file(ROOT/'VLA/openvla/train_privileged_prompt.py'),
                    'grad_accumulation':4,'learning_rate':.0001,'seed':20261015,'bound_files':bound,
                    'dependencies':[{'path':p,'sha256':digest} for p,digest in bound.items()],
                    'stable_model_SR_established':False})
        if result['completed_trials']!=40:raise ValueError('Keep incomplete diagnostic counts separate')
        from VLA.audit_r33_failure_traces import analyze
        analyze(output/'development40/progress.json',output/'development40/failure_symptoms_r34.json')
        stage='completed_actual_refinement_and40_diagnostics_NOT_final120';save('completed')
    except Exception as error:
        save('failed_preserve_actual_artifacts_NOT_zero_model_SR',f'{type(error).__name__}: {error}');raise
    finally:close_server()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-hours',type=float,default=60)
    parser.add_argument('--launch',action='store_true');a=parser.parse_args();a.output=a.output.resolve()
    if not 0<a.wait_hours<=60:raise ValueError('Finite deadline budget required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('Do not duplicate refinement')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_openvla_balanced_recovery_r34','--output',str(a.output),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'other_VLA_runs_not_interrupted':True});print(child.pid)
