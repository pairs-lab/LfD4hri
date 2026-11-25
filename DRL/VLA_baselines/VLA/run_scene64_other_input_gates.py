"""Automatic CPU-only native input gates for OTHER VLA families.

Waits for complete physical extraction/full-language audit. Never races a
GPU trainer, never grants long-fit permission from a pilot receipt. Fails
closed and retains every failed native loader artifact.
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

ROOT=Path(__file__).resolve().parents[1]
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PI=ROOT/'VLA/third_party/openpi/.venv/bin/python'
OV=ROOT/'VLA/.venvs/openvla/bin/python'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
LANGUAGE=ROOT/'VLA/audits/scene67_deadline_r31_20261015/automatic_full_gate_fit_v2/full_native_language/audit.json'


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'VLA/pi0/scene64_training.py',ROOT/'VLA/openvla/prepare_scene64_source.py',
        ROOT/'VLA/openvla/audit_scene64_loader.py',ROOT/'VLA/openvla/train_scene64_prompt.py',
        ROOT/'VLA/scene64_conditioning.py',ROOT/'VLA/scene64_dataset.py',ROOT/'VLA/scene64_schema.py']
    frozen={str(p):sha256_file(p) for p in files};history=[];stage='waiting_complete_physical_and_full_language_gates'
    until=time.monotonic()+a.wait_hours*3600
    env={**os.environ,'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENPI_DATA_HOME':str(ROOT/'VLA/checkpoints/openpi')}
    def save(status,error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'history':history,'error':error,
            'frozen_files':frozen,'GPU_allocations':0,'neural_controls':0,'all_family_finetuning_completed':False,
            'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    def check():
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Own native inputs changed')
    def execute(name,argv,proof):
        nonlocal stage
        stage=name;check();save('running')
        with (output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(list(map(str,argv)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'pid':child.pid,'stage':name,'GPU_stage':False})
            while child.poll() is None:save('running');time.sleep(15)
        history.append({'stage':name,'exit_code':child.returncode})
        if child.returncode:raise RuntimeError(name+' failed; artifacts retained')
        if proof and not json.loads(proof.read_text()).get('passed'):raise ValueError(name+' receipt failed')
    try:
        while not LANGUAGE.exists():
            check();save('running')
            parent=ROOT/'VLA/audits/scene67_deadline_r31_20261015/automatic_full_gate_fit_v2/status.json'
            if parent.exists() and json.loads(parent.read_text())['status'].startswith('failed'):raise RuntimeError('Upstream full-data gate failed')
            if time.monotonic()>until:raise TimeoutError('Finite CPU data-gate wait; no existing job signalled')
            time.sleep(30)
        for model in ('pi0','pi0_fast'):
            folder=output/(model+'_actual_native_batches')
            execute(model+'_actual_native_batches',[PI,'-m','VLA.pi0.scene64_training','audit','--model',model,
                '--projection',PROJECTION,'--output',folder],folder/'native_batches.json')
        dataset=output/'openvla_scene64_source'
        execute('openvla_prepare_immutable_source',[SIM,'-m','VLA.openvla.prepare_scene64_source','--projection',PROJECTION,
            '--native-preflight',LANGUAGE,'--output',dataset],dataset/'pilot_preconditions.json')
        batch=output/'openvla_actual_native_batches.json'
        execute('openvla_actual_native_batches',[OV,'-m','VLA.openvla.audit_scene64_loader','--dataset',dataset,'--output',batch],batch)
        stage='completed_native_CPU_gates_NOT_CUDA_or_SR_qualification';save('completed')
    except Exception as error:save('failed_preserve_all_artifacts',f'{type(error).__name__}: {error}');raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=48);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=60:raise ValueError('Finite deadline required')
    if not a.launch:return run(a)
    marker=a.output.with_name(a.output.name+'_launch.json')
    if marker.exists() or a.output.exists():raise FileExistsError('No duplicate CPU gate cycle')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
        child=subprocess.Popen([str(SIM),'-m','VLA.run_scene64_other_input_gates','--output',str(a.output.resolve()),
            '--wait-hours',str(a.wait_hours)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(marker,{'pid':child.pid,'GPU_allocations':0,'existing_jobs_signalled':False});print(child.pid)


if __name__=='__main__':main()
