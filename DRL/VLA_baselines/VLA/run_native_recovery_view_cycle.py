"""Finite fresh native correction38 -> exact replay -> data-only admission.

No GPU, no model success claims and no waived failed old parity receipt.
Respect the user's render-priority request before each new compute stage.
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
from VLA.wait_render_then_resume_queues import render_jobs

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
RDT=ROOT/'VLA/.venvs/rdt/bin/python'
PI=ROOT/'VLA/third_party/openpi/.venv/bin/python'
PARENT=ROOT/'VLA/audits/scene64_recovery_r30_20261014/recovery40'


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'VLA/collect_native_recovery_view.py',ROOT/'VLA/qualify_native_recovery_view.py',
        ROOT/'VLA/collect_demos.py',ROOT/'VLA/simstate_scene64.py',ROOT/'VLA/semantic_touch_teacher.py',
        ROOT/'VLA/common/environment.py',ROOT/'VLA/common/body_gripper.py',ROOT/'VLA/common/physics_safety.py',
        ROOT/'VLA/common/observation_profile.py',ROOT/'asset/params/reward.xml',PARENT/'plan.json',PARENT/'index.json',
        ROOT/'VLA/native_corrective_dataset.py',ROOT/'VLA/audit_native_corrective_batches.py',
        ROOT/'VLA/pi0/scene32_training.py',ROOT/'VLA/pi0/scene32_inputs.py',ROOT/'VLA/rdt/scene67_inputs.py']
    frozen={str(p):sha256_file(p) for p in files};history=[];models={};stage='starting';active=None
    deadline=time.monotonic()+12*3600
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'error':error,
            'history':history,'models':models,'frozen_files':frozen,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'GPU_allocations':0,'model_SR_established':False,'finetune_qualified':False,
            'no_existing_job_signalled':True,'user_render_priority_respected':True})
    def execute(name,command):
        nonlocal stage,active
        stage='waiting_render_priority_before_'+name
        while render_jobs():
            save()
            if time.monotonic()>deadline:raise TimeoutError('Render wait exceeded finite data budget')
            time.sleep(30)
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen native correction pipeline changed')
        stage=name;save()
        with (output/(name+'.log')).open('x') as log:
            active=subprocess.Popen(list(map(str,command)),cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                    'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'stage':name,'pid':active.pid,'command':list(map(str,command))})
            while active.poll() is None:
                save();time.sleep(15)
                if time.monotonic()>deadline:raise TimeoutError('Owned in-flight child preserved; no mid-episode termination')
        history.append({'stage':name,'exit_code':active.returncode})
        if active.returncode:raise RuntimeError(name+' failed; preserve all attempted rows')
    try:
        candidate=output/'fresh38';replay=output/'exact_replay38'
        execute('collect_full38',[PYTHON,'-m','VLA.collect_native_recovery_view','collect','--candidate',PARENT,'--output',candidate])
        proof=json.loads((candidate/'collection_audit.json').read_text())
        if not proof['passed'] or proof['pilot_only'] or len(proof['rows'])!=38:
            raise ValueError('Do not filter failed native corrective attempts')
        execute('exact_replay38',[PYTHON,'-m','VLA.collect_native_recovery_view','replay','--candidate',candidate,'--output',replay])
        execute('data_only_admission',[PYTHON,'-m','VLA.qualify_native_recovery_view','--candidate',candidate,
            '--replay',replay/'audit.json','--output',output/'native_view_admission.json'])
        for model,python in (('rdt',RDT),('pi0',PI),('pi0_fast',PI)):
            try:
                execute(model+'_actual_corrective_batches',[python,'-m','VLA.audit_native_corrective_batches',
                    '--model',model,'--admission',output/'native_view_admission.json','--output',output/(model+'_actual_corrective_batches')])
                models[model]={'actual_corrective_batch_gate_passed':True,'model_SR':None,'long_fit_qualified':False}
            except RuntimeError as error:
                models[model]={'actual_corrective_batch_gate_passed':False,'model_SR':None,
                    'error':str(error),'failed_artifacts_preserved':True}
                save()
        stage='completed_data_only_model_specific_gates_pending';save('completed')
    except Exception as error:
        save('failed_preserve_all_artifacts',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    if not a.launch:run(a.output)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate native data cycle')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(PYTHON),'-m','VLA.run_native_recovery_view_cycle','--output',str(a.output)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'GPU_allocations':0,'no_existing_job_signalled':True});print(child.pid)
