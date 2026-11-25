"""Automatic CPU-only FAST codec1479 -> offline physical4 quality check."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python';PI=ROOT/'VLA/third_party/openpi/.venv/bin/python'


def run(output):
    output.mkdir(parents=True,exist_ok=False);stage='actual_codec_all1479';history=[]
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','HF_HUB_OFFLINE':'1'}
    try:
        for name,command in [('actual_codec_all1479',[PI,'-m','VLA.audit_openpi_eef_codec','numerical',
            '--projection',ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64',
            '--admission',BASE/'all_family_privileged_r33_v4/native_view_admission.json','--output',output/'numerical']),
            ('physical_offline4',[SIM,'-m','VLA.audit_openpi_eef_codec','physical','--codec',output/'numerical','--output',output/'physical4'])]:
            stage=name
            with (output/(name+'.log')).open('x') as log:
                child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(output/'active_child.json',{'pid':child.pid,'stage':name})
                deadline=time.monotonic()+4*3600
                while child.poll() is None:
                    atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','model_SR_established':False})
                    time.sleep(15)
                    if time.monotonic()>deadline:raise TimeoutError('Finite CPU quality budget; in-flight child preserved')
            history.append({'stage':name,'exit_code':child.returncode})
            if child.returncode:raise RuntimeError('Actual codec/source quality stage failed')
        physical=json.loads((output/'physical4/audit.json').read_text())
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':'completed_offline_codec_quality_NOT_VLA_SR','status':'completed',
            'offline_physical_passed':physical['passed'],'history':history,'model_SR_established':False})
    except Exception as error:
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed_preserve_artifacts','error':str(error),
            'history':history,'model_SR_established':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    if not a.launch:run(a.output)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate codec quality cycle')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_fast_codec_preflight','--output',str(a.output)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'GPU_allocations':0});print(child.pid)
