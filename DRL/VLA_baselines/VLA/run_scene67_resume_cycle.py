"""Automatically finish the fitted scene67 cohort after a writer-only bug.

Wait for the already registered OpenPI cohort, then preserve ALL terminal
episodes and run only missing tasks. No training rerun or success selection.
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

ROOT=Path(__file__).resolve().parents[1];PYTHON=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
OLD=ROOT/'VLA/audits/scene67_deadline_r31_20261015/automatic_full_gate_fit_v2'
PI=ROOT/'VLA/audits/scene67_deadline_r31_20261015/openpi_native32_actual_fit_r32'


def owner_alive(status):
    proc=Path('/proc')/str(status.get('pid',0))
    try:return (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (FileNotFoundError,IndexError):return False


def run(output):
    output.mkdir(parents=True,exist_ok=False);stage='waiting_existing_OpenPI_cohort_and_render_priority';active=None
    files=[Path(__file__),ROOT/'VLA/evaluate_scene67_smolvla.py',ROOT/'VLA/resume_scene67_evidence.py',
        ROOT/'VLA/smolvla/scene67_adapter.py',ROOT/'VLA/smolvla/adapter.py',ROOT/'VLA/common/evaluation_statistics.py']
    frozen={str(p):sha256_file(p) for p in files};deadline=time.monotonic()+42*3600
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'error':error,
            'frozen_files':frozen,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'old_completed_controls_rerolled':False,'scorer_and_safety_unchanged':True,
            'learned_high_level_action_source_only':True})
    try:
        while True:
            pi=json.loads((PI/'status.json').read_text());gpu=gpu_state()
            boundary_ready=True
            if a_boundary is not None:
                other=json.loads((a_boundary/'status.json').read_text())
                boundary_ready=not owner_alive(other) or other.get('stage')=='waiting_registered_predecessors_and_exclusive_GPU'
            if boundary_ready and not owner_alive(pi) and not gpu['compute_pids'] and gpu['free_mib']>=14500 and not render_jobs():break
            save()
            if time.monotonic()>deadline:raise TimeoutError('Finite GPU wait; existing jobs preserved')
            time.sleep(30)
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen resumed evaluator changed')
        command=json.loads((OLD/'active_child.json').read_text())['command'];checkpoint=command[command.index('--checkpoint')+1]
        stage='resume_only_missing_development120_episodes';save()
        args=[str(PYTHON),'-m','VLA.evaluate_scene67_smolvla','--checkpoint',checkpoint,'--output',str(output/'development120'),
            '--resume-from',str(OLD/'development120')]
        with (output/'resume.log').open('x') as log:
            active=subprocess.Popen(args,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'stage':stage,'pid':active.pid,'command':args})
            while active.poll() is None:
                save();time.sleep(15)
                if time.monotonic()>deadline:raise TimeoutError('Owned in-flight episode retained')
        if active.returncode:raise RuntimeError('Resumed learned evaluation failed; preserve all actual episodes')
        result=json.loads((output/'development120/progress.json').read_text())
        if result['completed_trials']!=120:raise ValueError('Incomplete cohort')
        stage='completed_same_checkpoint_development120';save('completed')
    except Exception as error:save('failed_preserve_all_artifacts',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--yield-until-cycle-waiting',type=Path)
    p.add_argument('--launch',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    a_boundary=a.yield_until_cycle_waiting.resolve() if a.yield_until_cycle_waiting else None
    if not a.launch:run(a.output)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate continuation')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            command=[str(PYTHON),'-m','VLA.run_scene67_resume_cycle','--output',str(a.output)]
            if a_boundary is not None:command+=['--yield-until-cycle-waiting',str(a_boundary)]
            child=subprocess.Popen(command,
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'no_existing_job_interrupted':True});print(child.pid)
