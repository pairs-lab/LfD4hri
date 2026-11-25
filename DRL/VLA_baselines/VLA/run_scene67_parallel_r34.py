"""Resume fitted Smol67 under guarded OpenVLA-INFERENCE concurrency.

No parallel fitting. Separate CUDA contexts/private generators and DIRECT
physics clients; the same five old terminal episodes are retained. An OOM
can fall back once to exclusive inference only if no partial task is rerolled.
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
from VLA.wait_render_then_resume_queues import render_jobs,identity

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
OPENVLA=BASE/'all_family_privileged_r33_v4'
OLD=BASE/'automatic_full_gate_fit_v2/development120'


def resources():
    values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    return {'available_RAM_MiB':values['MemAvailable']/1024,**gpu_state()}


def allowed_parallel(state,gpu,server,child):
    server_info=identity(server.get('pid',0));child_info=identity(child.get('pid',0))
    return (state.get('stage')=='openvla_development120' and state.get('status')=='running'
        and server_info is not None and child_info is not None
        and b'VLA.openvla.scene64_server' in server_info['tokens']
        and b'VLA.evaluate_privileged_r33' in child_info['tokens']
        and server_info['cwd']==str(ROOT) and child_info['cwd']==str(ROOT)
        and str(OPENVLA).encode() in b' '.join(server_info['tokens'])
        and {int(pid) for pid in gpu['compute_pids']}=={int(server['pid'])}
        and gpu['free_mib']>=4608 and gpu['available_RAM_MiB']>=5120)


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);active=None;stage='waiting_guarded_inference_capacity'
    old=json.loads((OLD.parent/'active_child.json').read_text());command=old['command']
    checkpoint=Path(command[command.index('--checkpoint')+1])
    files=[Path(__file__),ROOT/'VLA/smolvla/evaluate_capped_scene67.py',ROOT/'VLA/evaluate_scene67_smolvla.py',
        ROOT/'VLA/resume_scene67_evidence.py',ROOT/'VLA/smolvla/scene67_adapter.py',ROOT/'VLA/smolvla/adapter.py',
        ROOT/'VLA/common/evaluation_statistics.py']
    frozen={str(p):sha256_file(p) for p in files};deadline=time.monotonic()+42*3600;attempt=None;history=[]
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'status':status,'error':error,'history':history,
            'active_attempt':str(attempt) if attempt else None,'frozen_files':frozen,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint':str(checkpoint),'parallel_inference_only_no_parallel_fit':True,'old_episodes_rerolled':False,
            'scorer_robot_policy_and_noise_unchanged':True})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite inference budget; in-flight child preserved')
        if any(sha256_file(p)!=digest for p,digest in frozen.items()):raise ValueError('Frozen capped evaluator changed')
    try:
        concurrent=False
        while True:
            check();gpu=resources();state=json.loads((OPENVLA/'status.json').read_text())
            server=json.loads((OPENVLA/'owned_server.json').read_text())
            child=json.loads((OPENVLA/'active_child.json').read_text())
            if not render_jobs() and allowed_parallel(state,gpu,server,child):concurrent=True;break
            if not render_jobs() and not gpu['compute_pids'] and gpu['free_mib']>=14500 and gpu['available_RAM_MiB']>=5120:break
            save();time.sleep(30)
        atomic_json(output/'launch_resource_guard.json',{'concurrent_with_exact_OpenVLA_inference':concurrent,
            'actual_resources':gpu,'OpenVLA_owner_stage':state['stage'],'OpenVLA_server':server,'OpenVLA_evaluator':child,
            'Smol_allocator_cap_MiB':3072 if concurrent else 5120,
            'new_simulation_or_high_level_action_policy':False})
        previous=OLD
        for number in range(2):
            if number:
                stage='waiting_exclusive_after_recorded_parallel_OOM';save()
                while True:
                    check();gpu=resources()
                    if not render_jobs() and not gpu['compute_pids'] and gpu['free_mib']>=14500 and gpu['available_RAM_MiB']>=5120:break
                    save();time.sleep(30)
            attempt=output/('parallel_attempt' if concurrent and not number else 'exclusive_attempt');attempt.mkdir()
            cap=3072 if concurrent and not number else 5120
            stage='actual_memory_capped_same_checkpoint_development120';save()
            command=[SIM,'-m','VLA.smolvla.evaluate_capped_scene67','--gpu-memory-cap-mib',cap,'--resource-audit-root',attempt,
                '--checkpoint',checkpoint,'--output',attempt/'development120','--resume-from',previous]
            with (attempt/'evaluate.log').open('x') as log:
                active=subprocess.Popen(list(map(str,command)),cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                child_record={'pid':active.pid,'command':list(map(str,command))}
                atomic_json(output/'active_child.json',child_record);atomic_json(attempt/'active_child.json',child_record)
                while active.poll() is None:save();time.sleep(15);check()
            history.append({'attempt':str(attempt),'exit_code':active.returncode,'allocator_cap_mib':cap})
            if not active.returncode:
                result=json.loads((attempt/'development120/progress.json').read_text())
                if result['completed_trials']!=120:raise ValueError('Incomplete actual episode cohort')
                stage='completed_same_checkpoint_preserved120';save('completed');return
            log=(attempt/'evaluate.log').read_text()
            journal=json.loads((attempt/'applied_step_resource_journal.json').read_text()) if (attempt/'applied_step_resource_journal.json').exists() else {}
            no_partial=not journal.get('applied_steps_current_task',0) or journal.get('terminal_current_task',False)
            if number or not concurrent or not no_partial or not ('OutOfMemoryError' in log or 'CUDA out of memory' in log):
                raise RuntimeError('Actual capped evaluator failed; no hidden partial-task reroll')
            previous=attempt/'development120'
        raise RuntimeError('Finite memory fallback exhausted')
    except Exception as error:save('failed_preserve_actual_neural_evidence_NOT_zero_SR',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--launch',action='store_true');a=parser.parse_args();a.output=a.output.resolve()
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate fitted-model evaluation')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_scene67_parallel_r34','--output',str(a.output)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'checkpoint_scoring_and_completed_episodes_unchanged':True});print(child.pid)
