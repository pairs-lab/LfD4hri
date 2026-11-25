"""Honor user-requested render priority without editing frozen VLA runtimes.

Suspend ONLY exact childless VLA waiters, never a trainer/server/episode or
render process. Resume those same process identities automatically after all
workspace cinematic render/encoder jobs are absent for30 seconds.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from VLA.common.artifact_io import atomic_json

ROOT=Path(__file__).resolve().parents[1]
WAITERS=[
 ('VLA.run_all_privileged_r33',ROOT/'VLA/audits/scene67_deadline_r31_20261015/all_family_privileged_r33_v4',{'openvla_precision_2'}),
 ('VLA.run_openpi_eef_repair_r33',ROOT/'VLA/audits/scene67_deadline_r31_20261015/openpi_EEF_repair_r33_v4',{'pi0_fast_actual_EEF_mixed_batches'}),
 ('VLA.run_scene67_resume_cycle',ROOT/'VLA/audits/scene67_deadline_r31_20261015/scene67_resume_r33_v2',{'waiting_existing_OpenPI_cohort_and_render_priority'}),
 ('VLA.run_scene67_finetune_cycle',ROOT/'VLA/audits/scene67_deadline_r31_20261015/automatic_full_gate_fit_v2',{'waiting_complete_physical_extraction'}),
 ('VLA.run_scene64_other_input_gates',ROOT/'VLA/audits/scene67_deadline_r31_20261015/other_native_full_input_gates',{'waiting_complete_physical_and_full_language_gates'}),
 ('VLA.run_scene32_openpi_cycle',ROOT/'VLA/audits/scene67_deadline_r31_20261015/openpi_native32_actual_fit_r32',{'waiting_complete_physical_extraction'}),
 ('VLA.run_other_privileged_serialized',ROOT/'VLA/audits/other_vla_privileged_r29_20261014/gpu_queue_rdt_serialized_20261015',{'waiting_gpu_for_pi0'}),
]


def identity(pid):
    proc=Path('/proc')/str(pid)
    try:
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return {'tokens':(proc/'cmdline').read_bytes().split(b'\0'),'start_ticks':stat[19],
            'state':stat[0],'cwd':str((proc/'cwd').resolve()),'children':(proc/'task'/str(pid)/'children').read_text().strip()}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def render_jobs():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        info=identity(int(proc.name))
        if not info or info['cwd']!=str(ROOT):continue
        tokens=info['tokens'];joined=b' '.join(tokens)
        renderer=any(module in tokens for module in (b'env_eval.oracle_cinematic',b'env_eval.oracle_feedback'))
        # A render parent may prepare the next camera/clip between ffmpeg
        # children. Do not mistake that short gap for completed user work.
        render_parent=any(token.startswith(b'env_eval.demos.') and token.endswith(b'.finish_render') for token in tokens)
        encoder=(b'ffmpeg' in tokens[0] and b'env_eval/demos/' in joined and b'render_jobs' in joined)
        if renderer or render_parent or encoder:found.append(int(proc.name))
    return sorted(found)


def run(output):
    output.mkdir(parents=True,exist_ok=False);paused=[];skipped=[]
    for module,root,stages in WAITERS:
        path=root/'status.json'
        if not path.exists():
            skipped.append({'module':module,'reason':'no registered owner state; no process signalled'});continue
        status=json.loads(path.read_text());pid=status['pid'];info=identity(pid)
        if (not info or status['stage'] not in stages or module.encode() not in info['tokens']
            or str(root).encode() not in info['tokens'] or info['cwd']!=str(ROOT) or info['children']):
            skipped.append({'pid':pid,'reason':'not an exact childless waiting owner; no process signalled'});continue
        if info['state']=='T':raise ValueError('Already suspended by another owner; do not take ownership')
        os.kill(pid,signal.SIGSTOP)
        paused.append({'pid':pid,'module':module,'output':str(root),'start_ticks':info['start_ticks'],'tokens':[x.decode() for x in info['tokens']]})
    quiet_since=None
    def save(stage,renders,resumed=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'stage':stage,'render_pids':renders,'paused_childless_owners':paused,
            'skipped':skipped,'resumed':resumed or [],'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'user_requested_wait_for_render':True,'render_trainer_server_or_episode_signalled':False,
            'frozen_input_or_training_files_changed':False})
    while True:
        renders=render_jobs()
        if renders:quiet_since=None
        elif quiet_since is None:quiet_since=time.monotonic()
        if quiet_since is not None and time.monotonic()-quiet_since>=30:break
        save('waiting_user_render_jobs_no_new_VLA_stage_launches',renders);time.sleep(15)
    resumed=[]
    for row in paused:
        info=identity(row['pid'])
        if (info and info['start_ticks']==row['start_ticks'] and info['tokens']==[x.encode() for x in row['tokens']]
            and info['state']=='T' and info['cwd']==str(ROOT)):
            os.kill(row['pid'],signal.SIGCONT);resumed.append(row['pid'])
    save('completed_render_jobs_then_resumed_same_VLA_waiters',[],resumed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:return run(a.output.resolve())
    marker=a.output.with_name(a.output.name+'_launch.json')
    if a.output.exists() or marker.exists():raise FileExistsError('Never duplicate render-priority waiter')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    python=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
    with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
        child=subprocess.Popen([str(python),'-m','VLA.wait_render_then_resume_queues','--output',str(a.output.resolve())],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(marker,{'pid':child.pid,'only_childless_waiter_suspension_authorized':True});print(child.pid)


if __name__=='__main__':main()
