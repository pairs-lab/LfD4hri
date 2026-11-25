"""Hold an inference-stage queue controller, never a training worker.

Its current evaluation child finishes normally. An independent finite watchdog
resumes the exact controller after FLOWER800, failure, or owner disappearance.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from VLA.common.artifact_io import atomic_json
from VLA.flower_rotation_r76.queue_guard import identity

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VLA/audits/flower800_r77_20261019'
INTERN=ROOT/'VLA/audits/internvla_native_r72_20261019/balanced_v2/owner_status.json'


def resume(record,path,reason):
    live=identity(record['pid']);same=live is not None and live['start_ticks']==record['start_ticks']
    sent=bool(same and live['state'] in ('T','t'))
    if sent:os.kill(record['pid'],signal.SIGCONT)
    atomic_json(path,{'same_original_process':same,'SIGCONT_sent':sent,'reason':reason,'pid':record['pid']})


def hold(out):
    out.mkdir(parents=True,exist_ok=True);record=out/'INTERN_QUEUE_HOLD.json'
    if record.exists():raise FileExistsError('Do not hold a queue twice')
    status=json.loads(INTERN.read_text());pid=int(status['pid']);live=identity(pid)
    child=status.get('active_child_pid');server=status.get('active_server_pid')
    if status['status']!='running' or not child or not server or live is None:
        raise ValueError('No verified inference-stage controller; do not pause any model job')
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    child_command=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    server_command=Path(f'/proc/{server}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if ('VLA.intern_r72.watch' not in command or 'balanced_v2' not in command or
        'VLA.intern_r72.evaluate' not in child_command or 'VLA.intern_r72.server' not in server_command):
        raise ValueError('Only pause the controller while a genuine evaluation, not training, is active')
    data={'pid':pid,'start_ticks':live['start_ticks'],'prior_status':status,
        'current_evaluation_child_finishes_normally':True,'training_job_stopped':False,
        'reason':'User requests frozen FLOWER800 immediately after qualification; prevent concurrent Intern extension training'}
    atomic_json(record,data)
    with (out/'queue_watchdog.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','VLA.flower800_r77.queue_guard',
            '--output',str(out),'--watch'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    os.kill(pid,signal.SIGSTOP)
    atomic_json(out/'QUEUE_WATCHDOG.json',{'pid':process.pid,'max_hold_hours':16,'owner_start_grace_seconds':600})
    return data


def watch(out):
    record=json.loads((out/'INTERN_QUEUE_HOLD.json').read_text());started=time.monotonic();reason='finite16h_hold_expired'
    while time.monotonic()-started<16*3600:
        p=out/'owner_status.json'
        if p.exists():
            status=json.loads(p.read_text())
            if status['status'] in ('completed','integration_error_NOT_model_SR'):
                reason='FLOWER800_owner_terminal';break
            owner=identity(status['pid'])
            if owner is None or owner['state']=='Z':reason='FLOWER800_owner_exited';break
        elif time.monotonic()-started>600:reason='FLOWER800_owner_not_started_within10min';break
        time.sleep(5)
    resume(record,out/'INTERN_WATCHDOG_RESUME.json',reason)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:watch(a.output)
    else:print(hold(a.output))
