"""Resume ONLY a verified, idle queued Intern owner; no model jobs are stopped.

Independent tiny watchdog survives the FLOWER owner's unexpected termination.
Target process start time protects against PID reuse.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from VLA.common.artifact_io import atomic_json


def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':stat[0],'start_ticks':stat[19]}
    except FileNotFoundError:return None


def resume(record,output):
    live=identity(record['pid'])
    same=live is not None and live['start_ticks']==record['start_ticks']
    resumed=bool(same and live['state'] in ('T','t'))
    if resumed:os.kill(record['pid'],signal.SIGCONT)
    atomic_json(output,{'verified_same_original_queued_process':same,'SIGCONT_sent':resumed,
                        'pid':record['pid'],'no_training_job_was_stopped':True})


def main(a):
    record=json.loads(a.record.read_text());end=time.monotonic()+4*3600
    while time.monotonic()<end:
        owner=identity(a.owner)
        if owner is None or owner['state']=='Z':break
        try:
            state=json.loads(a.status.read_text())
            if state['status'] in ('completed','integration_error_NOT_model_SR') or state.get('GPU_work_finished'):break
        except FileNotFoundError:pass
        time.sleep(5)
    resume(record,a.output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True)
    p.add_argument('--owner',type=int,required=True);p.add_argument('--status',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
