"""Reload presentation code ONLY at a read-only audit boundary of our batch."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
from VLA.common.artifact_io import atomic_json
from .qualify import OUT,ROOT,ROBOTS


def identity(pid):
    try:
        path=Path('/proc')/str(pid)
        return ((path/'stat').read_text().rsplit(')',1)[1].split()[19],(path/'cmdline').read_bytes())
    except FileNotFoundError:return None


def main(pid):
    captured=identity(pid)
    if not captured or b'env_eval.iccas_demo.batch\x00' not in captured[1]:raise ValueError('Not the exact owned demo batch')
    limit=time.monotonic()+1800
    while identity(pid)==captured:
        if time.monotonic()>limit:raise TimeoutError('No safe readonly audit boundary; healthy render left untouched')
        progress=json.loads((OUT/'PROGRESS.json').read_text())
        folder=OUT/ROBOTS[progress['robot']]/f"scene{progress['scene']:02d}"
        if not (folder/'RENDER.json').exists():time.sleep(3);continue
        os.kill(pid,signal.SIGSTOP)
        safe=False
        try:
            for _ in range(100):
                if (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]=='T':break
                time.sleep(.01)
            progress=json.loads((OUT/'PROGRESS.json').read_text())
            folder=OUT/ROBOTS[progress['robot']]/f"scene{progress['scene']:02d}"
            if not (folder/'RENDER.json').exists():continue
            # Every MP4 writer closes before RENDER.json is committed. Still
            # reject the boundary if ANY of this process's children writes mp4.
            children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
            commands=[(Path('/proc')/child/'cmdline').read_bytes() for child in children if (Path('/proc')/child/'cmdline').exists()]
            if any(b'.mp4\x00' in c and b'rawvideo' in c for c in commands):continue
            if identity(pid)!=captured:raise ValueError('PID identity changed before handoff')
            atomic_json(OUT/'PALETTE_HANDOFF.json',{'old_pid':pid,'safe_readonly_boundary':str(folder),
                'all_video_writers_already_closed':True,'no_training_or_physical_rollout_interrupted':True,
                'reason':'refresh requested procedural prop colors for EGL visual-shape-ID-zero bug'})
            os.kill(pid,signal.SIGINT);safe=True
        finally:
            if identity(pid)==captured:os.kill(pid,signal.SIGCONT)
        if safe:break
        time.sleep(3)
    while identity(pid)==captured:
        state=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
        if state=='Z':break
        time.sleep(.2)
    time.sleep(.5)
    run=subprocess.run([sys.executable,'-m','env_eval.iccas_demo.batch','--launch'],cwd=ROOT,check=True,capture_output=True,text=True)
    (OUT/'PALETTE_HANDOFF_RESTART.txt').write_text(run.stdout)
    print(run.stdout,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);main(p.parse_args().pid)
