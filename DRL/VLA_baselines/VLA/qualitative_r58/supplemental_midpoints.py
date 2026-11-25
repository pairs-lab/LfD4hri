"""Fill degenerate reset/event/terminal goldens without changing any video.

When an Oracle failure event IS the terminal frame, the original set has two
distinct states, not three. Independently check the midpoint with pinned and
NumPy render bindings, then compare that state to the EXISTING encoded video.
"""
import os
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import decode_at
from .common import OUT,ROOT,SIM,RENDER_PY,read


def main():
    directory=OUT/'supplemental_audit_v1';directory.mkdir(exist_ok=False)
    results=[];child=None
    def status(state='running',**extra):
        atomic_json(directory/'status.json',{'pid':os.getpid(),'child':child.pid if child else None,
            'status':state,'results':results,'no_video_actions_or_outcomes_changed':True,**extra})
    while not (OUT/'production_done.json').exists():
        status(wait='Full primary media batch before extra GPU probes');time.sleep(10)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def execute(interpreter,opts,label):
        nonlocal child
        while next(float(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))<3000:
            status(wait='RAM admission for one posthoc rendering probe',active=label);time.sleep(10)
        with (directory/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(interpreter),'-u','-m','VLA.qualitative_r58.render_probe',*map(str,opts)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status(active=label)
            while child.poll() is None:time.sleep(1)
            if child.returncode:raise RuntimeError('Supplemental probe failed: '+label)
        child=None
    try:
        for item in read(OUT/'produced_pairs.json'):
            pair=Path(item['paired_output'])
            for method,key,suffix in (('Scripted_Oracle','oracle','Oracle'),('XVLA_terminal_release','vla','VLA')):
                renderer=read(pair/method/'renderer.json');matches=renderer.get('per_rollout_golden_RGB_and_segmentation_matches',[])
                if not renderer.get('NumPy_transfer_enabled') or len(matches)==21:continue
                if len(matches)!=14 or len({r['frame'] for r in matches})!=2:
                    raise ValueError('Unexpected golden coverage; inspect, do not relax gate')
                prefix='backup' if item['backup'] else 'scene';label=f'{prefix}{item["scene_id"]:02d}_seed{item["seed"]}_{suffix}'
                cap=Path(item[key]);frames=read(pair/method/'frames.json');mid=len(frames)//2
                base=OUT/'audit/supplemental_midpoints'/label;golden=base/'pinned';fast=base/'numpy'
                execute(SIM,['--capture',cap,'--output',golden,'--frames',mid],label+'_pinned')
                execute(RENDER_PY,['--capture',cap,'--output',fast,'--frames',mid,'--reference',golden],label+'_numpy')
                expected={(r['frame'],r['role']):r for r in read(golden/'probe.json')['rows']}
                checks=[]
                for row in read(fast/'probe.json')['rows']:
                    original=expected[(row['frame'],row['role'])]
                    if not row['RGB_exact'] or row['segmentation_sha256']!=original['segmentation_sha256']:
                        raise ValueError('Supplemental RGB/segmentation is not exact')
                    if max(row['geometry_errors'].values())>1e-12:raise ValueError('Supplemental restored geometry differs')
                    raw=OUT/'_renders'/label/(row['role']+'.mp4')
                    encoded=decode_at(raw,mid);lossless=cv2.imread(str(fast/f'frame{mid:05d}_{row["role"]}.png'))
                    error=float(np.abs(encoded.astype(np.int16)-lossless.astype(np.int16)).mean())
                    if error>=6.:raise ValueError('Existing encoded midpoint does not match restored geometry/camera')
                    checks.append({'frame':mid,'role':row['role'],'RGB_exact':True,'segmentation_exact':True,
                        'existing_encoded_RGB_MAE':error,'existing_video_sha256':sha256_file(raw)})
                if len(checks)!=7:raise ValueError('All7 cameras required at the distinct supplemental state')
                results.append({'label':label,'passed':True,'original_distinct_golden_states':2,
                    'original_renderer_json_sha256':sha256_file(pair/method/'renderer.json'),
                    'source_state_sha256':sha256_file(cap/'recorded_physics_states.json'),
                    'source_frames_after_supplement':sorted({r['frame'] for r in matches}|{mid}),
                    'supplemental_checks':checks,'original_renderer_record_NOT_rewritten':True})
                atomic_json(OUT/'audit/supplemental_midpoint_checks.json',{'passed':True,'results':results})
                status()
        atomic_json(OUT/'audit/supplemental_midpoint_checks.json',{'passed':True,'results':results,
            'no_original_video_outcome_or_renderer_record_modified':True})
        status('complete');print({'supplemented_three_distinct_state_rollouts':len(results),'all_checks_passed':True},flush=True)
    except Exception as error:
        status('failed_preserved',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
