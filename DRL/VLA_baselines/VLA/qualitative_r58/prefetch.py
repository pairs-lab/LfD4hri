"""Resource-admitted second HD worker: distant tail only, atomic publication.

Never edits an active primary render, physics, checkpoint or quantitative job.
If the primary reaches the same item first, keep the staging output rather
than overwrite it. Separate golden directories avoid any concurrent writes.
"""
import os
import argparse
import ctypes
import errno
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
from VLA.common.artifact_io import atomic_json
from .common import OUT,ROOT,SIM,RENDER_PY,read


def publish_without_replace(staging,target):
    # Linux stdio.h: AT_FDCWD=-100, RENAME_NOREPLACE=(1<<0). The atomic
    # no-replace flag also protects an EMPTY directory just created by primary.
    # A check-then-Path.rename could otherwise replace that directory in a race.
    function=ctypes.CDLL(None,use_errno=True).renameat2
    function.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    function.restype=ctypes.c_int
    result=function(-100,os.fsencode(staging),-100,os.fsencode(target),1)
    if result==0:return True
    code=ctypes.get_errno()
    if code==errno.EEXIST:return False
    raise OSError(code,os.strerror(code),str(target))


def main(a):
    done=read(OUT/'capture_done.json')
    if not done['all11_main_CPU_selected']:raise ValueError('Only enable second HD worker after CPU qualification is complete')
    directory=a.output;directory.mkdir(exist_ok=False)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    child=None;records=[]
    def status(state='running',**kw):
        atomic_json(directory/'status.json',{'pid':os.getpid(),'status':state,'child':child.pid if child else None,
            'records':records,'updated_UTC':datetime.now(timezone.utc).isoformat(),**kw})
    def execute(interpreter,module,args,label):
        nonlocal child
        until=time.monotonic()+20*3600
        while True:
            ram=next(int(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
            free=float(gpu.stdout.splitlines()[0]) if gpu.returncode==0 else 0
            # Conservative admission. A running context may temporarily take
            # availability below6GB, but at most two HD jobs exist.
            if ram>=a.min_ram and free>=a.min_vram:break
            if time.monotonic()>until:raise TimeoutError('No safe secondary render admission')
            status(wait='Measured-cost RAM/VRAM admission',RAM_available_MiB=ram,GPU_free_MiB=free,
                required_RAM_MiB=a.min_ram,required_VRAM_MiB=a.min_vram);time.sleep(10)
        with (directory/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(interpreter),'-u','-m',module,*map(str,args)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);status(active=label)
            while child.poll() is None:time.sleep(2);status(active=label)
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label} exited{code}; staging preserved')
    candidates=[r for r in read(OUT/'assignments.json') if r['backup'] or r['scene_id']==11]
    for item in candidates:
        prefix='backup' if item['backup'] else 'scene';label=f'{prefix}{item["scene_id"]:02d}_seed{item["seed"]}'
        for method,key in (('VLA','vla'),('Oracle','oracle')):
            name=label+'_'+method;target=OUT/'_renders'/name
            row={'item':name,'status':'checking'};records.append(row);status()
            if target.exists():row['status']='primary_already_owns_output_skipped';status();continue
            try:
                cap=Path(item[key]);states=read(cap/'recorded_physics_states.json');evidence=read(cap/'content_evidence.json');n=len(states)
                event=n//2
                if evidence['full_put_complete']:event=max(0,evidence['stable_completion_frame']-15)
                elif evidence['events']:event=max(evidence['events'],key=lambda e:e.get('object_displacement_within250ms_m',0))['event_frame']
                indices=sorted(set((0,min(n-1,event),n-1)))
                golden=OUT/'audit/prefetch_goldens'/name;staging=OUT/'_pre_renders'/name
                if not (golden/'probe.json').exists():
                    execute(SIM,'VLA.qualitative_r58.render_probe',['--capture',cap,'--output',golden,'--frames',*indices],name+'_golden')
                if target.exists():row['status']='primary_arrived_before_secondary_HD_skipped';status();continue
                if not (staging/'render_complete.json').exists():
                    if staging.exists():raise ValueError('Incomplete previous staging kept; primary may render separately')
                    execute(RENDER_PY,'VLA.qualitative_r58.render',['--capture',cap,'--output',staging,'--golden',golden],name+'_HD')
                if not (staging/'render_complete.json').exists():raise ValueError('Incomplete staging may not be published')
                target.parent.mkdir(exist_ok=True)
                if publish_without_replace(staging,target):row['status']='golden_checked_complete_render_atomically_published'
                else:row['status']='primary_arrived_staging_preserved_NOT_overwritten'
                status()
            except Exception as error:row.update(status='secondary_error_preserved_primary_still_authoritative',error=f'{type(error).__name__}: {error}');status()
    status('secondary_complete_no_automatic_visual_approval')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT/'prefetch_v2')
    parser.add_argument('--min-ram',type=float,default=6000)
    parser.add_argument('--min-vram',type=float,default=6000)
    main(parser.parse_args())
