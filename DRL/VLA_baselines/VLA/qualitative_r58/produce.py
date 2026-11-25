"""One HD worker consumes real paired captures, then packages all7streams."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,ROOT,SIM,RENDER_PY,read


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);results=[];child=None
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'pid':os.getpid(),'child':child.pid if child else None,
            'results':results,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})
    def execute(interpreter,module,opts,label,render=False):
        nonlocal child
        while True:
            available=next(int(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
            free=float(gpu.stdout.splitlines()[0]) if gpu.returncode==0 else 0
            if available>=3000 and (not render or free>=1200):break
            status(wait='RAM/VRAM admission; training not preempted',active=label);time.sleep(10)
        log=a.output/(label+'.log');attempt=1
        while log.exists():attempt+=1;log=a.output/(label+f'_attempt{attempt}.log')
        atomic_json(log.with_suffix('.source.json'),{'entrypoint':module,'entrypoint_sha256':sha256_file(ROOT/(module.replace('.','/')+'.py')),
            'interpreter':str(interpreter),'options':list(map(str,opts))})
        with log.open('x') as stream:
            child=subprocess.Popen([str(interpreter),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);status(active=label)
            while child.poll() is None:time.sleep(2);status(active=label)
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label} exited{code}; partial output retained')
    def render(cap,label):
        path=OUT/'_renders'/label
        if (path/'render_complete.json').exists():return path
        if path.exists():raise ValueError('Preserve partial render; explicit revision required: '+str(path))
        if (cap/'renderer.json').exists() and all((cap/f'wide_{i}.mp4').exists() for i in range(1,7)) and (cap/'media_status.json').exists():
            # The reference pilot used the pinned binary and passed full decode.
            # Reuse physical images, not Oracle images or a different episode.
            path.mkdir(parents=True)
            for name in ('handeye.mp4',*[f'wide_{i}.mp4' for i in range(1,7)],'frames.json',
                    'initial_scene.json','content_evidence.json','renderer.json','replay_proof.json','protocol.json','outcome.json'):
                if (cap/name).exists():os.link(cap/name,path/name)
            atomic_json(path/'render_complete.json',{'reused_pinned_reference_renderer':True,
                'source_capture':str(cap),'source_states_sha256':sha256_file(cap/'recorded_physics_states.json'),
                'all7reference_streams_full_decode_sync_pass':True});return path
        states=read(cap/'recorded_physics_states.json');evidence=read(cap/'content_evidence.json');n=len(states)
        event=n//2
        if evidence['full_put_complete']:event=max(0,evidence['stable_completion_frame']-15)
        elif evidence['events']:
            event=max(evidence['events'],key=lambda e:e.get('object_displacement_within250ms_m',0))['event_frame']
        indices=sorted(set((0,min(n-1,event),n-1)))
        golden=OUT/'audit/goldens'/label
        if not (golden/'probe.json').exists():
            execute(SIM,'VLA.qualitative_r58.render_probe',['--capture',cap,'--output',golden,'--frames',*indices],label+'_pinned_golden',True)
        execute(RENDER_PY,'VLA.qualitative_r58.render',['--capture',cap,'--output',path,'--golden',golden],label+'_HD',True)
        if not (path/'render_complete.json').exists():raise ValueError('Missing closed-writer completion proof')
        return path
    deadline=time.monotonic()+30*3600;done=set()
    try:
        while True:
            assignments=read(OUT/'assignments.json') if (OUT/'assignments.json').exists() else []
            todo=[r for r in assignments if (r['backup'],r['scene_id']) not in done]
            if not todo:
                if (OUT/'capture_done.json').exists():break
                if time.monotonic()>deadline:raise TimeoutError('Finite producer wait exceeded')
                status(wait='Next actual CPU-qualified pair');time.sleep(5);continue
            item=todo[0];key=(item['backup'],item['scene_id']);prefix='backup' if item['backup'] else 'scene'
            label=f'{prefix}{item["scene_id"]:02d}_seed{item["seed"]}'
            row={**item,'status':'rendering'};results.append(row);status(active=label)
            try:
                vr=render(Path(item['vla']),label+'_VLA');orr=render(Path(item['oracle']),label+'_Oracle')
                target=OUT/item['group']/label
                if not (target/'PAIR_AUDIT.json').exists():
                    opts=['--vla',vr,'--oracle',orr,'--repeat-proof',item['repeat_proof'],'--output',target,
                        '--scene-id',item['scene_id'],'--source-pair',label]
                    if item['backup']:opts+=['--backup']
                    execute(SIM,'VLA.qualitative_r58.package',opts,label+'_package')
                row.update(status='media_all7streams_audited_NEEDS_actual_visual_review',paired_output=str(target))
            except Exception as error:row.update(status='production_error_preserved_REPAIR_REQUIRED',error=f'{type(error).__name__}: {error}')
            done.add(key);atomic_json(OUT/'produced_pairs.json',results);status()
        errors=[r for r in results if 'error' in r]
        status('media_batch_complete_visual_review_REQUIRED' if not errors else 'media_errors_REPAIR_REQUIRED',errors=len(errors))
        atomic_json(OUT/'production_done.json',{'media_pairs':len(results)-len(errors),'errors':errors,
            'all7streams_per_rollout':True,'manual_final_review_NOT_automatically_signed':True})
    except Exception as error:status('failed_preserved',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT/'produce_v1')
    main(parser.parse_args())
