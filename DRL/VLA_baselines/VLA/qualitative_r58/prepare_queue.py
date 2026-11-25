"""Prepare actual encoded review sheets automatically; NEVER approve them."""
import os
import argparse
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from .common import OUT,ROOT,SIM,read


def main(directory):
    directory.mkdir(exist_ok=False)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    done=set();rows=[];child=None
    def status(state='running',**extra):
        atomic_json(directory/'status.json',{'status':state,'pid':os.getpid(),
            'child':child.pid if child else None,'records':rows,
            'automatic_visual_approval_FORBIDDEN':True,**extra})
    while True:
        items=read(OUT/'produced_pairs.json') if (OUT/'produced_pairs.json').exists() else []
        todo=[r for r in items if r.get('paired_output') and r['paired_output'] not in done]
        if not todo:
            if (OUT/'production_done.json').exists():break
            status(wait='Next fully encoded/audited pair');time.sleep(10);continue
        item=todo[0];folder=Path(item['paired_output']);record={'pair':str(folder),'status':'preparing'}
        rows.append(record);status()
        try:
            for module,target in (('VLA.qualitative_r58.review',folder/'review/sequence_manifest.json'),
                                  ('VLA.qualitative_r58.explain',folder/'EXPLANATION.md')):
                if target.exists():continue
                while next(int(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))<1800:
                    status(wait='RAM admission for one software-decoded review child');time.sleep(10)
                label=f'{item["backup"]}_{item["scene_id"]}_{module.rsplit(".",1)[-1]}'
                with (directory/(label+'.log')).open('x') as stream:
                    child=subprocess.Popen([str(SIM),'-u','-m',module,'--pair',str(folder)],
                        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    status()
                    while child.poll() is None:time.sleep(2)
                    if child.returncode:raise RuntimeError(f'{label} exited{child.returncode}; no auto approval')
                    child=None
            record['status']='prepared_actual_images_NEEDS_HUMAN_OR_AGENT_VIEW';status()
        except Exception as error:
            child=None;record.update(status='preparation_error_preserved',error=f'{type(error).__name__}: {error}');status()
        done.add(str(folder))
    status('preparation_complete_MANUAL_VISUAL_APPROVAL_STILL_REQUIRED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT/'prepare_review_v1')
    main(parser.parse_args().output)
