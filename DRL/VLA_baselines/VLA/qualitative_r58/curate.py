"""Keep filling exact2/3/3/3 outcome slots and four backup pairs from real runs."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,PRIOR,ROOT,SIM,read
from .selection import assess

SLOTS=[
 (1,True,True,[20269719],['none']), (2,True,True,[20269711],['none']),
 (3,False,True,[20269727,20269721],['drop_in_transit']),
 (4,False,True,[20268104,20268016],['closed_jaw_approach','persistent_miss']),
 (5,False,True,[20269701,20269715],['slow_transport_timeout']),
 (6,True,False,[20269730],['none']), (7,True,False,[20269722],['none']),
 (8,True,False,[20269703,20268107],['none']),
 (9,False,False,[20269718,20268107],['drop_in_transit','persistent_miss']),
 (10,False,False,[20269704,20268020],['closed_jaw_approach','persistent_miss']),
 (11,False,False,[20268015,20268020,20268107],['persistent_miss','drop_in_transit']),
]
BACKUPS=[(1,20269721,'none'),(2,20269705,'slow_transport_timeout'),
         (3,20268016,'drop_in_transit'),(4,20269715,'persistent_miss')]


def group(o,v):
    return ('01_RL_succeed_VLA_succeed' if o and v else '02_RL_fail_VLA_succeed' if v
        else '03_RL_succeed_VLA_fail' if o else '04_RL_fail_VLA_fail')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);(OUT/'_captures').mkdir(parents=True,exist_ok=True)
    rows=[];assignments=[];selected_seeds=set();child=None
    sources={str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    atomic_json(a.output/'plan.json',{'main_slots':SLOTS,'backup_proposals':BACKUPS,'source_at_start':sources,
        'missing_slots_must_not_be_reported_complete':True,'all_outcomes_are_actual_not_proposal_labels':True})
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'pid':os.getpid(),'status':state,'child':child.pid if child else None,
            'selected_main':sum(not r['backup'] for r in assignments),'selected_backup':sum(r['backup'] for r in assignments),
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})
        atomic_json(OUT/'assignments.json',assignments);atomic_json(OUT/'candidate_ledger.json',rows)
    def execute(module,opts,label):
        nonlocal child
        while next(int(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))<2500:
            status(wait='RAM admission, no training preemption');time.sleep(10)
        log=a.output/(label+'.log');attempt=1
        while log.exists():attempt+=1;log=a.output/(label+f'_attempt{attempt}.log')
        with log.open('x') as stream:
            child=subprocess.Popen([str(SIM),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);status(active_label=label)
            while child.poll() is None:time.sleep(1);status(active_label=label)
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label}: exit{code}; actual failure artifact retained')
    def vla(seed):
        new=OUT/f'_captures/put_seed{seed}/XVLA_release_r58'
        oldroot=PRIOR/f'05_backups/put_seed{seed}'
        choices=[oldroot/'XVLA_capture_v2',oldroot/'XVLA_qual_v2',oldroot/'XVLA_qual_v1']
        cap=None
        if (new/'replay_proof.json').exists():cap=new
        if cap is None:
            for old in choices:
                if not (old/'replay_proof.json').exists():continue
                proof=read(old/'replay_proof.json');tail=proof.get('terminal_release_supervisor')
                # Existing successful opening yields identical commands under
                # the new attempt gate. Native failures never enter supervisor.
                if not proof['benchmark_region_success'] or (tail and tail.get('attempted') and tail.get('qualitative_put_complete')):
                    cap=old;break
        if cap is None:
            if new.exists():raise ValueError('Preserve incomplete attempt; needs explicit revision: '+str(new))
            execute('VLA.qualitative_r58.replay',['--seed',seed,'--output',new],f'{seed}_VLA')
            cap=new
        execute('VLA.qualitative_r58.events',['--capture',cap],f'{seed}_VLA_evidence')
        evidence=read(cap/'content_evidence.json');proof=read(cap/'replay_proof.json');strength=assess(evidence,proof)
        return cap,evidence,strength
    def oracle(seed,cap,fault):
        oldroot=PRIOR/f'05_backups/put_seed{seed}'
        old=oldroot/f'Oracle_{fault}_capture';oldrepeat=oldroot/f'Oracle_{fault}_repeat'
        if fault in ('none','drop_in_transit','slow_transport_timeout') and (old/'outcome.json').exists() and (oldrepeat/'outcome.json').exists():
            target,repeat=old,oldrepeat
        else:
            target=OUT/f'_captures/put_seed{seed}/Oracle_{fault}_capture'
            repeat=OUT/f'_captures/put_seed{seed}/Oracle_{fault}_repeat'
        for i,path in enumerate((target,repeat)):
            if not (path/'outcome.json').exists():
                if path.exists():raise ValueError('Incomplete Oracle attempt retained: '+str(path))
                execute('VLA.qualitative_r58.oracle',['--seed',seed,'--paired-capture',cap,'--fault',fault,'--output',path],f'{seed}_{fault}_{i}')
        names=['initial_scene.json','recorded_physics_states.json','physics_event_ticks.json','outcome.json','trajectory.json']
        equal={name:sha256_file(target/name)==sha256_file(repeat/name) for name in names}
        proof=OUT/f'_captures/put_seed{seed}/repeat_{fault}.json'
        atomic_json(proof,{'passed':all(equal.values()),'exact_files':equal,'target':str(target),'repeat':str(repeat)})
        if not all(equal.values()):raise ValueError('Oracle repeat parity failed')
        if read(cap/'initial_scene.json')['measured_state_sha256']!=read(target/'initial_scene.json')['measured_state_sha256']:
            raise ValueError('Paired measured initial state differs')
        execute('VLA.qualitative_r58.events',['--capture',target],f'{seed}_{fault}_evidence')
        return target,proof,read(target/'content_evidence.json')
    try:
        for slot,owant,vwant,seeds,faults in SLOTS:
            filled=False
            for seed in seeds:
                if seed in selected_seeds:continue
                row={'seed':seed,'slot_requested':slot,'status':'capturing'};rows.append(row);status(slot=slot)
                try:
                    cap,ve,strength=vla(seed);row.update(vla=str(cap),VLA_full_put=ve['full_put_complete'],strength=strength)
                    if ve['full_put_complete']!=vwant or (not vwant and not strength['strong_failure_candidate']):
                        row.update(status='not_matching_slot_outcome_or_content_kept');status();continue
                    for fault in faults:
                        ocap,proof,oe=oracle(seed,cap,fault)
                        row.update(oracle=str(ocap),Oracle_full_put=oe['full_put_complete'],oracle_fault=fault)
                        if oe['full_put_complete']!=owant:continue
                        if not owant and not oe['events']:continue
                        item={'scene_id':slot,'backup':False,'seed':seed,'group':group(owant,vwant),
                            'vla':str(cap),'oracle':str(ocap),'repeat_proof':str(proof),'oracle_fault':fault,
                            'VLA_full_put':vwant,'Oracle_full_put':owant,'status':'CPU_selected_NEEDS_video_and_visual_review'}
                        assignments.append(item);selected_seeds.add(seed);row['status']='selected_main_CPU';filled=True;status();break
                    if filled:break
                    row['status']='Oracle_outcome_not_matching_slot_kept';status()
                except Exception as error:row.update(status='candidate_error_preserved',error=f'{type(error).__name__}: {error}');status()
            if not filled:status(missing_main_slot=slot)
        for backup_id,seed,fault in BACKUPS:
            if seed in selected_seeds:continue
            row={'seed':seed,'backup_requested':backup_id,'status':'capturing_backup'};rows.append(row);status()
            try:
                cap,ve,strength=vla(seed);ocap,proof,oe=oracle(seed,cap,fault)
                assignments.append({'scene_id':backup_id,'backup':True,'seed':seed,'group':'05_backups',
                    'vla':str(cap),'oracle':str(ocap),'repeat_proof':str(proof),'oracle_fault':fault,
                    'VLA_full_put':ve['full_put_complete'],'Oracle_full_put':oe['full_put_complete'],
                    'status':'CPU_backup_NEEDS_video_and_visual_review'})
                row['status']='selected_backup_CPU';status()
            except Exception as error:row.update(status='backup_error_preserved',error=f'{type(error).__name__}: {error}');status()
        missing=sorted(set(range(1,12))-{r['scene_id'] for r in assignments if not r['backup']})
        status('CPU_selection_complete' if not missing else 'CPU_missing_slots_REPAIR_REQUIRED',missing_slots=missing)
        atomic_json(OUT/'capture_done.json',{'all11_main_CPU_selected':not missing,'missing_slots':missing,
            'backup_pairs':sum(r['backup'] for r in assignments)})
    except Exception as error:status('failed_preserved',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT/'capture_v1')
    main(parser.parse_args())
