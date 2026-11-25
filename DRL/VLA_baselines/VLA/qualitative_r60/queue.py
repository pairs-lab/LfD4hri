"""Finite two-pair CPU proof queue followed by one resource-gated HD worker."""
import os
from pathlib import Path
import subprocess
import sys
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_r57.common import read
from .prepare import OUT,OLD,ROOT


def wait_complete(path):
    deadline=time.monotonic()+1200
    while not (path/'physics_event_ticks.json').exists():
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(2)


def repeat_proof(cap,repeat,target):
    for path in (cap,repeat):wait_complete(path)
    names=['initial_scene.json','recorded_physics_states.json','physics_event_ticks.json','outcome.json','trajectory.json']
    checks={n:sha256_file(cap/n)==sha256_file(repeat/n) for n in names}
    result={'passed':all(checks.values()),'exact_files':checks,'target':str(cap),'repeat':str(repeat)}
    atomic_json(target,result)
    if not result['passed']:raise ValueError('Independent Oracle repeat mismatch')


def main():
    records=[]
    ground=OUT/'_captures/scene12_seed20269719'
    miss=OUT/'_captures/scene13_seed20271060'
    proof=ground/'repeat_proof.json'
    repeat_proof(ground/'Oracle_probe00',ground/'Oracle_repeat00',proof)
    old=next(r for r in read(OLD/'produced_pairs.json') if not r['backup'] and r['scene_id']==1)
    assert not read(ground/'Oracle_probe00/content_evidence.json')['full_put_complete']
    assert any(e.get('ground_slam_force_velocity_gate_pass') for e in read(ground/'Oracle_probe00/content_evidence.json')['events'])
    records.append({'scene_id':12,'backup':False,'seed':20269719,'group':'02_RL_fail_VLA_succeed',
       'vla':old['vla'],'oracle':str(ground/'Oracle_probe00'),'repeat_proof':str(proof),
       'oracle_fault':'STAGED_wrong_ground_height','VLA_full_put':True,'Oracle_full_put':False})
    atomic_json(OUT/'assignments.json',records)
    proof=miss/'repeat_proof.json'
    repeat_proof(miss/'Oracle_none',miss/'Oracle_repeat',proof)
    assert read(miss/'Oracle_none/content_evidence.json')['full_put_complete']
    assert not read(miss/'XVLA_verified/content_evidence.json')['full_put_complete']
    records.append({'scene_id':13,'backup':False,'seed':20271060,'group':'03_RL_succeed_VLA_fail',
       'vla':str(miss/'XVLA_verified'),'oracle':str(miss/'Oracle_none'),'repeat_proof':str(proof),
       'oracle_fault':'none','VLA_full_put':False,'Oracle_full_put':True})
    atomic_json(OUT/'assignments.json',records)
    atomic_json(OUT/'capture_done.json',{'all_new_pairs_CPU_verified':True,'pairs':2,
        'still_requires_media_and_actual_visual_review':True})
    atomic_json(OUT/'NEW_SOURCE_MANIFEST.json',{str(p.relative_to(ROOT)):sha256_file(p)
        for p in Path(__file__).parent.iterdir() if p.is_file()})
    status=OUT/'queue_status.json'
    with (OUT/'production.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u','-m','VLA.qualitative_r60.produce'],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(status,{'status':'single_HD_worker_running','pid':os.getpid(),'child':child.pid})
        code=child.wait()
    if code:raise RuntimeError('Production failed, see production.log; all artifacts retained')
    rows=read(OUT/'produced_pairs.json')
    for row in rows:
        if 'error' in row:raise ValueError(row['error'])
        subprocess.run([sys.executable,'-u','-m','VLA.qualitative_r58.review','--pair',row['paired_output']],
            cwd=ROOT,check=True)
    atomic_json(status,{'status':'media_ready_ACTUAL_visual_inspection_required','pairs':len(rows)})


if __name__=='__main__':main()
