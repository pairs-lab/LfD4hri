"""Sequential resource-guarded qualitative capture/render/package owner.

No neural model fitting, no alteration/preemption of quantitative jobs. All
failures and incomplete candidates are retained; desired category is NOT truth.
"""
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
from .common import OUT,ROOT,read
from .selection import assess_vla


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def resources():
    ram=next(int(l.split()[1])/1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return {'RAM_available_MiB':ram,'GPU_free_MiB':float(r.stdout.splitlines()[0]) if r.returncode==0 else 0}


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    cases=read(OUT/'candidate_plan.json')['candidate_pairs'];results=[]
    frozen={str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    child=None;stage='CPU_capture'
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'pid':os.getpid(),'stage':stage,
            'active_child_pid':None if child is None else child.pid,'results':results,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})
    def execute(module,opts,label,gpu=False):
        nonlocal child
        for path,digest in frozen.items():
            if sha256_file(path)!=digest:raise ValueError('Own frozen qualitative source changed: '+path)
        until=time.monotonic()+30*3600
        while True:
            available=resources()
            if available['RAM_available_MiB']>=3500 and (not gpu or available['GPU_free_MiB']>=1800):break
            if time.monotonic()>until:raise TimeoutError('Finite resource wait exceeded')
            status(resources=available,waiting_without_killing_any_training=True);time.sleep(20)
        with (a.output/(label+'.log')).open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);status()
            while child.poll() is None:time.sleep(10);status()
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label} exited{code}; keep failed artifact, not fake video')
    try:
        atomic_json(a.output/'plan.json',{'code':frozen,'candidate_plan_sha256':sha256_file(OUT/'candidate_plan.json'),
            'main_candidates':11,'backup_candidates':4,'put_only':True,
            'no_quantitative_optimizer_or_neural_server_started_here':True,
            'Oracle_faults_and_VLA_terminal_release_are_explicit_qualitative_variants':True,
            'desired_slots_NOT_guaranteed_outcomes':True,'no_memory_loss_or_identity_claim_without_proof':True})
        for case in cases:
            seed=case['seed'];root=OUT/'05_backups'/case['candidate_id'];root.mkdir(parents=True,exist_ok=True)
            row={'seed':seed,'proposed_scene_id':case['proposed_scene_id'],'proposed_group':case['proposed_group'],
                'oracle_fault':case['oracle_fault'],'status':'CPU_running'}
            results.append(row);status()
            try:
                if seed==20269730:
                    vla=root/'XVLA_capture_v2';oracle=root/'Oracle_none_capture';repeat=root/'Oracle_none_repeat'
                elif seed==20269719:
                    vla=root/'XVLA_qual_v2';oracle=root/'Oracle_none_capture';repeat=root/'Oracle_none_repeat'
                else:
                    vla=root/'XVLA_qual_v1';oracle=root/f'Oracle_{case["oracle_fault"]}_capture';repeat=root/f'Oracle_{case["oracle_fault"]}_repeat'
                if not (vla/'replay_proof.json').exists():
                    if vla.exists():raise ValueError('Preserve incomplete VLA attempt; explicit repair/revision needed')
                    execute('VLA.qualitative_r57.replay',['--seed',seed,'--terminal-release','--shortlist',OUT/'all_trace_candidates.json',
                        '--output',vla],f'{seed}_VLA')
                row['vla']=str(vla)
                for index,target in enumerate((oracle,repeat)):
                    if not (target/'outcome.json').exists():
                        if target.exists():raise ValueError('Preserve incomplete Oracle attempt')
                        execute('VLA.qualitative_r57.oracle',['--seed',seed,'--paired-capture',vla,
                            '--fault',case['oracle_fault'],'--output',target],f'{seed}_Oracle{index}')
                names=['initial_scene.json','recorded_physics_states.json','physics_event_ticks.json','outcome.json','trajectory.json']
                equal={name:sha256_file(oracle/name)==sha256_file(repeat/name) for name in names}
                proof=root/'Oracle_repeat_proof.json';atomic_json(proof,{'passed':all(equal.values()),'exact_files':equal})
                if not all(equal.values()):raise ValueError('Oracle whole CPU capture repeat differs')
                # Recompute small evidence artifacts under this frozen audit
                # revision; old classifiers may confuse contact loss with drop.
                for label,target in (('VLA',vla),('Oracle',oracle)):
                    execute('VLA.qualitative_r57.events',['--capture',target],f'{seed}_{label}_events')
                ve=read(vla/'content_evidence.json');oe=read(oracle/'content_evidence.json')
                vs,osuccess=ve['full_put_complete'],oe['full_put_complete']
                group=('01_RL_succeed_VLA_succeed' if vs and osuccess else '02_RL_fail_VLA_succeed' if vs
                    else '03_RL_succeed_VLA_fail' if osuccess else '04_RL_fail_VLA_fail')
                strength=assess_vla(ve,read(vla/'replay_proof.json'))
                atomic_json(vla/'selection_gate.json',strength)
                row.update(oracle=str(oracle),repeat_proof=str(proof),VLA_full_put=vs,Oracle_full_put=osuccess,
                    actual_group=group,desired_group_matches=group==case['proposed_group'],
                    VLA_failure_has_measured_pre_release_event=strength['strong_failure_candidate'],
                    selection_gate=strength,
                    VLA_failure_modes=[e['mode'] for e in ve['events']],status='CPU_audited_NEEDS_media_and_visual_review')
            except Exception as error:row.update(status='failed_preserved_NOT_certified',error=f'{type(error).__name__}: {error}')
            atomic_json(OUT/'candidate_ledger.json',results);status()
        stage='waiting_actual_pilot_media_and_visual_gates';status()
        pilot=OUT/'05_backups/put_seed20269730/XVLA_capture_v2'
        until=time.monotonic()+3*3600
        while not (pilot/'media_status.json').exists():
            if a.existing_render_pid and not alive(a.existing_render_pid):
                raise RuntimeError('Existing pilot renderer exited without media pass; inspect preserved output before batch render')
            if time.monotonic()>until:raise TimeoutError('Pilot full media gate not ready within3h')
            time.sleep(20);status()
        if not (OUT/'audit/pilot_actual_frame_review.json').exists():raise ValueError('Actual opened pilot event frames not reviewed')
        stage='one_render_worker_and_common_pair_layout';status()
        for row in results:
            if row['status'].startswith('failed'):continue
            try:
                vla=Path(row['vla']);oracle=Path(row['oracle'])
                # Do not spend hours rendering a region-only/incomplete release
                # as a persuasive main VLA failure against the user's request.
                if not row['VLA_full_put'] and not row['VLA_failure_has_measured_pre_release_event']:
                    row['status']='backup_only_no_convincing_pre_release_VLA_failure';status();continue
                for label,target in (('VLA',vla),('Oracle',oracle)):
                    if not (target/'renderer.json').exists():
                        execute('VLA.qualitative_r57.render',['--capture',target,'--render-only'],f'{row["seed"]}_{label}_render',True)
                certified_group=row['actual_group'] if row['desired_group_matches'] else '05_backups'
                scene=row['proposed_scene_id']
                target=OUT/certified_group/(f'scene{scene:02d}_put_seed{row["seed"]}' if scene else f'backup_put_seed{row["seed"]}')
                execute('VLA.qualitative_r57.package',['--vla',vla,'--oracle',oracle,'--repeat-proof',row['repeat_proof'],
                    '--scene-id',scene or 99,'--output',target],f'{row["seed"]}_paired_package')
                row.update(status='media_audited_FINAL_visual_review_pending',paired_output=str(target))
            except Exception as error:row.update(status='media_gate_failed_preserved',error=f'{type(error).__name__}: {error}')
            atomic_json(OUT/'candidate_ledger.json',results);status()
        stage='finite_capture_render_package_complete_REQUIRES_honest_final_visual_coverage_review'
        status('completed_batches_NOT_claimed22_certified_principal_videos')
    except Exception as error:
        status('failed_preserve_all_artifacts_and_training',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT/'automatic_v1')
    parser.add_argument('--existing-render-pid',type=int)
    parser.add_argument('--launch',action='store_true');a=parser.parse_args()
    if a.launch:
        with a.output.with_suffix('.log').open('x') as log:
            command=[sys.executable,'-u','-m','VLA.qualitative_r57.run','--output',str(a.output.resolve())]
            if a.existing_render_pid:command+=['--existing-render-pid',str(a.existing_render_pid)]
            owner=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':owner.pid,'output':str(a.output.resolve())},flush=True)
    else:main(a)
