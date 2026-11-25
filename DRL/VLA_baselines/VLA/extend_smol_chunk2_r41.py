"""Conditional preserve40/extend80 native chunk2 DEVELOPMENT confirmation.

No automatic model deployment or change to frozen800/pending fits. Only a
strict paired gain with no additional safety stops on ALL40 permits the
remaining20 original development seeds. First40 are preserved, not rerolled.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from VLA import evaluate_smol_chunk2_r41 as native
from VLA import evaluate_smol800_r39 as cohort
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.probe_smol_train_now_r40 import stable_window
from VLA.wait_render_then_resume_queues import identity,render_jobs

ROOT=native.ROOT;BASE=native.BASE;SIM=native.SIM
SOURCE=BASE/'Smol_native_chunk2_development40_r41'
SEEDS=cohort.OLD_SEEDS


def extend_allowed(comparison,remaining_seconds):
    return bool(comparison.get('trials_each')==40
        and comparison.get('paired_improvements',0)>comparison.get('paired_regressions',0)
        and comparison.get('native2_safety_stops',41)<=comparison.get('native1_safety_stops',0)
        and remaining_seconds>=6*3600)


def compare120(reference,rows):
    keys=[(s,t) for s in SEEDS for t in cohort.TASKS]
    if [(r['seed'],r['task']) for r in rows]!=keys or [(r['seed'],r['task']) for r in reference]!=keys:
        raise ValueError('Full same locked120 required')
    before={(r['seed'],r['task']):r for r in reference}
    wins=sum(r['success'] and not before[r['seed'],r['task']]['success'] for r in rows)
    losses=sum(not r['success'] and before[r['seed'],r['task']]['success'] for r in rows)
    return {'trials_each':120,'native1_successes':sum(r['success'] for r in reference),
        'native2_successes':sum(r['success'] for r in rows),'paired_improvements':wins,'paired_regressions':losses,
        'native1_safety_stops':sum(bool(r.get('safety_violation')) for r in reference),
        'native2_safety_stops':sum(bool(r.get('safety_violation')) for r in rows),
        'DEVELOPMENT_ONLY_NOT_final_paper_SR':True,'frozen800_and_pending_fit_unchanged':True,
        'same_NN_solver_inputs_robot_scoring_except_native_queue_length':True}


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[];stage='waiting_ALL40_terminal_before_conditional_decision';error=None;child=None
    deadline=time.monotonic()+a.wait_hours*3600
    frozen={str(p):sha256_file(p) for p in (Path(__file__).resolve(),ROOT/'VLA/evaluate_smol_chunk2_r41.py')}
    atomic_json(out/'extension_rule_declared.json',{'source40':str(SOURCE),'maximum_additional_trials':80,
        'rule':'ALL40: net paired gain, no extra safety stops, at least6h original800 budget remaining',
        'locked_remaining_seeds':SEEDS[10:],'preserve_all_first40_NOT_reroll':True,
        'no_automatic_promotion_or_new_finetune_recipe':True,'frozen_files':frozen})
    def save(status='running'):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'completed_trials':len(rows),'planned_total_if_extended':120,'maximum_new_trials':80,
            'active_child_pid':child.pid if child and child.poll() is None else None,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'other_process_signalled':False,
            '800_used_for_fitting_or_selection':False})
        if rows:atomic_json(out/'progress.json',{'status':status,'variant':'native10_native_chunk2',
            'rows':rows,'completed_trials':len(rows),'planned_trials':120,'successes':sum(r['success'] for r in rows),
            'by_action':cohort.counts(rows),'preserved_first40':40,'DEVELOPMENT_ONLY':True,'error':error})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite conditional DEVELOPMENT budget')
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen extension code changed')
    save()
    try:
        while True:
            check()
            state=json.loads((SOURCE/'status.json').read_text())
            if state['status'].startswith('failed'):
                stage='skipped_source40_failed_no_extra_GPU_or_zero_SR';save('skipped');return
            if state['status']=='completed' and not identity(state['pid']):break
            save();time.sleep(10)
        pointer=json.loads((BASE/'Smol_800_r39_active.json').read_text());owner_root=Path(pointer['cycle'])
        owner_state=json.loads((owner_root/'status.json').read_text());owner=identity(owner_state['pid'])
        comparison=json.loads((SOURCE/'matched_comparison.json').read_text())
        decision=extend_allowed(comparison,owner_state['deadline_remaining_seconds'])
        atomic_json(out/'conditional_decision.json',{'extend_remaining80':decision,'complete_source40':comparison,
            'remaining_original800_budget_seconds':owner_state['deadline_remaining_seconds'],
            'decision_before_any_extension_outcome':True,'no800_result_read':True})
        if not decision:
            stage='completed_no_extension_rule_not_met_NO_new_NN';save('completed_no_extension');return
        if not owner or b'VLA.resume_smol800_r40' not in owner['tokens'] or not cohort.held_priority_lease():
            raise ValueError('Exact800 owner/resource lease required for safe extension')
        plan=json.loads((SOURCE/'plan.json').read_text());reference=json.loads(native.REFERENCE.read_text())['rows']
        source_progress=json.loads((SOURCE/'progress.json').read_text());rows=list(source_progress['rows'])
        if len(rows)!=40 or [(r['seed'],r['task']) for r in rows]!=[(s,t) for s in SEEDS[:10] for t in cohort.TASKS]:
            raise ValueError('Preserve exact source40 outcomes')
        plan.update(seeds=SEEDS,trials=cohort.trial_specs(SEEDS),reference_rows=reference,planned_trials=120,
            preserved_terminal_trials=40,preserved_source40=str(SOURCE),source40_progress_sha256=sha256_file(SOURCE/'progress.json'))
        for path,digest in frozen.items():
            plan['frozen_files'][path]=digest;s=Path(path).stat();plan['frozen_file_identity'][path]=[s.st_ino,s.st_size,s.st_mtime_ns]
        native.verify_plan(plan);atomic_json(out/'plan.json',plan)
        atomic_json(out/'preserved40.json',{'rows':rows,'source_progress_sha256':plan['source40_progress_sha256'],
            'trajectory_sha256':{r['trajectory']:sha256_file(r['trajectory']) for r in rows},'rerolled_episodes':0})
        import torch
        torch.set_num_threads(1)
        gpu=cohort.resources()
        if gpu['available_RAM_MiB']<2048 or gpu['free_mib']<512:raise ValueError('Insufficient even for tiny reservation')
        torch.cuda.set_per_process_memory_fraction(64*2**20/torch.cuda.get_device_properties(0).total_memory)
        reservation=torch.empty(1,device='cuda')
        for seed in SEEDS[10:]:
            ready_since=None;stage='waiting_safe800_inter_seed_window_and_initial_RAM';save()
            while True:
                check();native.verify_plan(plan);actual=identity(owner_state['pid']);lease=cohort.held_priority_lease()
                if not actual or actual['start_ticks']!=owner['start_ticks'] or not lease:
                    raise ValueError('Existing800/priority ownership ended; preserve extension')
                state=json.loads((owner_root/'status.json').read_text());gpu=cohort.resources()
                if stable_window(state,identity(state.get('active_child_pid') or 0),gpu,
                        {os.getpid(),owner_state['pid'],lease['trainer_pid']}) and not render_jobs():
                    if ready_since is None:ready_since=time.monotonic()
                    if time.monotonic()-ready_since>=2:break
                else:ready_since=None
                save();time.sleep(2)
            atomic_json(out/f'seed{seed}_resource_admission.json',{'resources':gpu,'original800_worker_absent':True,
                'own_worker_allocator_cap_MiB':3072,'trainer_or_episode_signalled':False})
            command=[str(SIM),'-m','VLA.extend_smol_chunk2_r41','--output',str(out),'--seed',str(seed)]
            with (out/f'seed{seed}_worker.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                    'MALLOC_ARENA_MAX':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'},
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(out/'active_child.json',{'pid':child.pid,'seed':seed,'command':command})
                stage='actual_native_queue2_additional_development_seed';save()
                def harvest():
                    path=out/f'test_seed{seed}/seed_result.json'
                    if not path.exists():return
                    for row in json.loads(path.read_text())['rows']:
                        if (row['seed'],row['task']) in {(r['seed'],r['task']) for r in rows}:continue
                        expected=[(s,t) for s in SEEDS for t in cohort.TASKS]
                        if (row['seed'],row['task'])!=expected[len(rows)]:raise ValueError('No omitted/reordered/duplicate trial')
                        trace=json.loads(Path(row['trajectory']).read_text())[row['task']]
                        if len(trace)!=row['steps'] or bool(trace[-1]['success'])!=row['success']:raise ValueError('Terminal evidence mismatch')
                        rows.append(row);save()
                while child.poll() is None:harvest();save();time.sleep(5)
                harvest()
                if child.returncode:raise RuntimeError(f'Additional seed{seed} failed; NOT rerolled or zero-filled')
                if len(rows)!=4*(SEEDS.index(seed)+1):raise ValueError('Incomplete terminal seed block')
                child=None
        native.verify_plan(plan);atomic_json(out/'matched_comparison.json',compare120(reference,rows))
        atomic_json(out/'statistics.json',cohort.success_statistics(rows))
        stage='completed120_preserved40_plus_new80_REQUIRES_review_NOT_auto_promotion';save('completed120');del reservation
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';save('failed_preserve_actual_trials_NOT_zero_fill');raise
    finally:
        while child is not None and child.poll() is None:time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS[10:]);p.add_argument('--wait-hours',type=float,default=4)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=5:raise ValueError('Finite conditional extension only')
    if a.seed is not None:native.run_seed(a)
    elif not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate extension')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.extend_smol_chunk2_r41','--output',str(a.output.resolve()),
                '--wait-hours',str(a.wait_hours)],cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MALLOC_ARENA_MAX':'2'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'initial_CPU_only_conditional_watcher':True});print(child.pid)
