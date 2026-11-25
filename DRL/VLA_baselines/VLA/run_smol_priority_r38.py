"""User-requested Smol-first scheduling without stopping a neural worker.

After existing Smol diagnostics finish, hold ONLY the exact R33 supervisor
while its identified RDT trainer continues normally. Once that trainer exits,
run the already-audited GT-view Smol recipe before the next family. Restore
the SAME supervisor identity after Smol, even after a wrapper failure. No
training weights, environment, success or native action recipe are changed.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import ROOT
from VLA.wait_render_then_resume_queues import identity,render_jobs

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
OWNER=BASE/'all_family_privileged_r33_v4'
FIT=BASE/'Smol_GTcrop_motion_r36_v6_priority'
FIT_ROUND2=BASE/'Smol_GTcrop_motion_r36_v7_priority_continuation'
PREREQUISITES=('Smol_native20_extension_r37','Smol_contactstop_closedloop_r37','Smol_bounded_feedback40_r38')


def eligible(state, child, supervisor, trainer):
    if not supervisor or not trainer:return False
    return bool(state.get('status')=='running' and state.get('stage')=='rdt_fit_2000'
        and child.get('stage')=='rdt_fit_2000'
        and supervisor['cwd']==str(ROOT) and trainer['cwd']==str(ROOT)
        and supervisor['state'] not in ('T','Z') and trainer['state'] not in ('T','Z')
        and b'VLA.run_all_privileged_r33' in supervisor['tokens'] and str(OWNER).encode() in supervisor['tokens']
        and b'VLA.rdt.train_scene67' in trainer['tokens'] and str(OWNER/'rdt_fit_2000').encode() in trainer['tokens']
        and set(supervisor['children'].split())=={str(child['pid'])})


def same_supervisor(lease, process):
    return bool(process and process['cwd']==str(ROOT) and process['start_ticks']==lease['supervisor_start_ticks']
        and b'VLA.run_all_privileged_r33' in process['tokens'] and str(OWNER).encode() in process['tokens'])


def owned_fit_child_alive(root=None):
    fit=FIT
    if root is not None and (root/'active_fit_cycle.json').exists():
        fit=Path(json.loads((root/'active_fit_cycle.json').read_text())['cycle'])
        if fit not in (FIT,FIT_ROUND2):raise ValueError('Unknown owned continuation fit')
    path=fit/'active_child.json'
    if not path.exists():return False
    child=json.loads(path.read_text());p=identity(child['pid'])
    return bool(p and p['cwd']==str(ROOT) and str(fit).encode() in b' '.join(p['tokens']))


def allow_round2(first_successes, baseline_successes, remaining_hours, trials=120):
    """Predeclared exploratory rule, NOT a statistical/generalization claim."""
    return bool(trials==120 and baseline_successes<first_successes<72 and remaining_hours>=8)


def restore(lease):
    p=identity(lease['supervisor_pid'])
    if same_supervisor(lease,p) and p['state']=='T':
        os.kill(lease['supervisor_pid'],signal.SIGCONT)
        return True
    return False


def guard(root):
    """No broad kills: only CONT our exact stopped supervisor when safe."""
    lease=json.loads((root/'supervisor_lease.json').read_text())
    while True:
        parent=identity(lease['priority_owner_pid'])
        same_owner=bool(parent and parent['cwd']==str(ROOT) and parent['start_ticks']==lease['priority_owner_start_ticks']
            and b'VLA.run_smol_priority_r38' in parent['tokens'] and str(root).encode() in parent['tokens'])
        main=identity(lease['supervisor_pid'])
        if not same_supervisor(lease,main) or main['state']!='T':return
        if not same_owner and not owned_fit_child_alive(root):
            restored=restore(lease)
            atomic_json(root/'watchdog_restore.json',{'restored_exact_supervisor':restored,
                'orphan_Smol_child_still_running':False,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
            return
        time.sleep(15)


def run(a):
    out=a.output.resolve()
    if out.parent!=BASE or not out.name.startswith('Smol_GTcrop_motion_r36_'):raise ValueError('Registered local priority root required')
    out.mkdir(parents=True,exist_ok=False);stage='waiting_existing_Smol120_contact40_bounded40';lease=None;active_fit=FIT;round_number=1
    frozen={str(Path(__file__)):sha256_file(Path(__file__))};deadline=time.monotonic()+a.wait_hours*3600
    def save(status='running',error=None,**extra):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_files':frozen,
            'user_requested_Smol_first':True,'actual_fit_cycle':str(active_fit),'round':round_number,
            'RDT_trainer_server_or_episode_signalled':False,'other_VLA_families_keep_priority':False,**extra})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite priority budget')
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Priority coordinator changed')
    atomic_json(BASE/'Smol_GTcrop_motion_r36_active.json',{'cycle':str(out),'pid':os.getpid(),
        'actual_fit_cycle':str(FIT),'reason':'new explicit user request: Smol first, transfer verified repairs later'})
    atomic_json(out/'priority_plan.json',{'prerequisites':list(PREREQUISITES),'same_prepared_GTcrop_fit':True,
        'fit_optimizer_updates':2500,'development_trials':120,'pending_families_resume_same_process_after_Smol':True,
        'only_supervisor_may_receive_STOP_NOT_actual_trainer':True,'new_family_success_rate_NOT_established':True,
        'max_actual2500_update_rounds':2,'second_round_rule':'first completed120 strictly improves39/120 baseline, remains below72/120, at least8h left',
        'second_round_same_data_recipe_new_Adam_NOT_optimizer_resume':True,
        'frozen_files':frozen})
    save()
    try:
        while True:
            check();pending=[]
            for name in PREREQUISITES:
                path=BASE/name/'status.json'
                state=json.loads(path.read_text()) if path.exists() else {}
                terminal=state.get('status')=='completed' or str(state.get('status','')).startswith('failed')
                process=identity(state.get('pid',0))
                # Terminal status can precede NN/model cleanup by milliseconds.
                if not terminal or process:pending.append(name)
            if not pending:break
            save(pending_Smol_diagnostics=pending);time.sleep(15)
        state=json.loads((OWNER/'status.json').read_text());child=json.loads((OWNER/'active_child.json').read_text())
        supervisor=identity(state['pid']);trainer=identity(child['pid'])
        if eligible(state,child,supervisor,trainer) and not render_jobs():
            lease={'supervisor_pid':state['pid'],'supervisor_start_ticks':supervisor['start_ticks'],
                'trainer_pid_NOT_signalled':child['pid'],'trainer_start_ticks':trainer['start_ticks'],
                'priority_owner_pid':os.getpid(),'priority_owner_start_ticks':identity(os.getpid())['start_ticks'],
                'supervisor_only_pause_not_neural_training_pause':True}
            atomic_json(out/'supervisor_lease.json',lease)
            os.kill(state['pid'],signal.SIGSTOP)
            # Confirm no child transition raced the single supervisor signal.
            actual=identity(state['pid']);same_child=identity(child['pid'])
            until=time.monotonic()+2
            while actual and actual['state']!='T' and time.monotonic()<until:
                time.sleep(.02);actual=identity(state['pid'])
            if (not same_supervisor(lease,actual) or actual['state']!='T'
                    or not set(actual['children'].split())<={str(child['pid'])}
                    or (same_child and same_child['start_ticks']!=trainer['start_ticks'])):
                restore(lease);lease=None
                raise ValueError('Supervisor transition raced; restored, no new model started')
            with (out/'watchdog.log').open('x') as log:
                watchdog=subprocess.Popen([str(SIM),'-m','VLA.run_smol_priority_r38','--guard-root',str(out)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(out/'supervisor_pause_verified.json',{'paused_only_exact_supervisor':state['pid'],
                'actual_RDT_trainer_still_running':bool(same_child),'actual_RDT_trainer_state':same_child['state'] if same_child else 'already_exited',
                'watchdog_pid':watchdog.pid,'actual_trainer_server_episode_signalled':False})
        # A family transition that has already happened is not preempted.
        # In that case retain existing waits; no in-flight model is stopped.
        from VLA import run_target_crop_r36 as native
        if lease is not None:native.PARENTS=[]
        original_write=native.atomic_json
        def write(path,value):
            check()
            if path==BASE/'Smol_GTcrop_motion_r36_active.json':return
            original_write(path,value)
            if path==active_fit/'status.json':
                global_stage='priority_GTcrop_round'+str(round_number)+'_'+value['stage']
                original_write(out/'status.json',{**value,'stage':global_stage,'pid':os.getpid(),
                    'frozen_files':{**value.get('frozen_files',{}),**frozen},'actual_fit_cycle':str(active_fit),'round':round_number,
                    'user_requested_Smol_first':True,'other_VLA_families_keep_priority':False,
                    'supervisor_only_held_while_RDT_trainer_runs_normally':lease is not None,
                    'RDT_trainer_server_or_episode_signalled':False})
        native.atomic_json=write
        # Expose the exact trial artifacts at the registered monitor root;
        # no copies, rerolls, altered scores or separate hidden outcomes.
        (out/'development120').symlink_to(FIT/'development120',target_is_directory=True)
        (out/'round1_development120').symlink_to(FIT/'development120',target_is_directory=True)
        atomic_json(out/'active_fit_cycle.json',{'cycle':str(active_fit),'round':round_number})
        stage='same_qualified_GTcrop_recipe_wait_RDT_child_exit_then_fit_and120';save()
        native.run(argparse.Namespace(output=FIT,wait_hours=max(.01,(deadline-time.monotonic())/3600)))
        first=json.loads((FIT/'development120/progress.json').read_text())
        baseline_path=BASE/'scene67_resume_parallel_r34/parallel_attempt/development120/progress.json'
        baseline=json.loads(baseline_path.read_text())
        if first['completed_trials']!=120 or baseline['completed_trials']!=120:raise ValueError('Full matched120 required for continuation decision')
        old_keys={(r['seed'],r['task']) for r in baseline['rows']};new_keys={(r['seed'],r['task']) for r in first['rows']}
        if len(old_keys)!=120 or new_keys!=old_keys:raise ValueError('Continuation cannot select a different test cohort')
        remaining=(deadline-time.monotonic())/3600
        second=allow_round2(first['successes'],baseline['successes'],remaining)
        atomic_json(out/'round1_continuation_decision.json',{'first_successes':first['successes'],'baseline_successes':baseline['successes'],
            'trials_each':120,'remaining_hours':remaining,'run_second_round':second,'decision_before_round2_outcomes':True,
            'rule':'strictly improves baseline, below60percent, at least8h left','not_a_statistical_improvement_claim':True})
        completed=[{'cycle':str(FIT),'successes':first['successes'],'trials':120}]
        if second:
            active_fit=FIT_ROUND2;round_number=2
            native.WARM=FIT/'train/checkpoints/010000/pretrained_model'
            atomic_json(out/'active_fit_cycle.json',{'cycle':str(active_fit),'round':round_number})
            (out/'round2_development120').symlink_to(active_fit/'development120',target_is_directory=True)
            next_link=out/'development120.next';next_link.symlink_to(active_fit/'development120',target_is_directory=True)
            next_link.replace(out/'development120')
            stage='conditional_second2500_new_updates_same_source_own_qualification';save()
            native.run(argparse.Namespace(output=active_fit,wait_hours=max(.01,(deadline-time.monotonic())/3600)))
            result=json.loads((active_fit/'development120/progress.json').read_text())
            if result['completed_trials']!=120 or {(r['seed'],r['task']) for r in result['rows']}!=old_keys:
                raise ValueError('Second full matched120 incomplete or cohort changed')
            completed.append({'cycle':str(active_fit),'successes':result['successes'],'trials':120})
        winner=max(completed,key=lambda r:r['successes'])
        atomic_json(out/'completed_recipe.json',{'same_GTcrop_recipe_completed':True,'completed_rounds':completed,
            'best_new_fit_on_DEVELOPMENT120':winner,'original_baseline_successes':baseline['successes'],
            'native20_original_and_all_failed_rounds_preserved':True,'no_further_unbounded_fit_or_model_ceiling_claim':True,
            'best_development_checkpoint_NOT_unbiased_final_paper_SR':True})
    except Exception as exc:
        stage='failed_priority_preserve_all_neural_artifacts';save('failed_preserve_artifacts_NOT_zero_SR',f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if lease is not None:
            if owned_fit_child_alive(out):
                atomic_json(out/'restore_deferred_to_watchdog.json',{'owned_inflight_Smol_child_preserved':True,
                    'supervisor_will_resume_after_that_child_exits':True})
            else:
                atomic_json(out/'supervisor_restored.json',{'restored_same_supervisor':restore(lease),
                    'no_RDT_neural_worker_interrupted':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--guard-root',type=Path);p.add_argument('--wait-hours',type=float,default=52)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.guard_root:guard(a.guard_root.resolve())
    elif a.output is None or not 0<a.wait_hours<=54:raise ValueError('Fresh finite priority run required')
    elif not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists() or FIT.exists():raise FileExistsError('No duplicate GTcrop fitting')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_smol_priority_r38','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'same_prepared_Smol_recipe_prioritized':True,'actual_neural_worker_signalled':False});print(child.pid)
