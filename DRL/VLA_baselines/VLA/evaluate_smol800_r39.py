"""200 NEW fixed seeds x four tasks =800 actual neural SmolVLA episodes.

One checkpoint/native10 is locked BEFORE results. No merging old120, action
shield, input clipping, expert fallback, outcome selection or partial-as-final
report. Safe GPU admission uses the existing exact capped RDT + Smol-priority
supervisor lease; no other worker is signalled. All counts/failures and one
complete four-task video are retained. This evaluates the CURRENT fitted
checkpoint, not a promised future finetune.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks,randomize_rl_scene,TaskSpec
from VLA.common.evaluation_statistics import wilson_interval,success_statistics
from VLA.run_cross_vla120 import ROOT,PROFILE_PROTOCOL,SEEDS as OLD_SEEDS
from VLA.run_flow20_parallel_r37 import resources,allowed,OWNER
from VLA.run_flow20_extension_r36 import parent_roots,alive
from VLA.wait_render_then_resume_queues import identity,render_jobs
from VLA.audit_training_readiness import physical_scene_key

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
CHECKPOINT=ROOT/'VLA/smolvla_scene67/automatic_full_gate_fit_v2/train/checkpoints/010000/pretrained_model'
TRAIN=ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'
VALIDATION=TRAIN.parent/'validation_candidate'
TASKS=('touch','pick','move','pick_place')
NEW_SEEDS=list(range(20269001,20269201))
PREREQUISITES=(BASE/'Smol_contactstop_closedloop_r37',BASE/'Smol_bounded_feedback40_r38')


def trial_specs(seeds=NEW_SEEDS):
    result=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        result.extend({'seed':seed,'task':randomize_rl_scene(t,rng).to_dict()} for t in select_tasks('all'))
    return result


def counts(rows):
    if len({(r['seed'],r['task']) for r in rows})!=len(rows):raise ValueError('Duplicate episode outcome')
    if any(r['task'] not in TASKS for r in rows):raise ValueError('Unknown action')
    result={}
    for task in TASKS:
        selected=[r for r in rows if r['task']==task];n=len(selected);k=sum(bool(r['success']) for r in selected)
        result[task]={'trials':n,'successes':k,'success_rate':k/n if n else None,
            'wilson_95_interval':wilson_interval(k,n) if n else None,
            'safety_aborts':sum(bool(r.get('safety_violation')) for r in selected)}
    return result


def held_priority_lease():
    """A prior owned lease, not a new pause/permission inferred from idle VRAM."""
    try:
        pointer=json.loads((BASE/'Smol_GTcrop_motion_r36_active.json').read_text());root=Path(pointer['cycle'])
        if root.parent!=BASE or not root.name.startswith('Smol_GTcrop_motion_r36_priority_r38'):return None
        lease=json.loads((root/'supervisor_lease.json').read_text())
        owner=identity(lease['priority_owner_pid']);supervisor=identity(lease['supervisor_pid'])
        if (not owner or not supervisor or owner['cwd']!=str(ROOT) or supervisor['cwd']!=str(ROOT)
                or owner['start_ticks']!=lease['priority_owner_start_ticks']
                or supervisor['start_ticks']!=lease['supervisor_start_ticks'] or supervisor['state']!='T'
                or b'VLA.run_smol_priority_r38' not in owner['tokens'] or str(root).encode() not in owner['tokens']
                or b'VLA.run_all_privileged_r33' not in supervisor['tokens'] or str(OWNER).encode() not in supervisor['tokens']):return None
        return {'priority_root':str(root),'supervisor_pid':lease['supervisor_pid'],
            'trainer_pid':lease['trainer_pid_NOT_signalled'],'priority_owner_pid':lease['priority_owner_pid']}
    except (OSError,KeyError,ValueError):return None


def admitted(lease, state, child, process, gpu, own_pid=None):
    pids={int(p) for p in gpu['compute_pids']}
    own=set() if own_pid is None else {own_pid}
    if lease is not None:
        # Initial admission ONLY while the exact trainer is visibly holding
        # CUDA. An empty gap could race the waiting GTcrop fit. Once our NN
        # is already resident, its own CUDA PID safely holds the same gap.
        if pids<=own:return bool(own_pid is not None and own_pid in pids
            and gpu['free_mib']>=5120 and gpu['available_RAM_MiB']>=5120)
        if child.get('pid')!=lease['trainer_pid']:return False
        stripped={**gpu,'compute_pids':list(pids-own)}
        return allowed(state,child,process,stripped)
    return False


def initialize(out):
    from VLA.common.observation_profile import profile_hash
    from VLA.common.profile_binding import checkpoint_profile
    from VLA.common.runtime_provenance import environment_runtime_fingerprint
    protocol=json.loads(PROFILE_PROTOCOL.read_text());profile=checkpoint_profile(CHECKPOINT)
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Locked same environment/profile required')
    reserved_path=ROOT/protocol['reserved_final_protocol'];reserved=json.loads(reserved_path.read_text())
    if set(NEW_SEEDS)&set(OLD_SEEDS+reserved['final_test_seeds']):raise ValueError('Do not reopen development/reserved seed cohorts')
    specs=trial_specs()
    if len(specs)!=800 or Counter(v['task']['name'] for v in specs)!={t:200 for t in TASKS}:raise ValueError('Exactly200/action required')
    excluded=set()
    for source in (TRAIN,VALIDATION):
        excluded.update(physical_scene_key(r['task']) for r in json.loads((source/'index.json').read_text()))
    previous=trial_specs(OLD_SEEDS);excluded.update(physical_scene_key(v['task']) for v in previous)
    keys=[physical_scene_key(v['task']) for v in specs]
    if len(set(keys))!=800 or set(keys)&excluded:raise ValueError('New physical scenes must not overlap TRAIN/validation/old120')
    runtime=environment_runtime_fingerprint(profile)
    trainer_impl=ROOT/'VLA/rdt/train_scene67.py';trainer_owner=json.loads((OWNER/'status.json').read_text())
    if ('8500*2**20/' not in trainer_impl.read_text()
            or trainer_owner['frozen_files'][str(trainer_impl)]!=sha256_file(trainer_impl)):
        raise ValueError('Exact already-qualified RDT allocator/source identity required')
    paths=[Path(__file__),PROFILE_PROTOCOL,Path(protocol['observation_profile_file']),reserved_path,
        CHECKPOINT/'model.safetensors',CHECKPOINT/'config.json',CHECKPOINT/'scene67_contract.json',
        CHECKPOINT/'fp32_master_contract.json',CHECKPOINT/'policy_preprocessor.json',CHECKPOINT/'policy_postprocessor.json',
        CHECKPOINT/'policy_preprocessor_step_5_normalizer_processor.safetensors',
        CHECKPOINT/'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
        TRAIN/'index.json',VALIDATION/'index.json',ROOT/'VLA/smolvla/scene67_adapter.py',ROOT/'VLA/smolvla/adapter.py',
        ROOT/'VLA/smolvla/precision.py',ROOT/'VLA/scene64_conditioning.py',ROOT/'VLA/simstate_scene64.py',
        ROOT/'VLA/audit_r33_failure_traces.py',ROOT/'VLA/common/evaluation_statistics.py',ROOT/'VLA/common/tasks.py',
        ROOT/'VLA/benchmark.py',ROOT/'VLA/common/recorder.py',ROOT/'asset/params/reward.xml',ROOT/'utils/camera.py',ROOT/'config.py',
        ROOT/'VLA/common/action.py',ROOT/'VLA/run_flow20_parallel_r37.py',ROOT/'VLA/wait_render_then_resume_queues.py',trainer_impl]
    paths.extend(ROOT/name for name in runtime['sources_sha256'])
    from config import PICK_TARGETS,PLACE_TARGETS
    from VLA.common.spatial import FIXED_DESTINATION
    asset_types={name.split(' ',1)[1] for name in {*PICK_TARGETS,*PLACE_TARGETS} if name not in FIXED_DESTINATION}
    asset_roots=[ROOT/'asset'/name for name in ('ur5e','robotiq_2f_85',*sorted(asset_types-{'block','egg'}))]
    paths.extend(p for root in asset_roots for p in sorted(root.rglob('*')) if p.is_file())
    paths.append(ROOT/'asset/plane.urdf')
    paths=list(dict.fromkeys(paths));frozen={str(p.resolve()):sha256_file(p) for p in paths}
    stats={str(p.resolve()):[p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
    out.mkdir(parents=True,exist_ok=False)
    versions={}
    for name in ('torch','transformers','lerobot','numpy','pybullet','safetensors'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not_in_distribution_metadata'
    plan={'scope':'user_requested_200_NEW_episodes_per_action_one_frozen_CURRENT_SmolVLA',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(CHECKPOINT),'model':'SmolVLA+privileged_scene67',
        'planned_trials':800,'planned_trials_per_action':200,'seeds':NEW_SEEDS,'trials':specs,'tasks':TASKS,
        'inference_steps':10,'replan_steps':1,'robot':'ur5e','gripper':'articulated Robotiq2F85','image_size':384,
        'control_hz_SIMULATION_not_realtime':10,'success':protocol['success'],'environment_runtime_fingerprint':runtime,
        'observation_profile_sha256':profile_hash(profile),'observation_profile_file':protocol['observation_profile_file'],
        'weights_and_solver_locked_before_any_new_outcomes':True,'no_prior120_trials_reused':True,
        'all800_physical_scene_keys_unique_and_disjoint_TRAIN_val_old120':True,'training_excluded_scene_keys':len(excluded),
        'new_fixed_cohort_not_the_preexisting_reserved160':True,'existing_reserved_final_test_used':False,
        'this800_evaluation_only_never_training_source':True,'final_reporting_cohort_for_THIS_locked_checkpoint':True,
        'privileged_causal64_via_numeric67':True,'privileged_GT_crop':False,'input_bound_or_action_shield':False,
        'body_grasp_assist':'same existing task-independent bilateral-pad carrier; NOT pure contact-only hardware physics',
        'high_level_actions':'native neural VLA every control; existing common low-level IK servo only',
        'oracle_planner_or_recovery_fallback':False,'failure_or_seed_rerolls':0,'video_seeds':[NEW_SEEDS[0]],
        'target_counts_per_action':{t:dict(Counter(v['task']['pick_name'] for v in specs if v['task']['name']==t)) for t in TASKS},
        'frozen_files':frozen,'frozen_file_identity':stats,'dependency_versions':versions,
        'resource_policy':'after existing40+40 release GPU, capped3GiB alongside exact RDT under existing held supervisor lease, then exclusive; no worker signalled'}
    atomic_json(out/'plan.json',plan)
    atomic_json(BASE/'Smol_800_r39_active.json',{'cycle':str(out),'pid':os.getpid(),'checkpoint':str(CHECKPOINT),
        'planned_trials':800,'planned_per_action':200,'new_fixed_seed_cohort':True})
    return plan,profile


def write_report(out,plan,rows,status,error=None,failure_cases=None,stats=None):
    summary=counts(rows);complete=len(rows)==800 and all(v['trials']==200 for v in summary.values())
    if status=='completed800' and not complete:raise ValueError('Cannot report partial as completed800')
    title='COMPLETED 800' if complete else f'INCOMPLETE: {len(rows)}/800'
    lines=[f'# SmolVLA — {title}','',f'Frozen checkpoint: `{plan["checkpoint"]}`.',
        'Native10; privileged causal scene67; no GT target crop, input bounding, action shield or oracle fallback.',
        'Same UR5e + Robotiq2F85, objects, cameras, simulation10Hz, success thresholds and task horizons.',
        'The existing bilateral-pad grasp carrier/assist is enabled and must be disclosed.',
        '200 NEW fixed seeds ×4 tasks. Old120 are not added. Partial results are not a final SR800.','',
        '| Action | Success / completed | Planned | SR | Wilson95% CI | Safety failures |',
        '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        r=summary[task];ci=r['wilson_95_interval']
        rate=f'{100*r["success_rate"]:.2f}%' if r['trials'] else 'pending'
        interval=f'{100*ci[0]:.2f}–{100*ci[1]:.2f}%' if ci else 'pending'
        lines.append(f'| {task} | {r["successes"]}/{r["trials"]} |200| {rate} | {interval} | {r["safety_aborts"]} |')
    total=sum(r['success'] for r in rows)
    if rows:lines.extend(['',f'Observed aggregate: **{total}/{len(rows)} = {100*total/len(rows):.2f}%**.'
        +(' Balanced macro-average equals the aggregate because every task has200 trials.' if complete else ' INCOMPLETE, not the completed SR800.')])
    if stats and complete:
        ci=stats['seed_block_bootstrap_95_interval'];lines.extend(['',f'Seed-block bootstrap95% interval: {100*ci[0]:.2f}–{100*ci[1]:.2f}% (200 seed blocks).'])
    if failure_cases:
        symptoms=Counter()
        for c in failure_cases:
            if c['success']:continue
            if c['safety_violation']:symptoms['physical_safety_abort']+=1
            if c['task']!='touch' and not c['ever_correct_hold']:symptoms['never_verified_correct_hold']+=1
            if c['ever_wrong_hold']:symptoms['ever_wrong_object_held']+=1
            for key,value in c['overlapping_diagnostic_counts_NOT_causal_findings'].items():
                if value:symptoms[key]+=1
        atomic_json(out/'failure_analysis.json',{'completed_trials':len(rows),'overlapping_failure_episode_counts':dict(symptoms),
            'cases':failure_cases,'symptoms_NOT_proven_model_or_setup_causes':True,'all_failures_retained':True})
        lines.extend(['','## Observed failure symptoms (overlapping, not proven causes)',''])
        lines.extend(f'- {name}: {n} failed episodes.' for name,n in symptoms.most_common())
    if error:lines.extend(['','Execution error (NOT counted as a new task failure):',f'`{error}`'])
    lines.extend(['','A safety abort counts as failure. Infrastructure errors stop with partial evidence; no zero-filled or replacement episodes.',
        f'First predeclared seed raw video: `test_seed{NEW_SEEDS[0]}/rollout.mp4` (all4 tasks, including failures).',
        'All per-episode outcomes: `episodes.csv`; summary: `success_rates.csv`; exact plan/profile/weights: `plan.json`.',
        'Fresh800 outcomes are evaluation-only, not supplied to any active finetune.'])
    temporary=out/'REPORT.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(out/'REPORT.md')
    with (out/'success_rates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['action','successes','completed','planned','success_rate','wilson95_low','wilson95_high','safety_aborts'])
        for t in TASKS:
            v=summary[t];w.writerow([t,v['successes'],v['trials'],200,v['success_rate'],*(v['wilson_95_interval'] or [None,None]),v['safety_aborts']])
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','task','success','steps','safety_violation','trajectory']);w.writeheader()
        w.writerows({k:r.get(k) for k in w.fieldnames} for r in rows)


def run(a):
    out=a.output.resolve();plan,profile=initialize(out);rows=[];cases=[];policy=None;active_trial=None
    stage='waiting_existing_Smol_diagnostics_and_held_priority_capacity';error=None;deadline=time.monotonic()+a.wait_hours*3600
    stats=None
    def save(status='running',report=False):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'completed_trials':len(rows),'planned_trials':800,
            'planned_per_action':200,'active_trial':active_trial,'inference_steps':10,'frozen_checkpoint':str(CHECKPOINT),
            'other_worker_signalled':False,'deadline_remaining_seconds':max(0,deadline-time.monotonic())})
        atomic_json(out/'progress.json',{'status':status,'model':'SmolVLA_scene67_native10_frozen800','checkpoint':str(CHECKPOINT),
            'rows':rows,'completed_trials':len(rows),'planned_trials':800,'planned_per_action':200,'successes':sum(r['success'] for r in rows),
            'by_action':counts(rows),'statistics_complete_seed_blocks':stats,'active_trial':active_trial,'error':error,
            'new_800_evaluation_only_NOT_training':True,'no_prior120_reused':True,'privileged_inputs':True,
            'additional_action_shield':False,'existing_reserved_final_test_used':False})
        if report:write_report(out,plan,rows,status,error,cases,stats)
    def check(full=False):
        if time.monotonic()>deadline:raise TimeoutError('Finite800 evaluation budget; actual terminal trials retained')
        for path,fingerprint in plan['frozen_file_identity'].items():
            p=Path(path);s=p.stat()
            if [s.st_ino,s.st_size,s.st_mtime_ns]!=fingerprint:raise ValueError('Frozen checkpoint/runtime changed: '+path)
        if full and any(sha256_file(p)!=h for p,h in plan['frozen_files'].items()):raise ValueError('Frozen source/weight digest mismatch')
    save(report=True)
    try:
        while True:
            check();pending=[]
            for root in PREREQUISITES:
                state=json.loads((root/'status.json').read_text())
                terminal=state.get('status')=='completed' or str(state.get('status','')).startswith('failed')
                if not terminal or identity(state.get('pid',0)):pending.append(root.name)
            if not pending:break
            save();time.sleep(15)
        def wait_capacity():
            nonlocal stage
            stage='waiting_verified_RDT_Smol_priority_lease_or_all_queues_finished';save()
            while True:
                check();gpu=resources();state=json.loads((OWNER/'status.json').read_text());child=json.loads((OWNER/'active_child.json').read_text())
                lease=held_priority_lease()
                shared=admitted(lease,state,child,identity(child['pid']),gpu)
                exclusive=(not any(alive(p) for p in parent_roots()) and not gpu['compute_pids']
                    and gpu['free_mib']>=14500 and gpu['available_RAM_MiB']>=5120)
                if (shared or exclusive) and not render_jobs():
                    atomic_json(out/'latest_resource_admission.json',{'resources':gpu,'existing_priority_lease':lease,
                        'own_allocator_cap_MiB':3072,'neural_trainer_or_other_worker_signalled':False})
                    return
                save();time.sleep(15)
        wait_capacity()
        import torch
        from VLA.common.environment_factory import environment_class_for_profile
        from VLA.common.recorder import VideoRecorder,NullVideoRecorder
        from VLA.benchmark import _interaction_audit,_save_observation_snapshot
        from VLA.smolvla.scene67_adapter import Scene67SmolVLA
        from VLA.simstate_scene64 import scene_state
        from VLA.audit_r33_failure_traces import inspect_episode
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        def environment(seed):
            return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=plan['success']['reward_xml_threshold_scale'],semantic_success_thresholds=plan['success']['semantic_thresholds'])
        def attach(env,seed):
            nonlocal policy
            if policy is None:policy=Scene67SmolVLA(env,str(CHECKPOINT),replan_steps=1,seed=seed)
            policy.env=env;policy.seed=seed;policy._load();policy.model.config.num_steps=10
        def compatible():
            gpu=resources();state=json.loads((OWNER/'status.json').read_text());child=json.loads((OWNER/'active_child.json').read_text())
            lease=held_priority_lease()
            return admitted(lease,state,child,identity(child['pid']),gpu,os.getpid()) or (
                {int(p) for p in gpu['compute_pids']}<={os.getpid()} and not any(alive(p) for p in parent_roots()))
        # Eight real current-state/native-command checks are NOT800 test episodes.
        stage='actual_native10_baseline_reproduction8';save();proof=[];env=environment(OLD_SEEDS[0])
        reference_root=BASE/'scene67_resume_parallel_r34/parallel_attempt/development120'
        ref_progress=json.loads((reference_root/'progress.json').read_text())
        reference=json.loads(Path(ref_progress['rows'][0]['trajectory']).read_text())
        try:
            attach(env,OLD_SEEDS[0]);rng=np.random.default_rng(OLD_SEEDS[0])
            for task in [randomize_rl_scene(t,rng) for t in select_tasks('all')]:
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    np.testing.assert_allclose(scene_state(env,obs),reference[task.name][i]['pre_action_scene64'],rtol=0,atol=1e-6)
                    action=policy.predict(obs);np.testing.assert_allclose(action,reference[task.name][i]['action'],rtol=0,atol=1e-6)
                    obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'):raise ValueError('Actual8 ended unexpectedly')
                    proof.append({'task':task.name,'step':i+1,'actual_neural_action':action.tolist()})
        finally:env.close()
        atomic_json(out/'actual_native10_reproduction8.json',{'passed':True,'actual_neural_controls':8,'state_action_tolerance':1e-6,
            'not_counted_in800':True,'rows':proof,'allocator_peak_MiB':torch.cuda.max_memory_allocated()/2**20})
        for seed in NEW_SEEDS:
            check();env=environment(seed);folder=out/f'test_seed{seed}';folder.mkdir();traces={}
            recorder=VideoRecorder(folder/'rollout.mp4',fps=10,overview_render=lambda:env.render('overview')) if seed==NEW_SEEDS[0] else NullVideoRecorder()
            try:
                with recorder:
                    for item in [v for v in plan['trials'] if v['seed']==seed]:
                        task=TaskSpec.from_dict(item['task']);check()
                        if policy is not None and (render_jobs() or not compatible()):
                            policy.close();policy=None;gc.collect();torch.cuda.empty_cache()
                        if policy is None:wait_capacity()
                        attach(env,seed);obs=env.reset(task);policy.reset(task);trace=[];done=False
                        active_trial={'seed':seed,'task':task.name,'step':0};stage='actual_frozen_native10_200_per_action800';save()
                        if env.is_semantic_success():raise ValueError('Initially satisfied scene; preserve invalid audit, no replacement seed')
                        _save_observation_snapshot(folder,task.name,obs,env)
                        with (folder/(task.name+'_applied_controls.jsonl')).open('x') as journal:
                            while not done:
                                before=_interaction_audit(env);action=policy.predict(obs)
                                if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Malformed native neural output')
                                raw=policy.last_input['scene64'].copy();numeric=policy.last_input['numeric67'].copy()
                                obs,_,done,info=env.step(action)
                                frame={'task':task.name,'step':obs.step,'action':action.tolist(),
                                    'executed_canonical_action':info.get('executed_canonical_action'),
                                    'action_source':'native_neural_VLA_NO_action_repair','pre_action_scene64':raw.tolist(),
                                    'actual_numeric67_input':numeric.tolist(),'before':before,'after':_interaction_audit(env),
                                    'success':bool(info['success']),'safety_violation':info.get('safety_violation'),
                                    'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback')}
                                trace.append(frame);journal.write(json.dumps(frame,allow_nan=False)+'\n');journal.flush()
                                recorder.add(obs,'SmolVLA+scene67 native10 | '+task.prompt)
                                if obs.step%25==0:
                                    active_trial={'seed':seed,'task':task.name,'step':obs.step};save()
                            os.fsync(journal.fileno())
                        _save_observation_snapshot(folder,task.name,obs,env)
                        traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                        row={'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                            'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())}
                        rows.append(row);cases.append(inspect_episode(row,trace));active_trial=None
                        if len(rows)%4==0:stats=success_statistics(rows)
                        save(report=True)
            finally:env.close()
        if len(rows)!=800 or any(v['trials']!=200 for v in counts(rows).values()):raise ValueError('Incomplete requested800')
        check(full=True);stage='completed_actual800_200_per_action_no_rerolls';save('completed800',report=True)
        atomic_json(out/'completion_evidence.json',{'actual_neural_trials':800,'each_action':200,'weights_runtime_hashes_match':True,
            'no_prior_trials_adopted':True,'fixed_seed_plan_sha256':sha256_file(out/'plan.json'),
            'progress_sha256':sha256_file(out/'progress.json'),'all_safety_failures_included':True})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';stage='stopped_preserve_actual800_evidence_NOT_zero_fill';save('incomplete_preserve_actual_trials',report=True);raise
    finally:
        if policy is not None:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=36);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=40:raise ValueError('Finite user deadline budget required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if marker.exists() or a.output.exists():raise FileExistsError('No duplicate/rerolled800 run')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.evaluate_smol800_r39','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'requested_per_action':200,'requested_total':800,'other_workers_signalled':False});print(child.pid)
