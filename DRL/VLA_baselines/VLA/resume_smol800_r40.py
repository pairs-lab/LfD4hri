"""Evidence-preserving800 continuation with one disposable worker per seed.

The retired R39 worker retained ~4GiB host RSS after unloading its GPU model,
then waited for5GiB available RAM that its own retained memory prevented.
This changes process isolation/admission ONLY: same locked800, checkpoint,
private noise, native10, images, state67, actions, physics and success. No
terminal trial is rerolled. A tiny coordinator CUDA reservation prevents a
waiting fit racing the gap between seed workers. No trainer is signalled.
"""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from VLA import evaluate_smol800_r39 as old
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.wait_render_then_resume_queues import identity,render_jobs
from VLA.audit_r33_failure_traces import inspect_episode

ROOT=old.ROOT;SIM=old.SIM;BASE=old.BASE


def verify_prefix(plan,rows):
    expected=[(r['seed'],r['task']['name']) for r in plan['trials']]
    actual=[(r['seed'],r['task']) for r in rows]
    if actual!=expected[:len(rows)] or len(set(actual))!=len(actual):raise ValueError('Terminal outcomes must be the exact original prefix')
    return expected[len(rows):]


def check_files(plan,full=False):
    for path,value in plan['frozen_file_identity'].items():
        s=Path(path).stat()
        if [s.st_ino,s.st_size,s.st_mtime_ns]!=value:raise ValueError('Frozen runtime/weights changed: '+path)
    if full and any(sha256_file(p)!=h for p,h in plan['frozen_files'].items()):raise ValueError('Frozen content digest changed')


def run_seed(a):
    import torch
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.recorder import VideoRecorder,NullVideoRecorder
    from VLA.benchmark import _interaction_audit,_save_observation_snapshot
    from VLA.smolvla.scene67_adapter import Scene67SmolVLA
    from VLA.simstate_scene64 import scene_state
    out=a.output.resolve();plan=json.loads((out/'plan.json').read_text());check_files(plan)
    progress=json.loads((out/'progress.json').read_text());prior=progress['rows'];verify_prefix(plan,prior)
    profile=json.loads(Path(plan['observation_profile_file']).read_text());policy=None;rows=[];proof=[]
    folder=out/f'test_seed{a.seed}';folder.mkdir(parents=True,exist_ok=False);traces={}
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
    def environment(seed):
        return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,control_hz=10,
            assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
            success_threshold_scale=plan['success']['reward_xml_threshold_scale'],semantic_success_thresholds=plan['success']['semantic_thresholds'])
    def attach(env,seed):
        nonlocal policy
        if policy is None:policy=Scene67SmolVLA(env,plan['checkpoint'],replan_steps=1,seed=seed)
        policy.env=env;policy.seed=seed;policy._load();policy.model.config.num_steps=10
    # First worker independently checks exact same-input policy numerics.
    if not (out/'actual_native10_resume_reproduction8.json').exists():
        env=environment(old.OLD_SEEDS[0]);rng=np.random.default_rng(old.OLD_SEEDS[0])
        ref=json.loads((BASE/'scene67_resume_parallel_r34/parallel_attempt/development120/progress.json').read_text())
        source=json.loads(Path(ref['rows'][0]['trajectory']).read_text())
        try:
            attach(env,old.OLD_SEEDS[0])
            for task in [old.randomize_rl_scene(t,rng) for t in old.select_tasks('all')]:
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    np.testing.assert_allclose(scene_state(env,obs),source[task.name][i]['pre_action_scene64'],rtol=0,atol=1e-6)
                    action=policy.predict(obs);np.testing.assert_allclose(action,source[task.name][i]['action'],rtol=0,atol=1e-6)
                    obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'):raise ValueError('Resume8 unexpectedly ended')
                    proof.append({'task':task.name,'step':i+1,'actual_native_action':action.tolist()})
        finally:env.close()
        atomic_json(out/'actual_native10_resume_reproduction8.json',{'passed':True,'actual_neural_controls':8,
            'not_new800_trials':True,'same_state_and_action_tolerance':1e-6,'rows':proof})
    env=environment(a.seed);preserved={r['task']:r for r in prior if r['seed']==a.seed};reset_proof=[]
    recorder=VideoRecorder(folder/'rollout.mp4',fps=10,overview_render=lambda:env.render('overview')) if a.seed==plan['seeds'][0] else NullVideoRecorder()
    try:
        attach(env,a.seed)
        with recorder:
            for item in [v for v in plan['trials'] if v['seed']==a.seed]:
                task=old.TaskSpec.from_dict(item['task']);obs=env.reset(task);policy.reset(task)
                if task.name in preserved:
                    row=preserved[task.name];source=json.loads(Path(row['trajectory']).read_text())[task.name]
                    np.testing.assert_allclose(scene_state(env,obs),source[0]['pre_action_scene64'],rtol=0,atol=1e-6)
                    reset_proof.append({'task':task.name,'initial64_matches_at1e6':True,'prior_controls_or_outcomes_rerolled':False})
                    continue
                if env.is_semantic_success():raise ValueError('Initially satisfied scene; no replacement')
                _save_observation_snapshot(folder,task.name,obs,env);trace=[];done=False
                with (folder/(task.name+'_applied_controls.jsonl')).open('x') as journal:
                    while not done:
                        before=_interaction_audit(env);action=policy.predict(obs)
                        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Malformed native VLA output')
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
                        if obs.step%25==0:atomic_json(folder/'worker_status.json',{'pid':os.getpid(),'task':task.name,'step':obs.step,'completed_new_rows':len(rows)})
                    os.fsync(journal.fileno())
                _save_observation_snapshot(folder,task.name,obs,env);traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                rows.append({'seed':a.seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                    'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                atomic_json(folder/'seed_result.json',{'status':'running','rows':rows,'preserved_reset_checks':reset_proof})
        atomic_json(folder/'seed_result.json',{'status':'completed_seed','rows':rows,'preserved_reset_checks':reset_proof,
            'NN_weights_solver_input_physics_success_unchanged':True})
    finally:env.close()


def run(a):
    source=a.source.resolve();out=a.output.resolve();pause=json.loads((source/'MEMORY_GUARD_PAUSED_PRE_TASK_r40.json').read_text())
    source_plan=json.loads((source/'plan.json').read_text());progress=json.loads((source/'progress.json').read_text());rows=list(progress['rows'])
    if sha256_file(source/'progress.json')!=pause['progress_sha256']:raise ValueError('Preserved terminal outcomes changed')
    verify_prefix(source_plan,rows);check_files(source_plan,full=True)
    before=identity(pause['pid'])
    if (not before or before['state']!='T' or before['start_ticks']!=pause['start_ticks'] or before['children']
            or b'VLA.evaluate_smol800_r39' not in before['tokens'] or str(source).encode() not in before['tokens']):
        raise ValueError('Takeover requires exact paused pre-task waiter, not a trainer/episode')
    lease=old.held_priority_lease();gpu=old.resources()
    if not lease or not {int(p) for p in gpu['compute_pids']}<={pause['pid'],lease['trainer_pid']}:
        raise ValueError('Existing owned resource lease changed; no worker terminated')
    out.mkdir(parents=True,exist_ok=False);plan={**source_plan,'resumed_from':str(source),
        'preserved_terminal_trials':len(rows),'original_plan_sha256':sha256_file(source/'plan.json'),
        'original_progress_sha256':pause['progress_sha256'],'execution_change_ONLY':'one disposable NN/simulator worker per seed; tiny non-neural CUDA gap reservation',
        'same_checkpoint_solver_input_success_and800_cohort':True,'frozen_files':dict(source_plan['frozen_files']),
        'frozen_file_identity':dict(source_plan['frozen_file_identity'])}
    code=Path(__file__).resolve();plan['frozen_files'][str(code)]=sha256_file(code);s=code.stat()
    plan['frozen_file_identity'][str(code)]=[s.st_ino,s.st_size,s.st_mtime_ns]
    atomic_json(out/'plan.json',plan)
    # Reserve a negligible CUDA context BEFORE freeing the old context, so the
    # already-waiting GTcrop job cannot steal the inter-process gap.
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(64*2**20/torch.cuda.get_device_properties(0).total_memory)
    reservation=torch.empty(1,device='cuda')
    same=identity(pause['pid'])
    if not same or same['start_ticks']!=before['start_ticks'] or same['state']!='T':raise ValueError('Source waiter identity changed')
    os.kill(pause['pid'],signal.SIGTERM);os.kill(pause['pid'],signal.SIGCONT)
    until=time.monotonic()+10
    while identity(pause['pid']) is not None:
        if time.monotonic()>until:raise TimeoutError('Exact source waiter did not retire')
        time.sleep(.1)
    atomic_json(source/'RESUMED_43_TERMINAL_OUTCOMES_r40.json',{'new_cycle':str(out),'new_pid':os.getpid(),
        'retired_only_exact_paused_pre_task_waiter':pause['pid'],'completed_terminal_trials_preserved':len(rows),
        'trainer_or_active_neural_episode_signalled':False,'same800_cohort_checkpoint_actions_scoring':True})
    original_status=json.loads((source/'status.json').read_text())
    atomic_json(source/'status.json',{**original_status,'status':'superseded_pre_task_host_memory_isolated_resume','new_cycle':str(out)})
    atomic_json(BASE/'Smol_800_r39_active.json',{'cycle':str(out),'pid':os.getpid(),'checkpoint':plan['checkpoint'],
        'planned_trials':800,'planned_per_action':200,'preserved_terminal_trials':len(rows),'same_locked_new_seed_cohort':True})
    first=source/f'test_seed{plan["seeds"][0]}'
    if first.exists():(out/first.name).symlink_to(first,target_is_directory=True)
    cases=[]
    for row in rows:
        trace=json.loads(Path(row['trajectory']).read_text())[row['task']];cases.append(inspect_episode(row,trace))
    atomic_json(out/'preserved_terminal_evidence.json',{'rows':rows,'prior_controls_or_outcomes_rerolled':False,
        'source_progress_sha256':pause['progress_sha256'],'source_trajectories_sha256':{r['trajectory']:sha256_file(r['trajectory']) for r in rows}})
    stage='between_seed_worker_admission';error=None;active=None;active_seed=None
    deadline=time.monotonic()+min(a.wait_hours*3600,original_status['deadline_remaining_seconds']);stats=None
    if len(rows)>=4:stats=old.success_statistics(rows[:len(rows)//4*4])
    def save(status='running',report=False):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'completed_trials':len(rows),'planned_trials':800,
            'planned_per_action':200,'active_seed':active_seed,'active_child_pid':active.pid if active and active.poll() is None else None,
            'inference_steps':10,'frozen_checkpoint':plan['checkpoint'],'preserved_terminal_trials':plan['preserved_terminal_trials'],
            'allocator_reservation_MiB':64,'other_trainer_or_episode_signalled':False,'deadline_remaining_seconds':max(0,deadline-time.monotonic())})
        atomic_json(out/'progress.json',{'status':status,'model':'SmolVLA_scene67_native10_frozen800','checkpoint':plan['checkpoint'],
            'rows':rows,'completed_trials':len(rows),'planned_trials':800,'planned_per_action':200,
            'successes':sum(r['success'] for r in rows),'by_action':old.counts(rows),'statistics_complete_seed_blocks':stats,
            'error':error,'preserved_terminal_trials':plan['preserved_terminal_trials'],'same_original800_plan':True,
            'new_800_evaluation_only_NOT_training':True,'privileged_inputs':True,'additional_action_shield':False})
        if report:old.write_report(out,plan,rows,status,error,cases,stats)
    save(report=True)
    try:
        for seed in plan['seeds']:
            if sum(r['seed']==seed for r in rows)==4:continue
            active_seed=seed;stage='waiting_RAM_before_fresh_seed_worker';save()
            while True:
                if time.monotonic()>deadline:raise TimeoutError('Original finite800 budget exhausted; partial evidence retained')
                check_files(plan);current=old.held_priority_lease();gpu=old.resources()
                known={os.getpid()}
                if current:known.add(current['trainer_pid'])
                if (current and {int(p) for p in gpu['compute_pids']}<=known and gpu['free_mib']>=5120
                        and gpu['available_RAM_MiB']>=5120 and not render_jobs()):break
                save();time.sleep(15)
            command=[str(SIM),'-m','VLA.resume_smol800_r40','--output',str(out),'--seed',str(seed)]
            with (out/f'seed{seed}_worker.log').open('x') as log:
                active=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                    'MALLOC_ARENA_MAX':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'},
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(out/'active_child.json',{'pid':active.pid,'seed':seed,'command':command})
                stage='actual_frozen_native10_seed_worker';save()
                def harvest():
                    nonlocal stats
                    result=out/f'test_seed{seed}/seed_result.json'
                    if not result.exists():return
                    data=json.loads(result.read_text());seen={(r['seed'],r['task']) for r in rows}
                    for row in data['rows']:
                        if (row['seed'],row['task']) in seen:continue
                        verify_prefix(plan,[*rows,row]);trace=json.loads(Path(row['trajectory']).read_text())[row['task']]
                        if len(trace)!=row['steps'] or bool(trace[-1]['success'])!=row['success']:raise ValueError('Terminal child evidence mismatch')
                        rows.append(row);seen.add((row['seed'],row['task']));cases.append(inspect_episode(row,trace))
                        if len(rows)%4==0:stats=old.success_statistics(rows)
                        save(report=True)
                while active.poll() is None:harvest();save();time.sleep(5)
                harvest()
                if active.returncode:raise RuntimeError(f'Seed worker{seed} exited{active.returncode}; NOT counted as task failure or rerolled')
            if sum(r['seed']==seed for r in rows)!=4:raise ValueError('Seed worker ended before all4 terminal outcomes')
            # The model, PyBullet world and every native allocator exit here.
            # The coordinator has never loaded NN weights or simulator worlds.
            active=None;active_seed=None;gc.collect();save()
        if len(rows)!=800 or any(v['trials']!=200 for v in old.counts(rows).values()):raise ValueError('Requested800 not complete')
        check_files(plan,full=True);stage='completed_actual800_process_isolated_same_locked_policy';save('completed800',report=True)
        atomic_json(out/'completion_evidence.json',{'actual_neural_trials':800,'each_action':200,
            'preserved_terminal_trials':plan['preserved_terminal_trials'],'policy_or_seed_or_scorer_changed':False,
            'progress_sha256':sha256_file(out/'progress.json'),'all_original_failed_cases_retained':True})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';stage='incomplete_preserve_actual800_not_zero_fill';save('incomplete_preserve_actual_trials',report=True);raise
    finally:
        if active is not None and active.poll() is None:
            # Preserve an already-running episode; do not release our gap
            # reservation until that own bounded seed worker actually exits.
            while active.poll() is None:time.sleep(5)
        del reservation


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--seed',type=int);p.add_argument('--wait-hours',type=float,default=34)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.seed is not None:run_seed(a)
    elif not a.source or not 0<a.wait_hours<=36:raise ValueError('Frozen source and finite resume budget required')
    elif not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate800 continuation')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.resume_smol800_r40','--source',str(a.source.resolve()),
                '--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MALLOC_ARENA_MAX':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'same800_preserve_terminal_outcomes':True,'process_isolation_only':True});print(child.pid)
