"""Separate actual closed-loop VLA+contact-safety diagnostic, not plain SR.

No retraining, oracle target controller or IK planner. Native neural outputs
are recorded before an explicitly declared task-independent boundary shield.
Original unshielded40/120 remain untouched. All10 fixed seeds/all4 tasks run.
Only exact capped RDT spare capacity or post-queue exclusive GPU is allowed.
"""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import ROOT,SEEDS,PROFILE_PROTOCOL
from VLA.run_flow20_parallel_r37 import resources,allowed,OWNER
from VLA.run_flow20_extension_r36 import parent_roots,alive,PILOT
from VLA.wait_render_then_resume_queues import identity,render_jobs
from VLA.audit_contact_stop_candidate_r37 import project

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
CALIBRATION=BASE/'contact_stop_calibration_r37/audit.json'
PREVIOUS=BASE/'Smol_native20_extension_r37'
VARIANT='SmolVLA_native20_simscene67_plus_task_independent_contact_boundary_shield_r37'


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);stage='waiting_existing_native20_120_and_contact_calibration';rows=[];error=None;policy=None
    deadline=time.monotonic()+a.wait_hours*3600
    def save(status='running'):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'variant':VARIANT,
            'completed_neural_trials':len(rows),'planned_neural_trials':40,'other_neural_job_signalled':False,
            'original_unshielded_SR_not_overwritten':True,'new_SR_not_established':len(rows)!=40})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite between-task development deadline')
    save()
    try:
        while True:
            check();prior=json.loads((PREVIOUS/'status.json').read_text())
            if str(prior['status']).startswith('failed'):raise ValueError('Prior unshielded120 failed; preserve evidence before new variant')
            if prior['status']=='completed' and CALIBRATION.exists():break
            save();time.sleep(30)
        cal=json.loads(CALIBRATION.read_text())
        if (not cal['passed_execution'] or cal['TRAIN_demo_paths']!=76 or cal['TRAIN_expert_successes_NOT_VLA_SR']!=76
                or cal['TRAIN_paths_with_clamped_commands']!=0):raise ValueError('Broad unchanged TRAIN-demo compatibility required')
        for path,digest in cal['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Exact calibrated contact rule/source changed')
        source=json.loads((PILOT/'progress.json').read_text());cp=Path(source['checkpoint'])
        if source['completed_trials']!=40:raise ValueError('Complete matched unshielded40 required')
        files=[Path(__file__),ROOT/'VLA/audit_contact_stop_candidate_r37.py',ROOT/'VLA/run_flow20_parallel_r37.py',
            ROOT/'VLA/smolvla/scene67_adapter.py',ROOT/'VLA/smolvla/adapter.py',CALIBRATION,cp/'model.safetensors',PROFILE_PROTOCOL]
        bound={str(p):sha256_file(p) for p in files}
        atomic_json(out/'plan.json',{'variant':VARIANT,'checkpoint':str(cp),'seeds':SEEDS[:10],
            'planned_trials':40,'selection_before_outcomes':True,'paired_unshielded_baseline':str(PILOT/'progress.json'),
            'neural_commands_required_every_step':True,'contact_shield_additional_NOT_plain_VLA':True,
            'target_goal_instruction_or_expert_phase_used_by_shield':False,'final_test_used':False,'bound_files':bound})
        def wait_capacity():
            nonlocal stage
            stage='waiting_capped_RDT_spare_capacity_or_postqueue_exclusive';save()
            while True:
                check();gpu=resources();state=json.loads((OWNER/'status.json').read_text());child=json.loads((OWNER/'active_child.json').read_text())
                shared=allowed(state,child,identity(child['pid']),gpu)
                exclusive=(not any(alive(p) for p in parent_roots()) and not gpu['compute_pids']
                    and gpu['free_mib']>=14500 and gpu['available_RAM_MiB']>=5120)
                if (shared or exclusive) and not render_jobs():
                    atomic_json(out/'latest_resource_admission.json',{'shared_capped_RDT':shared,'resources':gpu,'own_allocator_cap_MiB':3072})
                    return
                save();time.sleep(30)
        wait_capacity()
        import numpy as np
        import torch
        import pybullet as p
        import imageio.v2 as imageio
        from VLA.common.environment_factory import environment_class_for_profile
        from VLA.common.profile_binding import checkpoint_profile
        from VLA.common.observation_profile import profile_hash
        from VLA.common.tasks import select_tasks,randomize_rl_scene
        from VLA.common.evaluation_statistics import success_statistics
        from VLA.benchmark import _interaction_audit
        from VLA.simstate_scene64 import scene_state
        from VLA.smolvla.scene67_adapter import Scene67SmolVLA
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        profile=checkpoint_profile(cp);protocol=json.loads(PROFILE_PROTOCOL.read_text())
        if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Frozen physical/camera profile required')
        def environment(seed):
            return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        def tasks(seed):
            rng=np.random.default_rng(seed);return [randomize_rl_scene(t,rng) for t in select_tasks('all')]
        def attach(env,seed):
            nonlocal policy
            if policy is None:policy=Scene67SmolVLA(env,str(cp),replan_steps=1,seed=seed)
            policy.env=env;policy.seed=seed;policy._load();policy.model.config.num_steps=20
        def compatible():
            gpu=resources();state=json.loads((OWNER/'status.json').read_text());pids={int(v) for v in gpu['compute_pids']}
            if state.get('stage')=='rdt_fit_2000':
                child=json.loads((OWNER/'active_child.json').read_text());return pids<={os.getpid(),int(child['pid'])} and identity(child['pid']) is not None
            return pids<={os.getpid()}
        # Same-input qualification, not an extra rerolled full episode.
        stage='actual_neural_reload_and_unshielded_reproduction8';save();checks=[];env=environment(SEEDS[0])
        reference=json.loads(Path(source['rows'][0]['trajectory']).read_text())
        try:
            attach(env,SEEDS[0])
            for task in tasks(SEEDS[0]):
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    np.testing.assert_allclose(scene_state(env,obs),reference[task.name][i]['pre_action_scene64'],rtol=0,atol=1e-6)
                    proposed=policy.predict(obs);action,stopped=project(proposed,p.getContactPoints(bodyA=env.gripper.body),set(env.gripper.PAD_LINKS))
                    if stopped:raise ValueError('Initial high-clearance baseline qualification unexpectedly engages shield')
                    np.testing.assert_allclose(proposed,reference[task.name][i]['action'],rtol=0,atol=1e-6)
                    obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'):raise ValueError('Actual integration8 terminated unexpectedly')
                    checks.append({'task':task.name,'step':i+1,'native_proposed_action':proposed.tolist()})
        finally:env.close()
        atomic_json(out/'actual_neural_reproduction8.json',{'passed':True,'actual_neural_controls':8,
            'state_and_command_tolerance':1e-6,'shield_not_engaged_at_initial_high_clearance':True,'rows':checks,'model_SR_established':False})
        def progress(status):
            atomic_json(out/'progress.json',{'status':status,'model':VARIANT,'checkpoint':str(cp),'rows':rows,
                'completed_trials':len(rows),'planned_trials':40,'successes':sum(r['success'] for r in rows),
                'privileged_inputs':True,'additional_contact_safety_shield':True,'NOT_plain_unshielded_VLA':True,
                'same_robot_assets_profile_scoring_safety_horizon':True,'final_test_used':False,'error':error,
                'statistics':success_statistics(rows) if rows and len(rows)%4==0 else None})
        for seed in SEEDS[:10]:
            env=environment(seed);folder=out/f'test_seed{seed}';folder.mkdir();video=None;traces={}
            try:
                for task in tasks(seed):
                    check()
                    if policy is not None and (render_jobs() or not compatible()):
                        policy.close();policy=None;gc.collect();torch.cuda.empty_cache()
                    if policy is None:wait_capacity()
                    attach(env,seed);obs=env.reset(task);policy.reset(task);trace=[];done=False
                    stage='actual_closedloop_native_VLA_plus_declared_contact_boundary_shield40';save();progress('running')
                    if env.is_semantic_success():raise ValueError('Initial success cannot replace a seed')
                    if seed==SEEDS[0] and video is None:video=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                    while not done:
                        before=_interaction_audit(env);proposed=policy.predict(obs);raw=policy.last_input['scene64'].copy()
                        action,stopped=project(proposed,p.getContactPoints(bodyA=env.gripper.body),set(env.gripper.PAD_LINKS))
                        obs,_,done,info=env.step(action)
                        trace.append({'task':task.name,'step':obs.step,'action':action.tolist(),'native_proposed_action':proposed.tolist(),
                            'contact_boundary_DZ_stop':stopped,'action_source':'native_neural_VLA_then_declared_task_independent_safety_projection',
                            'pre_action_scene64':raw.tolist(),'before':before,'after':_interaction_audit(env),
                            'success':bool(info['success']),'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback')})
                        traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                        if video:video.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'shield_interventions':sum(r['contact_boundary_DZ_stop'] for r in trace),
                        'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                    progress('running');save()
            finally:
                if video:video.close()
                env.close()
        if len(rows)!=40:raise ValueError('Incomplete actual40')
        if any(sha256_file(path)!=digest for path,digest in bound.items()):raise ValueError('Declared policy/shield implementation changed')
        baseline={(r['seed'],r['task']):r for r in source['rows']}
        atomic_json(out/'matched_comparison.json',{'unshielded_successes':source['successes'],'shielded_successes':sum(r['success'] for r in rows),
            'trials_each':40,'unshielded_safety_stops':sum(bool(r['safety_violation']) for r in source['rows']),
            'shielded_safety_stops':sum(bool(r['safety_violation']) for r in rows),
            'paired_improvements':sum(r['success'] and not baseline[r['seed'],r['task']]['success'] for r in rows),
            'paired_regressions':sum(not r['success'] and baseline[r['seed'],r['task']]['success'] for r in rows),
            'same_fixed_seeds_weights_scoring_horizons':True,'additional_shield_must_be_disclosed':True,'final_test_used':False})
        progress('completed40_native_VLA_plus_declared_contact_shield');stage='completed_actual_shield40_requires_review';save('completed')
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';save('failed_preserve_actual_trials_NOT_zero_SR');raise
    finally:
        if policy is not None:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=38);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=48:raise ValueError('Finite experiment only')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate shielded cohort')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.evaluate_contactstop_smol_r37','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'additional_shield_declared_NOT_plain_VLA':True,'no_other_neural_worker_signalled':True});print(child.pid)
