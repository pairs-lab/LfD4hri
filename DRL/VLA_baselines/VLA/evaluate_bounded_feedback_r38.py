"""Finite actual neural40: only native standardized feedback bounding changes.

Wait for the original120 and separate contact-shield40. No contact/action
shield is used here. Same40 seeds, neural private RNG and20 flow steps;
preserve failures, raw inputs, exact proposed/executed commands and videos.
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

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
AUDIT=BASE/'Smol_state_support_r38/audit.json'
PREVIOUS=BASE/'Smol_contactstop_closedloop_r37'


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    stage='waiting_original120_then_contact40_and_actual_state_audit';rows=[];policy=None;error=None
    deadline=time.monotonic()+a.wait_hours*3600
    def save(status='running'):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'completed_neural_trials':len(rows),
            'planned_neural_trials':40,'action_shield_or_oracle_fallback':False,
            'original_unbounded_results_preserved':True,'success_improvement_established':False})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite between-task development budget')
    save()
    try:
        while True:
            check();prior=json.loads((PREVIOUS/'status.json').read_text())
            if str(prior['status']).startswith('failed'):raise ValueError('Prior experiment failed; preserve its actual evidence')
            if prior['status']=='completed' and AUDIT.exists():break
            save();time.sleep(20)
        audit=json.loads(AUDIT.read_text())
        if not audit['passed_transport_audit'] or audit['TRAIN']['frames']!=117736:raise ValueError('Full native-state audit required')
        for path,digest in audit['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Bound state audit changed')
        source=json.loads((PILOT/'progress.json').read_text());cp=Path(source['checkpoint'])
        if source['completed_trials']!=40:raise ValueError('Complete fixed baseline40 required')
        from VLA.smolvla.bounded_feedback_r38 import SCHEMA
        files=[Path(__file__),ROOT/'VLA/smolvla/bounded_feedback_r38.py',ROOT/'VLA/smolvla/scene67_adapter.py',
            ROOT/'VLA/smolvla/adapter.py',AUDIT,PROFILE_PROTOCOL,cp/'model.safetensors',
            cp/'policy_preprocessor_step_5_normalizer_processor.safetensors']
        bound={str(p):sha256_file(p) for p in files}
        atomic_json(out/'plan.json',{'variant':SCHEMA,'checkpoint':str(cp),'seeds':SEEDS[:10],
            'planned_trials':40,'all_seeds_fixed_before_candidate_outcomes':True,'paired_unbounded_baseline':str(PILOT/'progress.json'),
            'additional_contact_shield':False,'native_action_source':'actual_neural_VLA_every_control',
            'new_finetuning_updates':0,'only_change':'saved_normalizer_standardized_state_bound10',
            'final_test_used':False,'bound_files':bound})
        def wait_capacity():
            nonlocal stage
            stage='waiting_exact_capped_RDT_capacity_or_postqueue_exclusive';save()
            while True:
                check();gpu=resources();state=json.loads((OWNER/'status.json').read_text());child=json.loads((OWNER/'active_child.json').read_text())
                shared=allowed(state,child,identity(child['pid']),gpu)
                exclusive=(not any(alive(p) for p in parent_roots()) and not gpu['compute_pids']
                    and gpu['free_mib']>=14500 and gpu['available_RAM_MiB']>=5120)
                if (shared or exclusive) and not render_jobs():
                    atomic_json(out/'resource_admission.json',{'shared_capped_RDT':shared,'resources':gpu,'own_allocator_cap_MiB':3072})
                    return
                save();time.sleep(20)
        wait_capacity()
        import numpy as np
        import torch
        import imageio.v2 as imageio
        from VLA.common.environment_factory import environment_class_for_profile
        from VLA.common.profile_binding import checkpoint_profile
        from VLA.common.observation_profile import profile_hash
        from VLA.common.tasks import select_tasks,randomize_rl_scene
        from VLA.common.evaluation_statistics import success_statistics
        from VLA.benchmark import _interaction_audit
        from VLA.simstate_scene64 import scene_state
        from VLA.smolvla.bounded_feedback_r38 import BoundedFeedbackSmolVLA
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        profile=checkpoint_profile(cp);protocol=json.loads(PROFILE_PROTOCOL.read_text())
        if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Physical profile changed')
        def environment(seed):
            return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        def tasks(seed):
            rng=np.random.default_rng(seed);return [randomize_rl_scene(t,rng) for t in select_tasks('all')]
        def attach(env,seed):
            nonlocal policy
            if policy is None:policy=BoundedFeedbackSmolVLA(env,str(cp),replan_steps=1,seed=seed)
            policy.env=env;policy.seed=seed;policy._load();policy.model.config.num_steps=20
        def compatible():
            gpu=resources();state=json.loads((OWNER/'status.json').read_text());pids={int(p) for p in gpu['compute_pids']}
            if state.get('stage')=='rdt_fit_2000':
                child=json.loads((OWNER/'active_child.json').read_text())
                return pids<={os.getpid(),int(child['pid'])} and identity(child['pid']) is not None
            return pids<={os.getpid()}
        stage='actual_native_reload_unclipped_reproduction8';save();checks=[];env=environment(SEEDS[0])
        reference=json.loads(Path(source['rows'][0]['trajectory']).read_text())
        try:
            attach(env,SEEDS[0])
            for task in tasks(SEEDS[0]):
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    np.testing.assert_allclose(scene_state(env,obs),reference[task.name][i]['pre_action_scene64'],rtol=0,atol=1e-6)
                    action=policy.predict(obs)
                    if policy.last_input['clipped_numeric_indices']:raise ValueError('Initial in-support reproduction unexpectedly clips')
                    np.testing.assert_allclose(action,reference[task.name][i]['action'],rtol=0,atol=1e-6)
                    obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'):raise ValueError('Actual8 ended early')
                    checks.append({'task':task.name,'step':i+1,'native_action':action.tolist()})
        finally:env.close()
        atomic_json(out/'actual_native_reproduction8.json',{'passed':True,'actual_neural_controls':8,
            'state_action_tolerance':1e-6,'in_support_state_unchanged':True,'rows':checks,'new_model_SR_established':False})
        def progress(status):
            atomic_json(out/'progress.json',{'status':status,'model':SCHEMA,'checkpoint':str(cp),'rows':rows,
                'completed_trials':len(rows),'planned_trials':40,'successes':sum(r['success'] for r in rows),
                'privileged_inputs':True,'only_native_neural_actions':True,'additional_action_shield':False,
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
                    stage='actual_native20_bounded_feedback_fixed40';save();progress('running')
                    if env.is_semantic_success():raise ValueError('Initially successful seed cannot be replaced')
                    if seed==SEEDS[0] and video is None:video=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                    while not done:
                        before=_interaction_audit(env);action=policy.predict(obs);inputs=policy.last_input
                        raw=inputs['scene64'].copy();numeric=inputs['numeric67'].copy();bounded=inputs['bounded_numeric67'].copy()
                        indices=list(inputs['clipped_numeric_indices']);obs,_,done,info=env.step(action)
                        trace.append({'task':task.name,'step':obs.step,'action':action.tolist(),
                            'action_source':'actual_native_neural_VLA_NO_action_projection','pre_action_scene64':raw.tolist(),
                            'unbounded_numeric67':numeric.tolist(),'actual_bounded_numeric67':bounded.tolist(),
                            'clipped_numeric_indices':indices,'before':before,'after':_interaction_audit(env),
                            'success':bool(info['success']),'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback')})
                        traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                        if video:video.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'steps_with_bounded_fields':sum(bool(x['clipped_numeric_indices']) for x in trace),
                        'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                    progress('running');save()
            finally:
                if video:video.close()
                env.close()
        if len(rows)!=40 or any(sha256_file(p)!=h for p,h in bound.items()):raise ValueError('Incomplete40 or frozen policy changed')
        baseline={(r['seed'],r['task']):r for r in source['rows']}
        win=sum(r['success'] and not baseline[r['seed'],r['task']]['success'] for r in rows)
        loss=sum(not r['success'] and baseline[r['seed'],r['task']]['success'] for r in rows)
        old_safety=sum(bool(r['safety_violation']) for r in source['rows']);new_safety=sum(bool(r['safety_violation']) for r in rows)
        atomic_json(out/'matched_comparison.json',{'unbounded_successes':source['successes'],'bounded_successes':sum(r['success'] for r in rows),
            'trials_each':40,'paired_improvements':win,'paired_regressions':loss,'unbounded_safety_stops':old_safety,
            'bounded_safety_stops':new_safety,'candidate_for_further_validation_NOT_final_best':win>loss and new_safety<=old_safety,
            'same_seeds_weights_scoring_horizons':True,'untrained_input_bound_must_be_disclosed':True,'final_test_used':False})
        progress('completed40_actual_native20_bounded_feedback');stage='completed_actual40_requires_review';save('completed')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';save('failed_preserve_actual_trials_NOT_zero_SR');raise
    finally:
        if policy is not None:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=36);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=48:raise ValueError('Finite experiment only')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate native40')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.evaluate_bounded_feedback_r38','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'neural_actions_only':True,'no_existing_worker_signalled':True});print(child.pid)
