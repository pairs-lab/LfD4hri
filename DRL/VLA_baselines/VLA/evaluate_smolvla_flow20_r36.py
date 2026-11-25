"""Bounded native inference ablation, NEVER a replacement success scorer.

Same trained privileged67 Smol checkpoint/actions/private task RNG. First
reproduce8 actual baseline10 controls; then qualify8 native20 controls and
evaluate all40 fixed-seed tasks, including failures. No oracle action source,
new fitting, action correction, camera edit or best-seed selection.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks, randomize_rl_scene, TaskSpec
from VLA.run_cross_vla120 import SEEDS, PROFILE_PROTOCOL
from VLA.run_scene67_parallel_r34 import allowed_parallel, resources, OPENVLA, ROOT, SIM
from VLA.wait_render_then_resume_queues import render_jobs

BASE = ROOT/'VLA/audits/scene67_deadline_r31_20261015'
BASELINE = BASE/'scene67_resume_parallel_r34/parallel_attempt/development120'
CHECKPOINT = ROOT/'VLA/smolvla_scene67/automatic_full_gate_fit_v2/train/checkpoints/010000/pretrained_model'


def run(output):
    import torch
    import imageio.v2 as imageio
    from VLA.benchmark import _interaction_audit
    from VLA.simstate_scene64 import scene_state
    from VLA.smolvla.scene67_adapter import Scene67SmolVLA
    from VLA.common.evaluation_statistics import success_statistics
    if output.exists(): raise FileExistsError('Never reroll an existing variant')
    output.mkdir(parents=True)
    stage='checking_exact_owned_OpenVLA_inference_capacity'; rows=[]; error=None
    def save(status='running'):
        atomic_json(output/'status.json', {'pid':os.getpid(),'stage':stage,'status':status,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'model':'smolvla_scene67',
            'checkpoint':str(CHECKPOINT),'inference_steps':20,'original_inference_steps':10,
            'optimizer_updates_performed':0,'high_level_actions_native_VLA_only':True})
        atomic_json(output/'progress.json', {'model':'smolvla_scene67_native_flow20','checkpoint':str(CHECKPOINT),
            'status':status,'planned_trials':40,'completed_trials':len(rows),'rows':rows,
            'successes':sum(r['success'] for r in rows),'error':error,'inference_steps':20,
            'privileged_inputs':True,'final_test_used':False,'native_10Hz_actions_and_scoring_unchanged':True,
            'statistics':success_statistics(rows) if rows and len(rows)%4==0 else None})
    save()
    source=json.loads((BASELINE/'progress.json').read_text())
    profile=checkpoint_profile(CHECKPOINT); protocol=json.loads(PROFILE_PROTOCOL.read_text())
    if source['completed_trials']!=120 or profile_hash(profile)!=protocol['observation_profile_sha256']:
        raise ValueError('Completed baseline and exact locked profile required')
    gpu=resources(); owner=json.loads((OPENVLA/'status.json').read_text())
    server=json.loads((OPENVLA/'owned_server.json').read_text()); child=json.loads((OPENVLA/'active_child.json').read_text())
    if render_jobs() or not allowed_parallel(owner,gpu,server,child):
        raise ValueError('No safe exact inference-only capacity; no worker is interrupted')
    atomic_json(output/'resource_guard.json', {'actual_resources':gpu,'OpenVLA_owner':owner['pid'],
        'OpenVLA_server':server['pid'],'OpenVLA_evaluator':child['pid'],'allocator_cap_MiB':3072})
    torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
    files=[Path(__file__),ROOT/'VLA/smolvla/scene67_adapter.py',ROOT/'VLA/smolvla/adapter.py',
        ROOT/'VLA/scene64_conditioning.py',PROFILE_PROTOCOL,CHECKPOINT/'model.safetensors']
    bound={str(p):sha256_file(p) for p in files}
    policy=None
    def tasks(seed):
        rng=np.random.default_rng(seed)
        return [randomize_rl_scene(t,rng) for t in select_tasks('all')]
    def environment(seed):
        return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
            control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
            success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
            semantic_success_thresholds=protocol['success']['semantic_thresholds'])
    def attach(env, seed, steps):
        nonlocal policy
        if policy is None: policy=Scene67SmolVLA(env,str(CHECKPOINT),replan_steps=1,seed=seed)
        policy.env=env;policy.seed=seed;policy._load()
        # The native model/policy share their config. No sampler replacement.
        if policy.model.model.config is not policy.model.config:
            raise ValueError('Native flow config is not shared as expected')
        policy.model.config.num_steps=steps
    try:
        # Independent real baseline-control reproduction catches reuse/RNG or
        # input wiring defects before changing an inference hyperparameter.
        stage='actual_baseline10_reproduction8';save(); env=environment(SEEDS[0]);proof=[]
        trace_path=Path(next(r for r in source['rows'] if r['seed']==SEEDS[0])['trajectory'])
        old=json.loads(trace_path.read_text())
        try:
            attach(env,SEEDS[0],10)
            for task in tasks(SEEDS[0]):
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    np.testing.assert_allclose(scene_state(env,obs),old[task.name][i]['pre_action_scene64'],rtol=0,atol=1e-6)
                    action=policy.predict(obs)
                    np.testing.assert_allclose(action,old[task.name][i]['action'],rtol=0,atol=1e-6)
                    obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'): raise ValueError('Baseline8 terminated early')
                    proof.append({'task':task.name,'step':i+1,'actual_neural_action':action.tolist()})
        finally: env.close()
        atomic_json(output/'baseline10_actual8.json', {'passed':True,'actual_neural_controls':8,'rows':proof,
            'maximum_action_tolerance':1e-6,'no_development_episodes_rerolled':True,
            'purpose':'isolated8_control_reproduction_NOT_new_SR','source_trajectory_sha256':sha256_file(trace_path)})
        stage='actual_native20_qualification8';save();env=environment(20261015);proof=[]
        source_root=ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'
        train=json.loads((source_root/'index.json').read_text())
        try:
            attach(env,20261015,20)
            for template in select_tasks('all'):
                task=TaskSpec.from_dict(next(r for r in train if r['task']['name']==template.name)['task'])
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    action=policy.predict(obs);obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'):raise ValueError('Native20 integration8 terminated early')
                    proof.append({'task':task.name,'step':i+1,'actual_neural_action':action.tolist()})
        finally:env.close()
        atomic_json(output/'native20_actual8.json', {'passed':True,'actual_neural_controls':8,'rows':proof,
            'inference_steps':20,'weights_or_high_level_controller_changed':False,'model_SR_established':False})
        stage='native20_development40';save()
        for seed in SEEDS[:10]:
            env=environment(seed);video=None;traces={};folder=output/f'test_seed{seed}';folder.mkdir()
            try:
                attach(env,seed,20)
                if seed==SEEDS[0]:video=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                for task in tasks(seed):
                    obs=env.reset(task);policy.reset(task);trace=[];done=False
                    if env.is_semantic_success():raise ValueError('Initially successful scene; never replace it')
                    while not done:
                        before=_interaction_audit(env);action=policy.predict(obs);raw=policy.last_input['scene64'].copy()
                        obs,_,done,info=env.step(action)
                        trace.append({'task':task.name,'step':obs.step,'action':action.tolist(),
                            'pre_action_scene64':raw.tolist(),'before':before,'after':_interaction_audit(env),
                            'success':bool(info['success']),'safety_violation':info.get('safety_violation')})
                        traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                        atomic_json(output/'applied_step_journal.json',{'seed':seed,'task':task.name,'step':obs.step,
                            'terminal':done,'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
                        if video:video.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                    save()
            finally:
                if video:video.close()
                env.close()
        if len(rows)!=40:raise ValueError('Incomplete declared40')
        matched=[r for r in source['rows'] if r['seed'] in SEEDS[:10]]
        pairs={(r['seed'],r['task']):r for r in matched}
        improvements=sum(not pairs[r['seed'],r['task']]['success'] and r['success'] for r in rows)
        regressions=sum(pairs[r['seed'],r['task']]['success'] and not r['success'] for r in rows)
        if any(sha256_file(Path(p))!=h for p,h in bound.items()):raise ValueError('Immutable model/runtime changed')
        atomic_json(output/'matched_comparison.json', {'baseline10_successes':sum(r['success'] for r in matched),
            'native20_successes':sum(r['success'] for r in rows),'trials_each':40,'paired_improvements':improvements,
            'paired_regressions':regressions,'same_seeds_profile_scoring_and_checkpoint':True,
            'selection_or_training_on_final_test':False,'tuning_cohort_NOT_final_paper_estimate':True,
            'bound_files':bound,'no_optimizer_updates':True})
        stage='completed_native20_matched40_NOT_claimed_highest_SR';save('completed')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';save('failed_preserve_actual_evidence_NOT_zero_SR');raise
    finally:
        if policy:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:run(a.output.resolve())
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate variant')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.evaluate_smolvla_flow20_r36','--output',str(a.output.resolve())],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'neural_inference_only':True,'existing_jobs_interrupted':False});print(child.pid)
