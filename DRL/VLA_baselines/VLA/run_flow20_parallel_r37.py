"""Preserve native20's40 trials; evaluate80 under a verified RDT GPU cap.

Only measured spare capacity of the exact owned capped RDT trainer is used.
No trainer/server/episode is stopped. First reproduce8 native20 controls under
concurrency. Same weights, input profile, task seeds, sampler and task budgets.
If compatibility ends, release our model between tasks and wait exclusively.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import ROOT,SEEDS,PROFILE_PROTOCOL,gpu_state
from VLA.run_flow20_extension_r36 import validate_preserved,PILOT,BASELINE,parent_roots,alive
from VLA.wait_render_then_resume_queues import identity,render_jobs

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
OWNER=BASE/'all_family_privileged_r33_v4'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'


def resources():
    values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    return {**gpu_state(),'available_RAM_MiB':values['MemAvailable']/1024}


def allowed(state,child,process,gpu):
    """Only the declared owned long RDT trainer, never an unknown GPU user."""
    return bool(state.get('stage')=='rdt_fit_2000' and state.get('status')=='running'
        and process and process.get('cwd')==str(ROOT)
        and b'VLA.rdt.train_scene67' in process.get('tokens',[])
        and str(OWNER/'rdt_fit_2000').encode() in process.get('tokens',[])
        and {int(p) for p in gpu['compute_pids']}=={int(child['pid'])}
        and gpu['free_mib']>=5120 and gpu['available_RAM_MiB']>=5120)


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);policy=None;error=None;stage='validating_original40'
    rows=[];deadline=time.monotonic()+a.wait_hours*3600
    atomic_json(BASE/'Smol_native20_extension_r36_active.json',{'cycle':str(out),'pid':os.getpid(),
        'same_original40_plus80':True,'reason':'verified capped RDT spare-capacity continuation'})
    def save(status='running'):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'preserved_trials':40 if rows else 0,
            'new_terminal_trials':max(0,len(rows)-40),'same_checkpoint_native20':True,
            'other_neural_job_signalled':False,'final_test_used':False,'allocator_cap_MiB':3072})
    def deadline_check():
        if time.monotonic()>deadline:raise TimeoutError('Finite between-task deadline; all actually applied controls retained')
    save()
    try:
        pilot=json.loads((PILOT/'progress.json').read_text());comparison=json.loads((PILOT/'matched_comparison.json').read_text())
        if pilot['completed_trials']!=40 or comparison['native20_successes']<=comparison['baseline10_successes']:
            raise ValueError('Completed matched40 improvement required')
        old=json.loads((BASELINE/'progress.json').read_text())['rows'];baseline=[r for r in old if r['seed'] in SEEDS[:10]]
        if sum(bool(r.get('safety_violation')) for r in pilot['rows'])>sum(bool(r.get('safety_violation')) for r in baseline):
            raise ValueError('No increased pilot safety-stop count permitted')
        rows=list(pilot['rows']);bound=validate_preserved(rows)
        for path,digest in comparison['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Original native20 model/runtime changed')
        train_impl=ROOT/'VLA/rdt/train_scene67.py'
        if '8500*2**20/' not in train_impl.read_text():raise ValueError('Owned RDT allocator is not capped as declared')
        state=json.loads((OWNER/'status.json').read_text())
        if state['frozen_files'][str(train_impl)]!=sha256_file(train_impl):raise ValueError('Live RDT trainer source identity changed')
        bound.update({str(Path(__file__)):sha256_file(Path(__file__)),str(train_impl):sha256_file(train_impl)})
        def wait_capacity():
            nonlocal stage
            stage='waiting_exact_capped_RDT_spare_capacity_or_idle_GPU';save()
            while True:
                deadline_check();gpu=resources();state=json.loads((OWNER/'status.json').read_text())
                child=json.loads((OWNER/'active_child.json').read_text());process=identity(child['pid'])
                shared=allowed(state,child,process,gpu)
                # An empty instant between predecessor children is NOT a
                # lease:the existing all-family queue must retain priority.
                exclusive=(not any(alive(p) for p in parent_roots()) and not gpu['compute_pids']
                    and gpu['free_mib']>=14500 and gpu['available_RAM_MiB']>=5120)
                if (shared or exclusive) and not render_jobs():
                    atomic_json(out/'latest_resource_admission.json',{'shared_capped_RDT':shared,'resources':gpu,
                        'owned_RDT_child':child['pid'] if shared else None,'Smol_cap_MiB':3072,
                        'RDT_cap_MiB':8500 if shared else None,'sum_declared_allocator_caps_MiB':11572 if shared else 3072,
                        'inflight_training_or_episode_interrupted':False})
                    return
                save();time.sleep(30)
        wait_capacity()
        import gc
        import numpy as np
        import imageio.v2 as imageio
        import torch
        from VLA.benchmark import _interaction_audit
        from VLA.common.environment_factory import environment_class_for_profile
        from VLA.common.profile_binding import checkpoint_profile
        from VLA.common.observation_profile import profile_hash
        from VLA.common.tasks import select_tasks,randomize_rl_scene
        from VLA.common.evaluation_statistics import success_statistics
        from VLA.simstate_scene64 import scene_state
        from VLA.smolvla.scene67_adapter import Scene67SmolVLA
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        cp=Path(pilot['checkpoint']);profile=checkpoint_profile(cp);protocol=json.loads(PROFILE_PROTOCOL.read_text())
        if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Locked observation profile differs')
        def environment(seed):
            return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        def tasks(seed):
            rng=np.random.default_rng(seed);return [randomize_rl_scene(t,rng) for t in select_tasks('all')]
        def attach(env,seed):
            nonlocal policy
            if policy is None:policy=Scene67SmolVLA(env,str(cp),replan_steps=1,seed=seed)
            policy.env=env;policy.seed=seed;policy._load();policy.model.config.num_steps=20
            if policy.model.model.config is not policy.model.config:raise ValueError('Native solver config not shared')
        def release():
            nonlocal policy
            if policy is not None:policy.close();policy=None;gc.collect();torch.cuda.empty_cache()
        def capacity_still_compatible():
            state=json.loads((OWNER/'status.json').read_text());gpu=resources();pids={int(p) for p in gpu['compute_pids']}
            if state.get('stage')=='rdt_fit_2000':
                child=json.loads((OWNER/'active_child.json').read_text())
                return pids<={os.getpid(),int(child['pid'])} and identity(child['pid']) is not None
            return pids<={os.getpid()}
        # Qualification only: no completed development episode is rerolled.
        stage='actual_concurrent_native20_reproduction8';save();env=environment(SEEDS[0]);checks=[]
        reference=json.loads(Path(pilot['rows'][0]['trajectory']).read_text())
        try:
            attach(env,SEEDS[0])
            for task in tasks(SEEDS[0]):
                obs=env.reset(task);policy.reset(task)
                for i in range(2):
                    np.testing.assert_allclose(scene_state(env,obs),reference[task.name][i]['pre_action_scene64'],rtol=0,atol=1e-6)
                    action=policy.predict(obs)
                    np.testing.assert_allclose(action,reference[task.name][i]['action'],rtol=0,atol=1e-6)
                    obs,_,done,info=env.step(action)
                    if done or info.get('safety_violation'):raise ValueError('Concurrent reproduction8 terminated unexpectedly')
                    checks.append({'task':task.name,'step':i+1,'actual_neural_action':action.tolist()})
        finally:env.close()
        atomic_json(out/'actual_concurrent_reproduction8.json',{'passed':True,'actual_neural_controls':8,'rows':checks,
            'state_and_action_tolerance':1e-6,'same_native20_checkpoint_and_private_rng':True,'new_completed_trials':0,
            'allocator_peak_MiB':torch.cuda.max_memory_allocated()/2**20})
        def progress(status):
            atomic_json(out/'progress.json',{'status':status,'model':'smolvla_scene67_native_flow20','checkpoint':str(cp),
                'inference_steps':20,'rows':rows,'completed_trials':len(rows),'planned_trials':120,
                'successes':sum(r['success'] for r in rows),'preserved_original40_not_rerolled':True,
                'privileged_inputs':True,'final_test_used':False,'error':error,'statistics':success_statistics(rows) if len(rows)%4==0 else None})
        for seed in SEEDS[10:]:
            env=environment(seed);video=None;traces={};folder=out/f'test_seed{seed}';folder.mkdir()
            try:
                for task in tasks(seed):
                    deadline_check()
                    if policy is not None and (render_jobs() or not capacity_still_compatible()):release()
                    if policy is None:wait_capacity()
                    attach(env,seed);obs=env.reset(task);policy.reset(task);trace=[];done=False
                    stage='same_native20_preserve40_evaluate_extra80';save();progress('running')
                    if env.is_semantic_success():raise ValueError('Initially successful scene cannot be replaced')
                    if seed==SEEDS[10] and video is None:video=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                    while not done:
                        before=_interaction_audit(env);action=policy.predict(obs);raw=policy.last_input['scene64'].copy()
                        obs,_,done,info=env.step(action)
                        trace.append({'task':task.name,'step':obs.step,'action':action.tolist(),
                            'pre_action_scene64':raw.tolist(),'before':before,'after':_interaction_audit(env),
                            'success':bool(info['success']),'safety_violation':info.get('safety_violation')})
                        traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                        atomic_json(out/'applied_step_journal.json',{'seed':seed,'task':task.name,'step':obs.step,'terminal':done,
                            'allocator_peak_MiB':torch.cuda.max_memory_allocated()/2**20})
                        if video:video.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                    progress('running');save()
            finally:
                if video:video.close()
                env.close()
        if len(rows)!=120:raise ValueError('Incomplete declared120')
        if any(sha256_file(p)!=digest for p,digest in bound.items()):raise ValueError('Original40 or native implementation changed')
        atomic_json(out/'preserved_evidence.json',{'passed':True,'bound_files':bound,'all_original40_preserved':True,
            'all80_remaining_fixed_trials_executed':True,'same_high_level_native_VLA_only':True})
        progress('completed120_privileged_native20_VLA');stage='completed_native20_120';save('completed')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';save('failed_preserve_actual_controls_NOT_zero_SR');raise
    finally:
        if policy is not None:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=38);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=48:raise ValueError('Finite remaining-development budget only')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate80 continuation')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_flow20_parallel_r37','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'same_original40_preserved':True,'other_neural_jobs_not_signalled':True});print(child.pid)
