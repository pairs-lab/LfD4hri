"""If native20 wins its COMPLETE matched40, preserve40 and evaluate80 more.

All other registered families/refinements keep GPU priority. No reroll of
the original40, no final-test selection, native learned actions only.
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
from VLA.wait_render_then_resume_queues import render_jobs

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015';SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PILOT=BASE/'Smol_native20_r36'
BASELINE=BASE/'scene67_resume_parallel_r34/parallel_attempt/development120'
PARENTS=[BASE/name for name in ('all_family_privileged_r33_v4','openpi_EEF_repair_r33_v4',
    'OpenVLA_balanced_recovery_r34_v2')]


def parent_roots():
    pointer=BASE/'Smol_GTcrop_motion_r36_active.json'
    crop=Path(json.loads(pointer.read_text())['cycle']) if pointer.exists() else BASE/'Smol_GTcrop_motion_r36_v2'
    if crop.parent!=BASE or not crop.name.startswith('Smol_GTcrop_motion_r36'):
        raise ValueError('Unexpected auxiliary fit owner; no GPU work launched')
    return [*PARENTS,crop]


def alive(root):
    try:
        s=json.loads((root/'status.json').read_text());p=Path('/proc')/str(s['pid'])
        return (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,KeyError,IndexError):return False


def validate_preserved(rows):
    expected=[(s,t) for s in SEEDS[:10] for t in ('touch','pick','move','pick_place')]
    if [(r['seed'],r['task']) for r in rows]!=expected:raise ValueError('All original40 in fixed order required')
    cache={};bound={}
    for row in rows:
        path=Path(row['trajectory'])
        if path not in cache:
            cache[path]=json.loads(path.read_text());bound[str(path)]=sha256_file(path)
        trace=cache[path][row['task']]
        if len(trace)!=row['steps'] or bool(trace[-1]['success'])!=bool(row['success']):raise ValueError('Original terminal neural evidence differs')
    return bound


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);stage='waiting_complete_matched40';rows=[];error=None
    atomic_json(BASE/'Smol_native20_extension_r36_active.json',{'cycle':str(output),'pid':os.getpid()})
    deadline=time.monotonic()+a.wait_hours*3600
    def save(status='running'):
        atomic_json(output/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'same_checkpoint_native20':True,
            'preserved_trials':40 if rows else 0,'new_terminal_trials':max(0,len(rows)-40),
            'no_existing_job_interrupted':True,'final_test_used':False})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite remaining development deadline')
    save()
    try:
        while not (PILOT/'matched_comparison.json').exists():
            check();state=json.loads((PILOT/'status.json').read_text())
            if str(state['status']).startswith('failed'):raise ValueError('Pilot failed; cannot infer a model SR')
            save();time.sleep(30)
        pilot=json.loads((PILOT/'progress.json').read_text());comparison=json.loads((PILOT/'matched_comparison.json').read_text())
        if pilot['completed_trials']!=40 or comparison['trials_each']!=40:raise ValueError('Complete matched40 required')
        baseline=json.loads((BASELINE/'progress.json').read_text())['rows']
        matched=[r for r in baseline if r['seed'] in SEEDS[:10]]
        baseline_safety=sum(bool(r.get('safety_violation')) for r in matched)
        candidate_safety=sum(bool(r.get('safety_violation')) for r in pilot['rows'])
        wins=comparison['native20_successes']>comparison['baseline10_successes'] and candidate_safety<=baseline_safety
        atomic_json(output/'decision.json',{'pilot_complete':True,'matched40':comparison,
            'baseline_safety_aborts':baseline_safety,'candidate_safety_aborts':candidate_safety,
            'extend':wins,'rule':'strictly more matched successes and no more safety aborts',
            'development_hyperparameter_selection_NOT_independent_final_test':True})
        if not wins:
            stage='completed_no_matched_gain_no_extra_rollouts';save('completed');return
        rows=list(pilot['rows']);bound=validate_preserved(rows)
        for path,digest in comparison['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Exact pilot model/runtime changed')
        stage='waiting_other_families_and_GTcrop_before_extra80';save()
        while True:
            check();gpu=gpu_state()
            memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
            if (not any(alive(p) for p in parent_roots()) and not gpu['compute_pids'] and gpu['free_mib']>=14500
                    and memory['MemAvailable']>=5*1024**2 and not render_jobs()):break
            save();time.sleep(30)
        import numpy as np
        import imageio.v2 as imageio
        import torch
        from VLA.benchmark import _interaction_audit
        from VLA.common.environment_factory import environment_class_for_profile
        from VLA.common.profile_binding import checkpoint_profile
        from VLA.common.observation_profile import profile_hash
        from VLA.common.tasks import select_tasks,randomize_rl_scene
        from VLA.common.evaluation_statistics import success_statistics
        from VLA.smolvla.scene67_adapter import Scene67SmolVLA
        cp=Path(pilot['checkpoint']);profile=checkpoint_profile(cp);protocol=json.loads(PROFILE_PROTOCOL.read_text())
        if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Frozen profile mismatch')
        torch.cuda.set_per_process_memory_fraction(5*2**30/torch.cuda.get_device_properties(0).total_memory)
        def progress(status):
            atomic_json(output/'progress.json',{'status':status,'model':'smolvla_scene67_native_flow20','checkpoint':str(cp),
                'inference_steps':20,'rows':rows,'completed_trials':len(rows),'planned_trials':120,
                'successes':sum(r['success'] for r in rows),'preserved_original40_not_rerolled':True,
                'privileged_inputs':True,'final_test_used':False,'error':error,
                'statistics':success_statistics(rows) if len(rows)%4==0 else None})
        stage='same_native20_preserve40_and_evaluate_extra80';save();progress('running');policy=None
        try:
            for seed in SEEDS[10:]:
                check();rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
                env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                    control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                    success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                    semantic_success_thresholds=protocol['success']['semantic_thresholds'])
                folder=output/f'test_seed{seed}';folder.mkdir();video=None;traces={}
                try:
                    if policy is None:policy=Scene67SmolVLA(env,str(cp),replan_steps=1,seed=seed)
                    policy.env=env;policy.seed=seed;policy._load();policy.model.config.num_steps=20
                    if seed==SEEDS[10]:video=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                    for task in tasks:
                        obs=env.reset(task);policy.reset(task);trace=[];done=False
                        if env.is_semantic_success():raise ValueError('Initially successful scene is not replaced')
                        while not done:
                            before=_interaction_audit(env);action=policy.predict(obs);raw=policy.last_input['scene64'].copy()
                            obs,_,done,info=env.step(action)
                            trace.append({'task':task.name,'step':obs.step,'action':action.tolist(),
                                'pre_action_scene64':raw.tolist(),'before':before,'after':_interaction_audit(env),
                                'success':bool(info['success']),'safety_violation':info.get('safety_violation')})
                            traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                            if video:video.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                        rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                            'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                        progress('running');save()
                finally:
                    if video:video.close()
                    env.close()
            if len(rows)!=120:raise ValueError('Incomplete actual120')
            if any(sha256_file(p)!=h for p,h in bound.items()):raise ValueError('Preserved original40 changed')
            progress('completed120_privileged_native20_VLA');stage='completed_same_checkpoint_native20_120';save('completed')
        finally:
            if policy:policy.close()
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';save('failed_preserve_actual_trials_NOT_zero_SR');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-hours',type=float,default=44);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=48:raise ValueError('Finite bounded deadline required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate extension')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_flow20_extension_r36','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'original40_will_not_be_rerolled':True});print(child.pid)
