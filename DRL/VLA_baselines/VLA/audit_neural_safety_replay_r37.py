"""Recover omitted pre-rollback diagnostics from recorded neural commands.

CPU/DIRECT physics replay ONLY. Every recorded pre-action64 must reproduce;
all commands are the original saved VLA outputs, not oracle corrections.
No checkpoint, source rollout, scoring, motor or safety threshold is edited.
Offending-state renders are diagnostic images, NEVER model observations.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'


def classify(diagnostics,names,pad_links):
    contacts=diagnostics['contacts'];active=[c for c in contacts if c['normal_force_n']>.1]
    labelled=[{**c,'other_body_name':names.get(c['other_body'],'unknown_body')} for c in active]
    return {'active_contacts':labelled,
        'max_abs_follower_error_rad':max(map(abs,diagnostics['follower_errors_rad']),default=0.),
        'strongest_contact_force_N':max((c['normal_force_n'] for c in active),default=0.),
        'nonpad_object_contact':any(c['gripper_link'] not in pad_links and names.get(c['other_body'],'').startswith('object:') for c in active),
        'pad_contact_object_names':sorted({names.get(c['other_body'],'unknown_body') for c in active if c['gripper_link'] in pad_links}),
        'ground_contact':any(names.get(c['other_body'])=='ground' for c in active),
        'robot_arm_contact':any(names.get(c['other_body'])=='robot_arm' for c in active),
        'labels_are_diagnostic_NOT_proven_root_cause':True}


def run(a):
    from collections import Counter
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.profile_binding import checkpoint_profile
    from VLA.common.observation_profile import profile_hash
    from VLA.common.tasks import select_tasks,randomize_rl_scene
    from VLA.common.physics_safety import PhysicsSafetyViolation
    from VLA.simstate_scene64 import scene_state
    from VLA.run_cross_vla120 import PROFILE_PROTOCOL
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);results=[];sources=[];cases=[]
    bound={str(p):sha256_file(p) for p in (Path(__file__),ROOT/'VLA/common/body_environment.py',
        ROOT/'VLA/common/body_gripper.py',ROOT/'VLA/common/physics_safety.py',ROOT/'VLA/common/environment.py',
        ROOT/'VLA/common/observation_profile.py',ROOT/'VLA/simstate_scene64.py',PROFILE_PROTOCOL)}
    protocol=json.loads(PROFILE_PROTOCOL.read_text())
    for number,path in enumerate(a.progress):
        path=path.resolve();raw=path.read_bytes();progress=json.loads(raw)
        atomic_json(output/f'source_progress_snapshot_{number}.json',progress)
        # These are fixed progress snapshots, not requests to reroll a model.
        rows=[r for r in progress['rows'] if r.get('safety_violation')]
        profile=checkpoint_profile(Path(progress['checkpoint']))
        if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Same pinned physics profile required')
        sources.append({'progress':str(path),'snapshot_sha256':hashlib.sha256(raw).hexdigest(),
            'model':progress['model'],'snapshot_completed_trials':len(progress['rows']),'safety_cases':len(rows)})
        for row in rows:
            trajectory=Path(row['trajectory']);payload=trajectory.read_bytes();traces=json.loads(payload);trace=traces[row['task']]
            if len(trace)!=row['steps'] or trace[-1].get('safety_violation')!=row['safety_violation']:
                raise ValueError('Actual terminal safety trace required')
            cases.append((number,progress,profile,row,trace,hashlib.sha256(payload).hexdigest()))
    atomic_json(output/'plan.json',{'sources':sources,'planned_saved_neural_safety_replays':len(cases),
        'all_safety_cases_in_snapshots_included':True,'bounded_new_neural_trials':0,'new_success_rate':None,
        'physics_safety_or_commands_changed':False,'bound_files':bound})
    def save(status,error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'status':status,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'completed_replays':len(results),'planned_replays':len(cases),
            'passed_replays':sum(r['passed'] for r in results),'rows':results,'new_neural_trials':0,'new_SR':None})
    save('replaying_recorded_neural_safety_cases_CPU_only')
    for number,(source,progress,profile,row,trace,tracehash) in enumerate(cases):
        folder=output/f"case{number:03d}_{progress['model']}_{row['seed']}_{row['task']}";folder.mkdir()
        result={'source_number':source,'model':progress['model'],'seed':row['seed'],'task':row['task'],
            'recorded_steps':len(trace),'recorded_safety_violation':row['safety_violation'],'passed':False,
            'source_trajectory':row['trajectory'],'source_trajectory_snapshot_sha256':tracehash}
        env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
            control_hz=10,assist_grasp=True,seed=row['seed'],required_success_steps=1,use_semantic_success=True,
            success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        try:
            rng=np.random.default_rng(row['seed']);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            task=next(t for t in tasks if t.name==row['task']);obs=env.reset(task)
            names={int(body):'object:'+name for name,body in env.object_ids.items()}
            names.update({int(body):'ground' for body in env.ground_bodies});names[int(env.robot_id)]='robot_arm'
            guard=env.gripper.physics_guard;offending=[]
            def capture():
                try:return guard()
                except PhysicsSafetyViolation:
                    for camera in ('overview','front','wrist'):
                        image=env.render(camera);name=f'actual_BEFORE_rollback_{camera}.png'
                        Image.fromarray(image).save(folder/name);offending.append(name)
                    raise
            env.gripper.physics_guard=capture
            # Rendering is unnecessary except at the exact offending state.
            # It does not change the motor/simulation stepping or scene64.
            env.render_observations=False;maximum=0.;info={}
            for i,entry in enumerate(trace):
                observed=scene_state(env,obs);expected=np.asarray(entry['pre_action_scene64'],np.float32)
                error=float(np.max(np.abs(observed-expected)));maximum=max(maximum,error)
                if error>1e-6:raise ValueError(f'Pre-action64 mismatch at saved step{i+1}: {error}')
                obs,_,done,info=env.step(np.asarray(entry['action'],np.float32))
                if done and i+1!=len(trace):raise ValueError('Recorded command replay terminated earlier; not the same path')
            if info.get('safety_violation')!=row['safety_violation'] or not info.get('rollback_restored'):
                raise ValueError('Exact original terminal safety event/rollback was not reproduced')
            diagnostics=info.get('safety_diagnostics_before_rollback')
            if not diagnostics or len(offending)!=3:raise ValueError('Missing actual pre-rollback physical evidence')
            summary=classify(diagnostics,names,set(env.gripper.PAD_LINKS))
            atomic_json(folder/'actual_pre_rollback_diagnostics.json',{'passed':True,'diagnostics':diagnostics,
                'classification':summary,'same_pre_action64_max_error':maximum,'requested_target':task.pick_name,
                'saved_neural_commands_not_changed':True,'offending_images':offending,
                'replay_NOT_new_model_episode_or_success':True})
            result.update(passed=True,pre_action64_max_error=maximum,classification=summary,
                actual_diagnostics=str((folder/'actual_pre_rollback_diagnostics.json').resolve()))
        except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
        finally:env.close()
        results.append(result);save('replaying_recorded_neural_safety_cases_CPU_only')
    if any(sha256_file(p)!=digest for p,digest in bound.items()):raise ValueError('Pinned physics/probe source changed')
    counts=Counter()
    for r in results:
        if r['passed']:
            for key in ('nonpad_object_contact','ground_contact','robot_arm_contact'):
                if r['classification'][key]:counts[key]+=1
    atomic_json(output/'audit.json',{'passed':bool(cases) and all(r['passed'] for r in results),
        'completed_replays':len(results),'planned_replays':len(cases),'diagnostic_counts':dict(counts),'rows':results,
        'scope':'recorded_neural_safety_replay_NOT_new_SR_or_proven_unique_root_causes','bound_files':bound,
        'new_neural_trials':0,'evaluator_and_guard_thresholds_modified':False})
    save('completed_all_replays' if all(r['passed'] for r in results) else 'completed_with_mismatches_preserved_NOT_zero_SR')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--progress',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('Preserve every actual replay')
        cmd=[str(SIM),'-m','VLA.audit_neural_safety_replay_r37','--output',str(a.output.resolve())]
        for path in a.progress:cmd+=['--progress',str(path.resolve())]
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'CPU_only_recorded_commands_not_new_model_trials':True});print(child.pid)
