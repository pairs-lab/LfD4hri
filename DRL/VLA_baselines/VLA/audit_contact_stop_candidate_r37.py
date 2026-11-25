"""Offline calibration of the preserved task-independent pad-contact stop.

NOT a deployed controller or VLA success-rate experiment. Uses the old
mechanics pilot's exact .7-normal/.1N rule, never a target/goal/success input.
Replays original neural controls only through their recorded abort horizon;
after any clamp those controls are OPEN LOOP, not fresh neural predictions.
Separately checks76 predeclared ordinary TRAIN demos (19 targets x4 tasks).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
TASKS=('touch','pick','move','pick_place')


def project(action,contacts,pads):
    """Current contact+OPEN/down only; no object identities or task arguments."""
    value=np.asarray(action,np.float32).copy()
    if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('Finite original7D command required')
    upward=any(c[3] in pads and c[7][2]>.7 and c[9]>.1 for c in contacts)
    stop=bool(value[6]<-.2 and value[2]<0 and upward)
    if stop:value[2]=0
    return value,stop


def run(a):
    import pybullet as p
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.profile_binding import checkpoint_profile
    from VLA.common.tasks import select_tasks,randomize_rl_scene,TaskSpec
    from VLA.dataset_replay_context import audit_environment,reset_source_episode
    from VLA.common.physics_safety import physics_diagnostics
    from VLA.simstate_scene64 import scene_state
    from VLA.run_cross_vla120 import PROFILE_PROTOCOL
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[];cases=[];results=[];protocol=json.loads(PROFILE_PROTOCOL.read_text())
    projection=json.loads((PROJECTION/'audit.json').read_text());source=Path(projection['source'])
    originals=json.loads((source/'index.json').read_text());proofs={r['file']:r for r in projection['rows']}
    bound={str(path):sha256_file(path) for path in (Path(__file__),PROJECTION/'audit.json',PROFILE_PROTOCOL,
        ROOT/'VLA/common/body_environment.py',ROOT/'VLA/common/body_gripper.py',ROOT/'VLA/common/physics_safety.py',
        ROOT/'VLA/audits/body_grasp_v9_20261008/mechanics_followup/contact_force_cases.py')}
    for progress in a.progress:
        raw=progress.read_bytes();d=json.loads(raw);sources.append(str(progress.resolve()))
        for row in d['rows']:
            if not row.get('safety_violation'):continue
            payload=Path(row['trajectory']).read_bytes();trace=json.loads(payload)[row['task']]
            if len(trace)!=row['steps'] or trace[-1].get('safety_violation')!=row['safety_violation']:raise ValueError('Need complete actual safety path')
            cases.append({'kind':'neural_OPEN_LOOP_after_intervention_NOT_SR','model':d['model'],'checkpoint':d['checkpoint'],
                'row':row,'trace':trace,'source_trajectory':row['trajectory']})
    for task in TASKS:
        names=set()
        for row in originals:
            if row['task']['name']!=task or row.get('paired_scene_source') or row.get('recovery_start_actions_not_training_labels'):continue
            name=row['task']['pick_name']
            if name in names:continue
            names.add(name);cases.append({'kind':'TRAIN_teacher_demo_compatibility_NOT_VLA_SR','model':'recorded_teacher','row':row})
        if len(names)!=19:raise ValueError('All19 target types per task required; do not filter by result')
    atomic_json(out/'plan.json',{'sources':sources,'neural_safety_paths':sum(c['model']!='recorded_teacher' for c in cases),
        'predeclared_TRAIN_demo_paths':76,'all19_target_types_all4_tasks':True,'planned_cases':len(cases),
        'candidate_rule_inherited_without_threshold_tuning':{'open_below':-.2,'down_below':0,'pad_normal_z_above':.7,'force_N_above':.1},
        'candidate_is_NOT_deployed':True,'new_model_SR':None,'bound_files':bound})
    def save(status):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'completed_cases':len(results),'planned_cases':len(cases),
            'rows':results,'new_neural_trials':0,'new_model_SR':None,'production_or_running_code_changed':False})
    save('running_CPU_contact_stop_calibration')
    for ci,case in enumerate(cases):
        row=case['row'];neural=case['model']!='recorded_teacher';folder=out/f'case{ci:03d}';folder.mkdir()
        result={k:v for k,v in case.items() if k not in ('trace','row')};result.update(task=row['task'] if neural else row['task']['name'],
            passed_replay=False,interventions=[],recorded_controls=row['steps'])
        if neural:
            seed=row['seed'];profile=checkpoint_profile(Path(case['checkpoint']))
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        else:env=audit_environment(source,image_size=384,control_hz=10,seed=0,required_success_steps=1,use_semantic_success=False)
        try:
            if neural:
                rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
                task=next(t for t in tasks if t.name==row['task']);obs=env.reset(task)
                actions=np.asarray([r['action'] for r in case['trace']],np.float32)
                states=np.asarray([r['pre_action_scene64'] for r in case['trace']],np.float32)
                result['requested_object']=task.pick_name
            else:
                path=source/row['file'];statepath=PROJECTION/'states'/row['file'];proof=proofs[row['file']]
                if sha256_file(path)!=proof['source_payload_sha256'] or sha256_file(statepath)!=proof['state_payload_sha256']:
                    raise ValueError('Selected actual TRAIN source changed')
                with np.load(path,allow_pickle=False) as z:actions=z['canonical_actions'].astype(np.float32)
                with np.load(statepath,allow_pickle=False) as z:states=z['state'].copy()
                obs=reset_source_episode(env,source,row);result['requested_object']=row['task']['pick_name'];result['source_file']=str(path)
            env.render_observations=False;maximum=0.;info={};modified=False;executed=0;steps=[]
            for fi,action in enumerate(actions):
                if not modified:
                    error=float(np.max(np.abs(scene_state(env,obs)-states[fi])));maximum=max(maximum,error)
                    if error>1e-6:raise ValueError(f'Unchanged prefix differs at{fi+1}: {error}')
                applied,stop=project(action,p.getContactPoints(bodyA=env.gripper.body),set(env.gripper.PAD_LINKS))
                if stop:
                    modified=True;result['interventions'].append({'step':fi+1,'requested_dz':float(action[2]),'applied_dz':0.})
                obs,_,done,info=env.step(applied);executed+=1
                steps.append({'step':fi+1,'original_recorded_action':action.tolist(),'applied_action':applied.tolist(),
                    'current_contact_boundary_stop':stop,'safety_violation':info.get('safety_violation')})
                if done:break
            atomic_json(folder/'applied_commands.json',steps)
            result.update(passed_replay=True,unchanged_prefix_max_error=maximum,executed_controls=executed,
                safety_violation=info.get('safety_violation'),new_model_success_rate=None,
                max_guard_recorded_mimic_error_rad=float(env.safety_max_mimic_error),
                safe_through_original_abort_horizon=bool(neural and not info.get('safety_violation') and executed==len(actions)),
                expert_demo_success_NOT_VLA_SR=bool(not neural and info.get('success')),
                controls_after_first_clamp_are_NOT_new_closed_loop_neural_outputs=bool(neural and modified))
            if info.get('safety_diagnostics_before_rollback'):result['actual_pre_rollback']=info['safety_diagnostics_before_rollback']
        except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
        finally:env.close()
        results.append(result);save('running_CPU_contact_stop_calibration')
    if any(sha256_file(path)!=digest for path,digest in bound.items()):raise ValueError('Pinned source/controller changed')
    neural=[r for r in results if r['model']!='recorded_teacher'];demos=[r for r in results if r['model']=='recorded_teacher']
    atomic_json(out/'audit.json',{'passed_execution':all(r['passed_replay'] for r in results),'candidate_NOT_deployed':True,
        'neural_recorded_safety_paths':len(neural),'safe_through_original_abort_horizon_NOT_model_success':sum(r.get('safe_through_original_abort_horizon',False) for r in neural),
        'TRAIN_demo_paths':len(demos),'TRAIN_expert_successes_NOT_VLA_SR':sum(r.get('expert_demo_success_NOT_VLA_SR',False) for r in demos),
        'TRAIN_paths_with_clamped_commands':sum(bool(r['interventions']) for r in demos),
        'new_neural_trials':0,'new_model_SR':None,'rows':results,'bound_files':bound,
        'cannot_attribute_model_SR_improvement_from_openloop_replays':True})
    save('completed_offline_calibration_NOT_production_admission')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--progress',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('Preserve all physical calibration attempts')
        cmd=[str(SIM),'-m','VLA.audit_contact_stop_candidate_r37','--output',str(a.output.resolve())]
        for path in a.progress:cmd+=['--progress',str(path.resolve())]
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'offline_calibration_only_NO_production_change':True});print(child.pid)
