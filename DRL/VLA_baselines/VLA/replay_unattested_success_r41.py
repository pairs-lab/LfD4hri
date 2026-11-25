"""Substep attachment audit of two saved DEVELOPMENT target-region successes.

Replays RECORDED genuine VLA controls in a separate CPU PyBullet world and
requires EVERY before-action64 state and per-step outcome to reproduce1e-6.
No NN/oracle/new trial SR. Read-only event hooks distinguish brief within-
control grasps from actually never-attached displacement; no policy repair.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA import evaluate_smol800_r39 as cohort
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import TaskSpec
from VLA.simstate_scene64 import scene_state

BASE=cohort.BASE
SOURCE=BASE/'scene67_resume_parallel_r34/parallel_attempt/development120/progress.json'
CASES=((20268005,'move'),(20268023,'pick_place'))


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(cohort.PROFILE_PROTOCOL.read_text());profile=json.loads(Path(protocol['observation_profile_file']).read_text())
    progress=json.loads(SOURCE.read_text());results=[]
    for seed,task_name in CASES:
        row=next(r for r in progress['rows'] if (r['seed'],r['task'])==(seed,task_name))
        if not row['success'] or row['safety_violation']:raise ValueError('Declared successful no-abort trace required')
        trajectory=Path(row['trajectory']);trace=json.loads(trajectory.read_text())[task_name];digest=sha256_file(trajectory)
        env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
            control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
            success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
            semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        events=[];counter={'control':0,'tick':0,'correct_attached_ticks':0,'current_run':0,'longest_run':0};matched=0
        try:
            for item in cohort.trial_specs([seed]):
                obs=env.reset(TaskSpec.from_dict(item['task']))
                if item['task']['name']==task_name:break
            create,release,post=env.gripper._create_assisted_grasp,env.gripper._release_assisted_grasp,env.gripper.post_step
            def on_create(body):
                events.append({'event':'create','control':counter['control'],'tick':counter['tick'],
                    'correct_requested_object':body==env.pick_id,'object':next((n for n,b in env.object_ids.items() if b==body),None)})
                return create(body)
            def on_release():
                if env.gripper.grasped_object is not None:
                    events.append({'event':'release','control':counter['control'],'tick':counter['tick'],
                        'correct_requested_object':env.gripper.grasped_object==env.pick_id})
                return release()
            def on_post():
                result=post();counter['tick']+=1
                held=env.gripper.grasped_object==env.pick_id
                counter['correct_attached_ticks']+=int(held)
                counter['current_run']=counter['current_run']+1 if held else 0
                counter['longest_run']=max(counter['longest_run'],counter['current_run'])
                return result
            env.gripper._create_assisted_grasp=on_create;env.gripper._release_assisted_grasp=on_release
            env.gripper.post_step=on_post
            for frame in trace:
                counter['control']+=1
                np.testing.assert_allclose(scene_state(env,obs),frame['pre_action_scene64'],rtol=0,atol=1e-6)
                obs,_,done,info=env.step(np.asarray(frame['action'],np.float32))
                if bool(info['success'])!=bool(frame['success']) or info.get('safety_violation')!=frame.get('safety_violation'):
                    raise ValueError('Original recorded control outcome did not reproduce')
                matched+=1
            if not done or not info['success']:raise ValueError('Original terminal region success did not reproduce')
            timestep=p.getPhysicsEngineParameters()['fixedTimeStep']
            result={'seed':seed,'task':task_name,'original_region_success':True,'all_before64_and_outcomes_reproduced_at1e6':True,
                'replayed_recorded_VLA_controls_NOT_new_NN_trials':matched,'substep_events':events,
                'observed_correct_assisted_attachment_substeps':counter['correct_attached_ticks'],
                'longest_correct_attachment_seconds':counter['longest_run']*timestep,
                'physics_fixed_timestep_seconds':timestep,'source_trajectory_sha256':digest,
                'never_correctly_attached_at_ANY_recorded_physics_substep':not any(e['event']=='create' and e['correct_requested_object'] for e in events)}
            if sha256_file(trajectory)!=digest:raise ValueError('Source changed')
            results.append(result);atomic_json(out/'progress.json',{'completed':len(results),'planned':2,'rows':results})
        finally:env.close()
    atomic_json(out/'audit.json',{'passed_actual_replay':True,'scope':'substep_CPU_replay_diagnostics_NOT_new_neural_SR',
        'rows':results,'original_region_scores_not_relabelled':True,'model_or_oracle_loaded':False,
        'physics_controller_thresholds_and_actions_unchanged':True,
        'implementation_sha256':sha256_file(Path(__file__)),'original_progress_sha256':sha256_file(SOURCE)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
