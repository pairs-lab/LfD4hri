"""Actual native full64-input VLA robot integration8 or locked dev40/120."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.benchmark import _interaction_audit,_save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.recorder import VideoRecorder,NullVideoRecorder
from VLA.common.tasks import TaskSpec,select_tasks,randomize_rl_scene
from VLA.common.evaluation_statistics import success_statistics
from VLA.scene64_native_client import Scene64NativePolicy
from VLA.run_cross_vla120 import SEEDS,PROFILE_PROTOCOL


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('pi0','pi0_fast','openvla','cogact'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-source',type=Path);p.add_argument('--trials-per-action',type=int,choices=(10,30),default=30)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('No reroll or silent replacement of actual episodes')
    a.output.mkdir(parents=True);cp=a.checkpoint.resolve();profile=checkpoint_profile(cp)
    protocol=json.loads(PROFILE_PROTOCOL.read_text())
    if profile is None or profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Same robot/objects/camera/physics required')
    smoke=a.smoke_source is not None;rows=[];controls=[];planned=0 if smoke else 4*a.trials_per_action
    def save(status,error=None):
        atomic_json(a.output/'progress.json',{'status':status,'model':a.model,'planned_trials':planned,'completed_trials':len(rows),
            'successes':sum(r['success'] for r in rows),'rows':rows,'neural_controls':len(controls),
            'controls':controls if smoke else [],'integration8_NOT_SR':smoke,'error':error,
            'same_locked_development_seeds':True,'final_test_used':False,'oracle_action_fallback':False,
            'statistics':success_statistics(rows) if rows and len(rows)%4==0 else None})
    save('running');source_rows=json.loads((a.smoke_source/'index.json').read_text()) if smoke else None
    try:
        for seed in ([20261015] if smoke else SEEDS[:a.trials_per_action]):
            if smoke:tasks=[TaskSpec.from_dict(next(r for r in source_rows if r['task']['name']==t.name)['task']) for t in select_tasks('all')]
            else:
                rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            folder=a.output/f'test_seed{seed}';folder.mkdir();traces={}
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,control_hz=10,
                assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            policy=None
            try:
                policy=Scene64NativePolicy(env,a.model,str(cp),a.port,seed)
                if not policy.server_metadata['target_domain_finetuned']:raise ValueError('Never score official weights as target-finetuned')
                video=VideoRecorder(folder/'rollout.mp4',fps=10,overview_render=lambda:env.render('overview')) if seed in (20261015,SEEDS[0]) else NullVideoRecorder()
                with video:
                    for task in tasks:
                        obs=env.reset(task);policy.reset(task);trace=[];done=False
                        if env.is_semantic_success():raise ValueError('Initial success; preserve invalid scene, never reroll')
                        _save_observation_snapshot(folder,task.name,obs,env)
                        while not done and (not smoke or len(trace)<2):
                            before=_interaction_audit(env);action=policy.predict(obs)
                            if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Malformed neural command')
                            raw=policy.last_input['scene64'].copy();obs,_,done,info=env.step(action)
                            entry={'task':task.name,'step':obs.step,'action':action.tolist(),'native_neural_action':policy.last_native.tolist(),
                                'pre_action_scene64':raw.tolist(),'actual_model_prompt':policy.last_input['prompt'],'before':before,'after':_interaction_audit(env),
                                'success':bool(info['success']),'safety_violation':info.get('safety_violation')}
                            controls.append(entry);trace.append(entry);video.add(obs,a.model+'+SimScene | '+task.prompt)
                            if smoke or obs.step%10==0:_save_observation_snapshot(folder,task.name,obs,env)
                            traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                            if smoke and done:raise ValueError('Native integration terminated early; actual controls retained')
                        _save_observation_snapshot(folder,task.name,obs,env)
                        if not smoke:rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                            'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                        save('running')
            finally:
                if policy:policy.close()
                env.close()
        if smoke and len(controls)!=8:raise ValueError('Incomplete actual native control gate')
        if not smoke and len(rows)!=planned:raise ValueError('Incomplete locked learned cohort')
        save('completed8_integration_NOT_SR' if smoke else f'completed{planned}_privileged_native_VLA')
    except Exception as error:save('failed_preserve_actual_trials',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
