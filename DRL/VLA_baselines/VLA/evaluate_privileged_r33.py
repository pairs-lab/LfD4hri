"""All-family actual neural controls on the unchanged locked PyBullet task.

Native8 integration or development40/120. No text is drawn into the video.
No oracle/IK high-level controller, rerolls or relaxed scoring.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from VLA.common.artifact_io import atomic_json
from VLA.benchmark import _interaction_audit,_save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.tasks import TaskSpec,select_tasks,randomize_rl_scene
from VLA.common.evaluation_statistics import success_statistics
from VLA.scene64_r33_client import Scene64R33Policy
from VLA.run_cross_vla120 import SEEDS,PROFILE_PROTOCOL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('openvla','rdt','cogact','pi0','pi0_fast'),required=True)
    for name in ('checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--smoke-source',type=Path)
    p.add_argument('--trials-per-action',type=int,choices=(10,30),default=30);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Never reroll an existing cohort')
    a.output.mkdir(parents=True);cp=a.checkpoint.resolve();profile=checkpoint_profile(cp)
    protocol=json.loads(PROFILE_PROTOCOL.read_text())
    if profile is None or profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Same frozen environment profile required')
    smoke=a.smoke_source is not None;rows=[];controls=0;metadata=None
    def save(status,error=None):
        atomic_json(a.output/'progress.json',{'status':status,'model':a.model,'checkpoint':str(cp),
            'completed_trials':len(rows),'planned_trials':0 if smoke else a.trials_per_action*4,
            'successes':sum(r['success'] for r in rows),'rows':rows,'neural_controls':controls,'error':error,
            'integration8_NOT_SR':smoke,'privileged_inputs':True,'same_locked_development_seeds':True,
            'final_test_used':False,'oracle_action_fallback':False,'video_contains_overlays':False,
            'statistics':success_statistics(rows) if rows and len(rows)%4==0 else None})
    source_rows=json.loads((a.smoke_source/'index.json').read_text()) if smoke else None;save('running')
    try:
        for seed in ([20261015] if smoke else SEEDS[:a.trials_per_action]):
            if smoke:tasks=[TaskSpec.from_dict(next(r for r in source_rows if r['task']['name']==t.name)['task']) for t in select_tasks('all')]
            else:
                rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            folder=a.output/f'test_seed{seed}';folder.mkdir();traces={};video=None;policy=None
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,control_hz=10,
                assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            try:
                policy=Scene64R33Policy(env,a.model,str(cp),a.port,seed)
                metadata=policy.server_metadata
                if metadata.get('target_domain_finetuned') is not True:raise ValueError('Unfitted checkpoint is not a post-fit baseline')
                atomic_json(a.output/'actual_server_metadata.json',metadata)
                if seed in (20261015,SEEDS[0]):video=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                for task in tasks:
                    obs=env.reset(task);policy.reset(task);trace=[];done=False
                    if env.is_semantic_success():raise ValueError('Initial success scene invalid; never replace it')
                    _save_observation_snapshot(folder,task.name,obs,env)
                    while not done and (not smoke or len(trace)<2):
                        before=_interaction_audit(env);action=policy.predict(obs)
                        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Malformed neural command')
                        raw=policy.last_input['scene64'].copy();obs,_,done,info=env.step(action)
                        entry={'task':task.name,'step':obs.step,'action':action.tolist(),'native_neural_action':policy.last_native.tolist(),
                            'pre_action_scene64':raw.tolist(),'before':before,'after':_interaction_audit(env),
                            'success':bool(info['success']),'safety_violation':info.get('safety_violation')}
                        controls+=1;trace.append(entry);traces[task.name]=trace
                        atomic_json(folder/'trajectory.json',traces)
                        if video:video.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                        if smoke or obs.step%10==0:_save_observation_snapshot(folder,task.name,obs,env)
                        if smoke and done:raise ValueError('Actual smoke control terminated early; preserve safety event')
                    if not smoke:rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),
                        'steps':len(trace),'safety_violation':info.get('safety_violation'),'trajectory':str((folder/'trajectory.json').resolve())})
                    save('running')
            finally:
                if video:video.close()
                if policy:policy.close()
                env.close()
        if smoke and controls!=8:raise ValueError('Actual native8 incomplete')
        if not smoke and len(rows)!=4*a.trials_per_action:raise ValueError('Actual learned cohort incomplete')
        save('completed8_integration_NOT_SR' if smoke else f'completed{len(rows)}_privileged_native_VLA')
    except Exception as error:save('failed_preserve_actual_trials',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
