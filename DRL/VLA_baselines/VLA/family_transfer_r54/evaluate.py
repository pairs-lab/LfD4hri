"""Predeclared cohorts, native neural queues, unchanged shared PyBullet rules."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import requests
from VLA.benchmark import _interaction_audit,_save_observation_snapshot
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import model_input
from VLA.new_vlas.evaluate import png
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.privileged_visual_r53.scene_map import render
from .common import SCHEMA,TASKS,DATA,read


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    cohort=read(a.cohort);seeds=cohort['seeds']
    if len(seeds)!=len(set(seeds)) or not 1<=a.prefix<=5:raise ValueError('Explicit unique seeds/native prefix1..5 required')
    session=requests.Session();session.trust_env=False;url=f'http://127.0.0.1:{a.port}'
    response=session.get(url+'/health',timeout=30);response.raise_for_status();meta=response.json()
    if meta['schema']!=SCHEMA or meta['model']!=a.model or meta['native_horizon']!=5:
        raise ValueError('Wrong actual model server')
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Frozen physical/camera profile changed')
    data_proof=read(DATA/'audit.json')
    projection=next(Path(p) for p in data_proof['bound_files'] if p.endswith('full_train64/audit.json'))
    source=read(projection);manifest=Path(source['source'])/'dataset_manifest.json'
    if sha256_file(manifest)!=source['source_manifest_sha256'] or profile_hash(read(manifest)['observation_profile'])!=profile_hash(profile):
        raise ValueError('Actual TRAIN/runtime camera/home/robot profile mismatch')
    atomic_json(a.output/'plan.json',{'model':a.model,'metadata':meta,'cohort':str(a.cohort.resolve()),
        'cohort_sha256':sha256_file(a.cohort),'prefix':a.prefix,'scope':cohort['scope'],
        'same_environment_controller_robot_gripper_success_horizon':True,'new_action_repair_or_Oracle':False,
        'primary_place_success_does_NOT_require_release':True,'environment_fingerprint':environment_runtime_fingerprint(profile),
        'pybullet_sha256':sha256_file(__import__('pybullet').__file__)})
    rows=[];queries=controls=0
    def progress(status,**kw):
        atomic_json(a.output/'progress.json',{'status':status,'model':a.model,'checkpoint':meta['checkpoint'],
            'metadata':meta,'cohort':str(a.cohort.resolve()),'scope':cohort['scope'],'prefix':a.prefix,
            'completed_trials':len(rows),'planned_trials':4*len(seeds),'successes':sum(r['success'] for r in rows),
            'rows':rows,'native_queries':queries,'motor_controls':controls,
            'statistics':success_statistics(rows) if rows and len(rows)%4==0 else None,**kw})
    try:
        progress('running_actual_native_rollouts')
        for seed in seeds:
            rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            folder=a.output/f'test_seed{seed}';folder.mkdir();traces={};writer=None
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            try:
                if seed==seeds[0]:writer=imageio.get_writer(str(folder/'rollout.mp4'),fps=10,codec='libx264',quality=8)
                for task in tasks:
                    obs=env.reset(task)
                    if env.is_semantic_success():raise ValueError('Initially successful scene; no reroll')
                    _save_observation_snapshot(folder,task.name,obs,env)
                    queue=deque();trace=[];done=False;index=0;receipt=None
                    while not done:
                        state=model_input(env,obs)
                        if not queue:
                            payload={'instruction':obs.prompt,'scene64':state['scene64'].tolist(),
                                'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                                'seed':seed,'inference_index':index}
                            reply=session.post(url+'/act',json=payload,timeout=1200);reply.raise_for_status();result=reply.json()
                            digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            if not result.get('actual_neural_model') or result.get('input_sha256')!=digest:
                                raise ValueError('Actual native model/input identity missing')
                            if meta['GT3_valid']:
                                gt=render(state['scene64'],task.name)
                                if result.get('native_views')!=3 or result.get('map_RGB_sha256')!=hashlib.sha256(gt.tobytes()).hexdigest():
                                    raise ValueError('Actual third GT input differs from identical causal TRAIN transform')
                            action=np.asarray(result['actions'],np.float32)
                            if action.shape!=(5,7) or not np.isfinite(action).all():raise ValueError('Malformed native chunk')
                            queue.extend(action[:a.prefix].copy());queries+=1;index+=1
                            receipt={'query_step':int(obs.step),'input_sha256':digest,'map_RGB_sha256':result.get('map_RGB_sha256')}
                        action=queue.popleft();before=_interaction_audit(env)
                        before.update(privileged_policy_input=True,R54_input_mode=meta['input_mode'])
                        obs,_,done,info=env.step(action);controls+=1
                        trace.append({'task':task.name,'step':int(obs.step),'action':action.tolist(),'before':before,
                            'after':_interaction_audit(env),'pre_action_scene64':state['scene64'].tolist(),'neural_receipt':receipt,
                            'success':bool(info['success']),'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback'),
                            'executed_canonical_action':info.get('executed_canonical_action')})
                        traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                        if writer:writer.append_data(np.concatenate((env.render('overview'),obs.front_rgb,obs.wrist_rgb),axis=1))
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'trajectory':str(folder/'trajectory.json')})
                    progress('running_actual_native_rollouts')
            finally:
                if writer:writer.close()
                env.close()
        progress('completed_actual_cohort')
    except Exception as e:progress('failed_integration_preserve_all_completed_and_partial_NOT_fake_failure',error=f'{type(e).__name__}: {e}');raise
    finally:session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('smolvla','pi0'),required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--prefix',type=int,required=True)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
