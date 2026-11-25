"""Shared frozen PyBullet cohort, real neural queues, no final media files."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.benchmark import _interaction_audit
from VLA.new_vlas.evaluate import png
from VLA.scene64_conditioning import model_input
from VLA.privileged_visual_r53.scene_map import render
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.cross_vla800_r62.media import discard_snapshot
from .common import SCHEMA,HORIZONS,read,utc


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);cohort=read(a.cohort);seeds=cohort['seeds']
    if len(set(seeds))!=len(seeds) or not 1<=a.prefix<=HORIZONS[a.model]:
        raise ValueError('Explicit unique cohort and valid native prefix required')
    session=requests.Session();session.trust_env=False;url=f'http://127.0.0.1:{a.port}'
    meta=session.get(url+'/health',timeout=30).json()
    if meta['schema']!=SCHEMA or meta['model']!=a.model or meta['native_horizon']!=HORIZONS[a.model]:
        raise ValueError('Wrong actual native server')
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Frozen observation profile changed')
    atomic_json(a.output/'plan.json',{'model':a.model,'metadata':meta,'cohort':str(a.cohort.resolve()),
        'role':cohort['role'],'seeds':seeds,'prefix':a.prefix,'same_robot_physics_safety_success':True,
        'PUT_requires_release':False,'new_Oracle_or_action_repair':False,'video_saved':False})
    rows=[];queries=controls=0;journal=(a.output/'NEURAL_QUERY_RECEIPTS.jsonl').open('x')
    def progress(status,**kw):
        atomic_json(a.output/'progress.json',{'status':status,'model':a.model,'checkpoint':meta['checkpoint'],
            'completed_trials':len(rows),'planned_trials':4*len(seeds),'rows':rows,'UTC':utc(),
            'role':cohort['role'],'native_queries':queries,'motor_controls':controls,
            'used_for_DEVELOPMENT_selection':cohort['role']!='untouched_final',**kw})
    try:
        progress('running_actual_native_rollouts')
        for seed in seeds:
            rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            folder=a.output/f'test_seed{seed}';folder.mkdir();traces={}
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            try:
                for task in tasks:
                    obs=env.reset(task)
                    if env.is_semantic_success():raise ValueError('Initially successful episode, no reroll')
                    discard_snapshot(folder,task.name,obs,env)
                    queue=deque();trace=[];done=False;query_index=0;receipt=None
                    while not done:
                        state=model_input(env,obs)
                        if not queue:
                            payload={'instruction':obs.prompt,'scene64':state['scene64'].tolist(),
                                'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                                'seed':seed,'inference_index':query_index}
                            start=time.monotonic();response=session.post(url+'/act',json=payload,timeout=1200)
                            response.raise_for_status();result=response.json();digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            gt=render(state['scene64'],task.name);gtsha=hashlib.sha256(gt.tobytes()).hexdigest()
                            if (not result.get('actual_neural_model') or result['input_sha256']!=digest
                                or result['map_RGB_sha256']!=gtsha or result['native_views']!=3):
                                raise ValueError('Actual native input/GT3 receipt mismatch')
                            native=result['actual_native_input_receipt']
                            if a.model=='vla_jepa':
                                if len(native['actual_Qwen_image_grid_thw'])!=3 or native['actual_DiT_state_dimensions']!=67:
                                    raise ValueError('Actual Qwen/DiT privileged feature consumption failed')
                            elif (native['actual_native_views']!=3 or native['actual_state_dimensions']!=32 or
                                  not native['complete_scene64_text_verified']):
                                raise ValueError('Actual native FAST privileged feature consumption failed')
                            actions=np.asarray(result['actions'],np.float32)
                            if actions.shape!=(HORIZONS[a.model],7) or not np.isfinite(actions).all():
                                raise ValueError('Native chunk malformed')
                            journal.write(json.dumps({'seed':seed,'task':task.name,'input_sha256':digest,
                                'actions':actions.tolist(),'GT_map_SHA256':gtsha,'native_views':3,
                                'actual_native_input_receipt':native,
                                'wall_latency_s':time.monotonic()-start})+'\n');journal.flush()
                            queue.extend(actions[:a.prefix].copy());queries+=1;query_index+=1
                            receipt={'query_step':int(obs.step),'input_sha256':digest}
                        action=queue.popleft();before=_interaction_audit(env)
                        obs,_,done,info=env.step(action);controls+=1
                        trace.append({'task':task.name,'step':int(obs.step),'action':action.tolist(),
                            'pre_action_scene64':state['scene64'].tolist(),'before':before,'after':_interaction_audit(env),
                            'success':bool(info['success']),'neural_receipt':receipt,
                            'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback'),
                            'executed_canonical_action':info.get('executed_canonical_action')})
                        if seed==seeds[0]:env.render('overview')  # Original mask/render cadence; no file IO.
                        if a.smoke and len(trace)==2:break
                    traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),
                        'steps':len(trace),'safety_violation':info.get('safety_violation'),
                        'trajectory':str(folder/'trajectory.json')})
                    if a.smoke and (len(trace)!=2 or info.get('safety_violation')):
                        raise ValueError('Own native smoke stopped early/unsafe, not a success-rate trial')
                    progress('running_actual_native_rollouts')
            finally:env.close()
        if a.smoke:
            if controls!=8:raise ValueError('Own actual8-control integration incomplete')
            atomic_json(a.output/'SMOKE_NOT_SR.json',{'passed':True,'actual_native_controls':8,
                'model_success_rate_established':False,'scope':'DEVELOPMENT integration only NOT success-rate data'})
        progress('completed_actual_cohort')
    except Exception as e:
        progress('integration_error_preserve_actual_trials_NOT_fake_failure',error=f'{type(e).__name__}: {e}')
        raise
    finally:journal.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=tuple(HORIZONS),required=True)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--prefix',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
