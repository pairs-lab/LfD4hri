"""Same R62 plant/metric; separate DEVELOPMENT and frozen final800 admission."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import requests
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.benchmark import _interaction_audit
from VLA.new_vlas.evaluate import png
from VLA.scene64_conditioning import model_input
from VLA.privileged_visual_r53.scene_map import render
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.cross_vla800_r62.common import SEEDS
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from .common import bindings,verify,read,PORT


def main(a):
    cohort=read(a.cohort);seeds=cohort['seeds'];final=cohort['role']=='final_fixed_evaluation800'
    if final:
        method=read(a.frozen_method)
        if seeds!=SEEDS or a.smoke_controls or a.prefix!=method['prefix']:
            raise ValueError('Final800 cohort/method must remain fixed')
    elif cohort['role'] not in ('development_selection','development_confirmation'):
        raise ValueError('Unknown cohort role')
    if not 1<=a.prefix<=50:raise ValueError('Invalid prefix')
    session=requests.Session();session.trust_env=False;url=f'http://127.0.0.1:{a.port}'
    response=session.get(url+'/health',timeout=15);response.raise_for_status();meta=response.json()
    if meta['model']!='evo1_r80' or not meta['actual_neural_model']:raise ValueError('Wrong native model')
    if final and (meta['checkpoint']!=method['checkpoint'] or meta['checkpoint_sha256']!=method['checkpoint_sha256']):
        raise ValueError('Frozen model differs')
    bound=bindings();a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Shared camera profile changed')
    atomic_json(a.output/'PLAN.json',{'model':meta,'cohort':cohort,'prefix':a.prefix,'code':bound,
        'same_R62_UR5e_Robotiq_objects_physics_home_safety_metric':True,
        'success':protocol['success'],'PUT_requires_release_settle':False,
        'no_added_Oracle_action_controller':True,'smoke_controls_per_task':a.smoke_controls})
    rows=[];queries=controls=0;started=time.monotonic()
    journal=(a.output/'NEURAL_QUERIES.jsonl').open('x')
    def progress(status,**extra):
        per={t:{'successes':sum(r['success'] for r in rows if r['task']==t),
            'n':sum(r['task']==t for r in rows)} for t in ('touch','pick','move','pick_place')}
        for v in per.values():v['SR_percent']=None if a.smoke_controls or not v['n'] else 100*v['successes']/v['n']
        atomic_json(a.output/'progress.json',{'status':status,'completed_trials':len(rows),
            'planned_trials':4*len(seeds),'per_action':per,'rows':rows,
            'native_queries':queries,'motor_controls':controls,'checkpoint':meta['checkpoint'],
            'elapsed_wall_s':time.monotonic()-started,'final800':final,
            'robot_SR_measured':not bool(a.smoke_controls),**extra})
    try:
        progress('running_actual_neural_evaluation')
        for seed in seeds:
            verify(bound);rng=np.random.default_rng(seed)
            tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            folder=a.output/f'seed{seed}';folder.mkdir();traces={}
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            try:
                for task in tasks:
                    obs=env.reset(task)
                    if env.is_semantic_success():raise ValueError('Initially successful scene; never reroll')
                    discard_snapshot(folder,task.name,obs,env)
                    queue=deque();trace=[];index=0;done=False;events={};pictures={};closest=float('inf');digest=None
                    def capture(label):
                        if final:return
                        pictures[label]=(int(obs.step),[np.asarray(obs.front_rgb).copy(),np.asarray(obs.wrist_rgb).copy(),
                            render(model_input(env,obs)['scene64'],task.name)])
                    capture('reset')
                    while not done:
                        raw=model_input(env,obs)['scene64'];before=_interaction_audit(env)
                        if not queue:
                            payload={'instruction':obs.prompt,'scene64':raw.tolist(),
                                'ee_position':obs.ee_position.tolist(),'ee_rotation_6d':obs.ee_rotation_6d.tolist(),
                                'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                                'seed':seed,'inference_index':index}
                            t0=time.monotonic();res=session.post(url+'/act',json=payload,timeout=300);res.raise_for_status()
                            reply=res.json();digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            actual=reply['native_input_receipt'];gt=hashlib.sha256(render(raw,task.name).tobytes()).hexdigest()
                            if (not reply['actual_neural_model'] or reply['input_sha256']!=digest or
                                reply['map_RGB_sha256']!=gt or actual['numeric']!=[1,67] or actual['state']!=[1,91] or
                                actual['pixels']!=[3,3,448,448] or actual['future_labels_teacher_phase_or_success']):
                                raise ValueError('Actual neural input receipt incomplete')
                            action=np.asarray(reply['actions'],np.float32)
                            if action.shape!=(50,7) or not np.isfinite(action).all():raise ValueError('Malformed native chunk')
                            journal.write(json.dumps({'seed':seed,'task':task.name,'index':index,'input_sha256':digest,
                                'native_input':actual,'actions':action.tolist(),'latency_s':time.monotonic()-t0})+'\n');journal.flush()
                            queue.extend(action[:a.prefix].copy());queries+=1;index+=1
                        command=queue.popleft();obs,_,done,info=env.step(command);controls+=1
                        post=model_input(env,obs)['scene64'];after=_interaction_audit(env)
                        trace.append({'step':int(obs.step),'action':command.tolist(),
                            'executed_canonical_action':info['executed_canonical_action'],
                            'pre_action_scene64':raw.tolist(),'post_action_scene64':post.tolist(),
                            'before':before,'after':after,'success':bool(info['success']),
                            'safety_violation':info.get('safety_violation'),'query_sha256':digest})
                        distance=float(np.linalg.norm(post[11:14]))
                        if distance<closest:closest=distance;capture('closest_target')
                        for name,flag in [('contact',post[28:30].any()),('hold',after['correct_object_held']),
                            ('drop',before['correct_object_held'] and not after['correct_object_held'])]:
                            if flag and name not in events:events[name]=int(obs.step);capture(name)
                        if a.smoke_controls and len(trace)>=a.smoke_controls:break
                    capture('terminal');traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                    if pictures:
                        items=sorted(pictures.items(),key=lambda x:x[1][0]);canvas=Image.new('RGB',(768,282*len(items)),'white')
                        draw=ImageDraw.Draw(canvas)
                        for n,(label,(step,views)) in enumerate(items):
                            for col,image in enumerate(views):
                                canvas.paste(Image.fromarray(image).resize((256,256)),(col*256,n*282+26))
                                draw.text((col*256+3,n*282+5),f'{label} t{step}/'+('front','wrist','GT')[col],fill='black')
                        canvas.save(folder/(task.name+'_REVIEW.png'))
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'events':events,'minimum_target_distance_m':closest,
                        'trajectory':str(folder/'trajectory.json')})
                    progress('running_actual_neural_evaluation')
            finally:env.close()
        progress('completed_actual_smoke_NOT_SR' if a.smoke_controls else 'completed_actual_final800' if final else 'completed_actual_development')
    except Exception as error:
        progress('integration_error_NOT_zero_model_SR',error=str(error));raise
    finally:journal.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=int,default=10)
    p.add_argument('--smoke-controls',type=int,default=0);p.add_argument('--frozen-method',type=Path)
    p.add_argument('--port',type=int,default=PORT);main(p.parse_args())
