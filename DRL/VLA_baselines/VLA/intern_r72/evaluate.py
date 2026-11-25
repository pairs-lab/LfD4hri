"""Real native InternVLA actions in exactly the shared PyBullet plant."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
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
from VLA.intern_r72.common import read,bindings,verify


def main(a):
    cohort=read(a.cohort)
    if cohort['role'] not in ('development_selection','development_confirmation'):raise ValueError('Never consume final800 for selection')
    if not 1<=a.prefix<=50:raise ValueError('Bad execution prefix')
    bound=bindings();session=requests.Session();session.trust_env=False
    url='http://127.0.0.1:5940';meta=session.get(url+'/health',timeout=10).json()
    if meta['model']!='internvla_a15_r72' or not meta['actual_neural_model']:raise ValueError('Wrong actual policy server')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Shared camera profile changed')
    atomic_json(a.output/'PLAN.json',{'model':meta,'cohort':cohort,'prefix':a.prefix,'smoke_controls_per_task':a.smoke_controls,
        'same_R62_robot_gripper_plant_success_safety':True,'PUT_requires_release':False,'code':bound})
    rows=[];queries=controls=0
    journal=(a.output/'NATIVE_QUERIES.jsonl').open('x')
    def progress(status,**extra):
        per={t:{'successes':sum(r['success'] for r in rows if r['task']==t),
            'n':sum(r['task']==t for r in rows)} for t in ('touch','pick','move','pick_place')}
        for value in per.values():value['SR_percent']=None if a.smoke_controls or not value['n'] else 100*value['successes']/value['n']
        atomic_json(a.output/'progress.json',{'status':status,'completed_trials':len(rows),'planned_trials':4*len(cohort['seeds']),
            'per_action':per,'rows':rows,'native_queries':queries,'motor_controls':controls,
            'checkpoint':meta['checkpoint'],'robot_success_rate_measured':not bool(a.smoke_controls),**extra})
    try:
        for seed in cohort['seeds']:
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
                    if env.is_semantic_success():raise ValueError('Initially successful reset; preserve, never reroll')
                    queue=deque();trace=[];done=False;index=0;images={};events={};closest=float('inf');digest=None
                    def capture(label):
                        images[label]=(int(obs.step),[np.asarray(obs.front_rgb).copy(),np.asarray(obs.wrist_rgb).copy(),
                            render(model_input(env,obs)['scene64'],task.name)])
                    capture('reset')
                    while not done:
                        pre=model_input(env,obs)['scene64'];before=_interaction_audit(env)
                        if not queue:
                            payload={'instruction':obs.prompt,'scene64':pre.tolist(),'front_png':png(obs.front_rgb),
                                'wrist_png':png(obs.wrist_rgb),'seed':seed,'inference_index':index}
                            t0=time.monotonic();reply=session.post(url+'/act',json=payload,timeout=300)
                            reply.raise_for_status();r=reply.json()
                            digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            gt=hashlib.sha256(render(pre,task.name).tobytes()).hexdigest()
                            actual=r['native_input_receipt']
                            if not r['actual_neural_model'] or r['input_sha256']!=digest or r['map_RGB_sha256']!=gt:
                                raise ValueError('Actual neural input receipt differs')
                            if actual['numeric']!=[1,67] or actual['images']!=3 or actual['FAST_label_tokens']:
                                raise ValueError('Input dropped a view/state or included teacher FAST labels')
                            action=np.asarray(r['actions'],np.float32)
                            if action.shape!=(50,7) or not np.isfinite(action).all():raise ValueError('Malformed native action chunk')
                            journal.write(json.dumps({'seed':seed,'task':task.name,'index':index,'input_sha256':digest,
                                'native_input':actual,'actions':action.tolist(),'wall_latency_s':time.monotonic()-t0})+'\n');journal.flush()
                            queue.extend(action[:a.prefix]);queries+=1;index+=1
                        command=queue.popleft();obs,_,done,info=env.step(command);controls+=1
                        post=model_input(env,obs)['scene64'];after=_interaction_audit(env)
                        trace.append({'step':int(obs.step),'action':command.tolist(),'before':before,'after':after,
                            'pre_action_scene64':pre.tolist(),'post_action_scene64':post.tolist(),
                            'executed_canonical_action':info['executed_canonical_action'],'success':bool(info['success']),
                            'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback'),
                            'query_sha256':digest})
                        distance=float(np.linalg.norm(post[11:14]))
                        if distance<closest:closest=distance;capture('closest_target')
                        for label,flag in [('first_contact',post[28:30].any()),('first_correct_hold',after['correct_object_held']),
                            ('drop',before['correct_object_held'] and not after['correct_object_held'])]:
                            if flag and label not in events:events[label]=int(obs.step);capture(label)
                        if a.smoke_controls and len(trace)>=a.smoke_controls:break
                    capture('terminal');traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                    items=sorted(images.items(),key=lambda item:item[1][0])
                    sheet=Image.new('RGB',(768,len(items)*282),'white');draw=ImageDraw.Draw(sheet)
                    for n,(label,(step,views)) in enumerate(items):
                        for col,pix in enumerate(views):
                            sheet.paste(Image.fromarray(pix).resize((256,256)),(col*256,n*282+26))
                            draw.text((col*256+2,n*282+4),f'{label} t{step}/'+('front','wrist','GT')[col],fill='black')
                    path=folder/(task.name+'_REVIEW.png');sheet.save(path)
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'events':events,
                        'trajectory':str(folder/'trajectory.json'),'review':str(path)})
                    progress('running_actual_neural_controls' if a.smoke_controls else 'running_actual_development')
            finally:env.close()
        if a.smoke_controls and (controls!=8 or any(r['safety_violation'] for r in rows)):
            raise ValueError('Actual8-control safety gate failed; not a model SR measurement')
        progress('completed_actual_smoke_NOT_SR' if a.smoke_controls else 'completed_actual_development')
    except Exception as error:
        progress('integration_or_smoke_gate_error_NOT_zero_model_SR',error=str(error));raise
    finally:journal.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',type=int,default=5);p.add_argument('--smoke-controls',type=int,default=0);main(p.parse_args())
