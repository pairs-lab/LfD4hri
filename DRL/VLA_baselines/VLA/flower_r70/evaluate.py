"""Real closed-loop development trials, identical R62 plant/success protocol."""
import argparse
from collections import deque,Counter
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
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.flower_r70.common import read,freeze_test_code,verify


def main(a):
    cohort=read(a.cohort);seeds=cohort['seeds']
    if cohort['role'] not in ('development_selection','development_confirmation'):
        raise ValueError('This evaluator is NOT authorized to consume final800')
    if not 1<=a.prefix<=10 or len(set(seeds))!=len(seeds):raise ValueError('Bad prefix/cohort')
    binding=freeze_test_code();session=requests.Session();session.trust_env=False
    url=f'http://127.0.0.1:{a.port}'
    meta=session.get(url+'/health',timeout=15).json()
    if meta['model']!='flower_r69' or meta['native_views']!=3 or meta['numeric_dimensions']!=67:
        raise ValueError('Wrong model/input schema')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Shared camera profile changed')
    atomic_json(a.output/'PLAN.json',{'model':meta,'cohort':cohort,'prefix':a.prefix,
        'same_plant_as_R62_XVLA':True,'same_success':protocol['success'],'PUT_requires_release':False,
        'no_Oracle_or_action_repair':True,'video_saved':False,'code':binding})
    rows=[];queries=controls=0
    journal=(a.output/'NEURAL_QUERY_RECEIPTS.jsonl').open('x')
    def progress(state,**extra):
        per={t:{'successes':sum(r['success'] for r in rows if r['task']==t),
                    'n':sum(r['task']==t for r in rows)} for t in ('touch','pick','move','pick_place')}
        for value in per.values():value['SR_percent']=None if not value['n'] else 100*value['successes']/value['n']
        atomic_json(a.output/'progress.json',{'status':state,'completed_trials':len(rows),'planned_trials':4*len(seeds),
            'per_action':per,'rows':rows,'native_queries':queries,'motor_controls':controls,
            'checkpoint':meta['checkpoint'],'development_only_NOT_final_benchmark':True,**extra})
    try:
        progress('running_actual_development')
        for seed in seeds:
            verify(binding)
            rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
            folder=a.output/f'test_seed{seed}';folder.mkdir();traces={}
            env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            try:
                for task in tasks:
                    obs=env.reset(task)
                    if env.is_semantic_success():raise ValueError('Initially successful reset; preserve, do not reroll')
                    discard_snapshot(folder,task.name,obs,env)
                    queue=deque();trace=[];index=0;done=False;events={};pictures={};picture_steps={};queried=None
                    best_distance=float('inf')
                    def capture(label,replace=False):
                        if label in pictures and not replace:return
                        pictures[label]=[np.asarray(obs.front_rgb).copy(),np.asarray(obs.wrist_rgb).copy(),
                            render(model_input(env,obs)['scene64'],task.name)]
                        picture_steps[label]=int(obs.step)
                    capture('reset')
                    while not done:
                        state=model_input(env,obs)
                        if not queue:
                            payload={'instruction':obs.prompt,'scene64':state['scene64'].tolist(),
                                'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                                'seed':seed,'inference_index':index}
                            start=time.monotonic();response=session.post(url+'/act',json=payload,timeout=240)
                            response.raise_for_status();reply=response.json()
                            digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            expected_map=hashlib.sha256(render(state['scene64'],task.name).tobytes()).hexdigest()
                            actual=reply['actual_native_input_receipt']
                            if (not reply['actual_neural_model'] or reply['input_sha256']!=digest or
                                reply['map_RGB_sha256']!=expected_map or actual['numeric']!=[1,67] or
                                actual['images']!=[1,3,3,224,224] or actual['frequency_Hz']!=10):
                                raise ValueError('Native actual input consumption receipt differs')
                            action=np.asarray(reply['actions'],np.float32)
                            if action.shape!=(10,7) or not np.isfinite(action).all():raise ValueError('Malformed neural chunk')
                            journal.write(json.dumps({'seed':seed,'task':task.name,'query_index':index,
                                'input_sha256':digest,'GT_map_SHA256':expected_map,'native_input':actual,
                                'actions':action.tolist(),'wall_latency_s':time.monotonic()-start})+'\n');journal.flush()
                            queue.extend(action[:a.prefix].copy());index+=1;queries+=1;queried=digest
                        command=queue.popleft();before=_interaction_audit(env)
                        obs,_,done,info=env.step(command);after=_interaction_audit(env);controls+=1
                        if seed==seeds[0]:env.render('overview')
                        post=model_input(env,obs)['scene64']
                        trace.append({'step':int(obs.step),'action':command.tolist(),
                            'executed_canonical_action':info.get('executed_canonical_action'),
                            'before':before,'after':after,'pre_action_scene64':state['scene64'].tolist(),
                            'post_action_scene64':post.tolist(),
                            'query_sha256':queried,'success':bool(info['success']),
                            'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback')})
                        distance=float(np.linalg.norm(post[11:14]))
                        if distance<best_distance:
                            best_distance=distance;capture('closest_target',replace=True)
                        if post[28:30].any() and 'first_contact' not in events:
                            events['first_contact']=int(obs.step);capture('first_contact')
                        if after['correct_object_held'] and 'first_correct_hold' not in events:
                            events['first_correct_hold']=int(obs.step);capture('first_correct_hold')
                        if before['correct_object_held'] and not after['correct_object_held'] and 'drop' not in events:
                            events['drop']=int(obs.step);capture('drop')
                        if info.get('safety_violation'):capture('safety_abort')
                    capture('terminal');traces[task.name]=trace
                    labels=sorted(pictures,key=lambda k:picture_steps[k])
                    sheet=Image.new('RGB',(672,len(labels)*248),'white');draw=ImageDraw.Draw(sheet)
                    for row,label in enumerate(labels):
                        for col,pixels in enumerate(pictures[label]):
                            sheet.paste(Image.fromarray(pixels).resize((224,224)),(col*224,row*248+24))
                            draw.text((col*224+3,row*248+4),f't{picture_steps[label]} '+label+'/'+('front','wrist','GT')[col],fill='black')
                    sheet.save(folder/(task.name+'_ROLL_OUT_REVIEW.png'))
                    atomic_json(folder/'trajectory.json',traces)
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'events':events,
                        'trajectory':str(folder/'trajectory.json'),'review':str(folder/(task.name+'_ROLL_OUT_REVIEW.png'))})
                    progress('running_actual_development')
            finally:env.close()
        progress('completed_actual_development')
    except Exception as error:
        progress('integration_error_preserve_actual_trials_NOT_zero_SR',error=str(error));raise
    finally:journal.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=5930)
    p.add_argument('--prefix',type=int,default=5);main(p.parse_args())
