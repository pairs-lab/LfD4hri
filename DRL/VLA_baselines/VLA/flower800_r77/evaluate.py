"""Explicit final800 admission; same R70 native-model/plant/action loop.

No developmental-role alias is used to bypass R70's held-out data guard.
No video or image files are saved. Full model-query and dynamics traces remain.
"""
import argparse
from collections import deque
from datetime import datetime,timezone
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
from VLA.common.provenance import sha256_file
from VLA.benchmark import _interaction_audit
from VLA.new_vlas.evaluate import png
from VLA.scene64_conditioning import model_input
from VLA.privileged_visual_r53.scene_map import render
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.cross_vla800_r62.common import SEEDS,TASKS
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.flower_r70.common import read,freeze_test_code,verify


def binding():
    result=freeze_test_code()
    result.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    return result


def validate_cohort(cohort,control=False):
    expected=[20268011] if control else SEEDS
    role='implementation_parity_control4' if control else 'final_fixed_evaluation800'
    if cohort.get('role')!=role or cohort.get('seeds')!=expected:
        raise ValueError('Require the exact locked cohort and explicit authorized role')
    return expected


def main(a):
    cohort=read(a.cohort);seeds=validate_cohort(cohort,a.control)
    method=read(a.method)
    if method['evaluation_entry_point']!='VLA.flower_r70.evaluate' or method['prefix']!=10:
        raise ValueError('Only the actually selected and tested static10 method is admitted')
    bound=binding();bound[str(a.method.resolve())]=sha256_file(a.method)
    session=requests.Session();session.trust_env=False;url=f'http://127.0.0.1:{a.port}'
    response=session.get(url+'/health',timeout=15);response.raise_for_status();meta=response.json()
    if (meta['checkpoint']!=method['checkpoint'] or meta['checkpoint_sha256']!=method['trainables_sha256'] or
        meta['model']!='flower_r69' or meta['numeric_dimensions']!=67 or meta['native_views']!=3 or
        not meta['all_saved_FP32_masters_exact']):raise ValueError('Frozen method/native reload differs')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    if profile_hash(profile)!=protocol['observation_profile_sha256']:raise ValueError('Shared profile changed')
    atomic_json(a.output/'PLAN.json',{'model':meta,'cohort':cohort,'prefix':10,'same_plant_as_R62_XVLA':True,
        'same_success':protocol['success'],'PUT_requires_release':False,'assist_grasp':True,
        'low_level_controller':'unchanged common delta-EEF to joint IK; no Oracle/planner policy',
        'video_saved':False,'images_saved':False,'code':bound,'frozen_method':method,
        'one_evaluation_cohort_NOT_five_training_seeds':not a.control,'implementation_control_only':a.control})
    rows=[];queries=controls=0;started=time.monotonic()
    journal=(a.output/'NEURAL_QUERY_RECEIPTS.jsonl').open('x')
    def progress(state,**extra):
        per={t:{'successes':sum(r['success'] for r in rows if r['task']==t),
                'n':sum(r['task']==t for r in rows)} for t in TASKS}
        for v in per.values():v['SR_percent']=None if not v['n'] else 100*v['successes']/v['n']
        atomic_json(a.output/'progress.json',{'status':state,'completed_trials':len(rows),'planned_trials':len(seeds)*4,
            'per_action':per,'rows':rows,'native_queries':queries,'motor_controls':controls,
            'checkpoint':meta['checkpoint'],'elapsed_wall_s':time.monotonic()-started,
            'updated_UTC':datetime.now(timezone.utc).isoformat(),
            'final_fixed_evaluation800':not a.control,'implementation_control_only':a.control,**extra})
    try:
        progress('running_actual_control4' if a.control else 'running_actual_final800')
        for seed in seeds:
            verify(bound);rng=np.random.default_rng(seed)
            tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
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
                    queue=deque();trace=[];index=0;done=False;events={};queried=None
                    while not done:
                        state=model_input(env,obs)
                        if not queue:
                            payload={'instruction':obs.prompt,'scene64':state['scene64'].tolist(),
                                'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                                'seed':seed,'inference_index':index}
                            start=time.monotonic();r=session.post(url+'/act',json=payload,timeout=240)
                            r.raise_for_status();reply=r.json()
                            digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            expected=hashlib.sha256(render(state['scene64'],task.name).tobytes()).hexdigest()
                            actual=reply['actual_native_input_receipt']
                            if (not reply['actual_neural_model'] or reply['input_sha256']!=digest or
                                reply['map_RGB_sha256']!=expected or actual['numeric']!=[1,67] or
                                actual['images']!=[1,3,3,224,224] or actual['frequency_Hz']!=10):
                                raise ValueError('Actual native inputs/neural receipt mismatch')
                            action=np.asarray(reply['actions'],np.float32)
                            if action.shape!=(10,7) or not np.isfinite(action).all():raise ValueError('Malformed neural chunk')
                            journal.write(json.dumps({'seed':seed,'task':task.name,'query_index':index,
                                'input_sha256':digest,'GT_map_SHA256':expected,'native_input':actual,
                                'actions':action.tolist(),'wall_latency_s':time.monotonic()-start})+'\n');journal.flush()
                            queue.extend(action.copy());index+=1;queries+=1;queried=digest
                        command=queue.popleft();before=_interaction_audit(env)
                        obs,_,done,info=env.step(command);after=_interaction_audit(env);controls+=1
                        if seed==seeds[0]:env.render('overview')
                        post=model_input(env,obs)['scene64']
                        trace.append({'step':int(obs.step),'action':command.tolist(),
                            'executed_canonical_action':info.get('executed_canonical_action'),
                            'before':before,'after':after,'pre_action_scene64':state['scene64'].tolist(),
                            'post_action_scene64':post.tolist(),'query_sha256':queried,'success':bool(info['success']),
                            'safety_violation':info.get('safety_violation'),
                            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback')})
                        if post[28:30].any() and 'first_contact' not in events:events['first_contact']=int(obs.step)
                        if after['correct_object_held'] and 'first_correct_hold' not in events:events['first_correct_hold']=int(obs.step)
                        if before['correct_object_held'] and not after['correct_object_held'] and 'drop' not in events:events['drop']=int(obs.step)
                    traces[task.name]=trace;atomic_json(folder/'trajectory.json',traces)
                    rows.append({'seed':seed,'task':task.name,'success':bool(info['success']),'steps':len(trace),
                        'safety_violation':info.get('safety_violation'),'events':events,'trajectory':str(folder/'trajectory.json')})
                    progress('running_actual_control4' if a.control else 'running_actual_final800')
            finally:env.close()
        progress('completed_actual_control4' if a.control else 'completed_actual_final800')
    except Exception as error:
        progress('integration_error_preserve_actual_trials_NOT_zero_SR',error=str(error));raise
    finally:journal.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--method',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=5980);p.add_argument('--control',action='store_true');main(p.parse_args())
