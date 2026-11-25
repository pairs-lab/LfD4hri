"""Eight actual neural motor controls, unchanged shared PyBullet plant/guards.

Two controls per task is an integration/safety check, NOT an episode SR test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.new_vlas.evaluate import png
from VLA.scene64_conditioning import model_input
from VLA.privileged_visual_r53.scene_map import render
from VLA.benchmark import _interaction_audit


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    session=requests.Session();session.trust_env=False
    url=f'http://127.0.0.1:{a.port}'
    meta=session.get(url+'/health',timeout=20).json()
    if meta['model']!='flower_r69' or meta['native_views']!=3 or meta['numeric_dimensions']!=67:
        raise ValueError('Wrong neural server or incomplete privileged input')
    protocol=json.loads(PROFILE_PROTOCOL.read_text())
    profile=json.loads(Path(protocol['observation_profile_file']).read_text())
    if profile_hash(profile)!=protocol['observation_profile_sha256']:
        raise ValueError('Shared frozen observation profile changed')
    seed=20266901
    tasks=[randomize_rl_scene(t,np.random.default_rng(seed+i)) for i,t in enumerate(select_tasks('all'))]
    env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
        control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
        success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=protocol['success']['semantic_thresholds'])
    records=[]
    try:
        for task in tasks:
            obs=env.reset(task)
            if env.is_semantic_success():
                raise ValueError('Initially successful smoke setup, no reroll')
            for step in range(2):
                state=model_input(env,obs)
                payload={'instruction':obs.prompt,'scene64':state['scene64'].tolist(),
                    'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                    'seed':seed,'inference_index':step}
                reply=session.post(url+'/act',json=payload,timeout=120)
                reply.raise_for_status();reply=reply.json()
                digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                if (reply['input_sha256']!=digest or not reply['actual_neural_model']
                    or reply['map_RGB_sha256']!=hashlib.sha256(render(state['scene64'],task.name).tobytes()).hexdigest()):
                    raise ValueError('Neural input/actual GT3 receipt mismatch')
                actions=np.asarray(reply['actions'],np.float32)
                if actions.shape!=(10,7) or not np.isfinite(actions).all():
                    raise ValueError('Native neural chunk malformed')
                before=_interaction_audit(env)
                obs,_,done,info=env.step(actions[0]);env.render('overview')
                records.append({'task':task.name,'step':step,'model_action':actions[0].tolist(),
                    'executed_canonical_action':info.get('executed_canonical_action'),
                    'before':before,'after':_interaction_audit(env),
                    'input_sha256':digest,'native':reply['actual_native_input_receipt'],
                    'safety_violation':info.get('safety_violation'),'done':bool(done)})
                atomic_json(a.output/'progress.json',{'actual_neural_controls':len(records),'records':records})
                if done or info.get('safety_violation'):
                    raise ValueError('Smoke ended early or unsafe; preserve trace, NOT a measured SR')
        atomic_json(a.output/'SMOKE8_NOT_SR.json',{'passed':len(records)==8,'actual_neural_controls':len(records),
            'tasks':[t.name for t in tasks],'metadata':meta,'records':records,
            'same_robot_gripper_physics_camera_safety_success_protocol':True,
            'new_Oracle_or_action_repair':False,'model_success_rate_established':False})
        print({'passed':True,'actual_neural_controls':len(records),'model_SR_measured':False},flush=True)
    except Exception as error:
        atomic_json(a.output/'FAILED_GATE_NOT_MODEL_SR.json',{'passed':False,'error':str(error),
            'actual_neural_controls':len(records),'records':records,'not_measured_SR':True})
        raise
    finally:
        env.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=5927)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
