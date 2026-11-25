"""Real PyBullet A/B: adding the recorder must preserve inputs, actions, state."""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.scene64_conditioning import model_input
from VLA.recovery_r85.common import read
from .evaluate import observed_class


def main(a):
    protocol=read(PROFILE_PROTOCOL);profile=read(protocol['observation_profile_file'])
    seed=20283001
    task=randomize_rl_scene(next(t for t in select_tasks('all') if t.name=='pick'),np.random.default_rng(seed))
    base=environment_class_for_profile(profile)
    recorded=observed_class(base,a.output,{seed},lambda e,o:model_input(e,o)['scene64'])
    histories=[]
    for cls in (base,recorded):
        env=cls(observation_profile=profile,robot='ur5e',image_size=384,control_hz=10,
            assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
            success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
            semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        history=[]
        def capture(obs,info=None):
            history.append({'scene64':model_input(env,obs)['scene64'].tolist(),
                'EE_position':obs.ee_position.tolist(),'rotation6D':obs.ee_rotation_6d.tolist(),
                'RGB_sha256':[hashlib.sha256(v.tobytes()).hexdigest() for v in (obs.front_rgb,obs.wrist_rgb)],
                'success':bool(info.get('success')) if info else False,
                'executed_action':info['executed_canonical_action'] if info else None})
        try:
            obs=env.reset(copy.deepcopy(task));capture(obs)
            for i in range(6):
                command=np.array([.002,.001,-.004,0.,0.,.001,-1.],np.float32)
                obs,_,done,info=env.step(command);capture(obs,info)
                if done:raise ValueError('Unexpected termination during short recorder qualification')
            if cls is recorded:
                env._review_capture(obs,['end_of_AB_probe_NOT_task_success'])
                env._review_save()
        finally:env.close()
        histories.append(history)
    pairs=list(zip(*histories,strict=True))
    errors={key:float(max(np.max(abs(np.asarray(x[key])-np.asarray(y[key]))) for x,y in pairs))
            for key in ('scene64','EE_position','rotation6D')}
    equal_RGB=all(x['RGB_sha256']==y['RGB_sha256'] for x,y in pairs)
    equal_actions=all(x['executed_action']==y['executed_action'] for x,y in pairs)
    equal_outcomes=all(x['success']==y['success'] for x,y in pairs)
    passed=all(v<=1e-7 for v in errors.values()) and equal_RGB and equal_actions and equal_outcomes
    atomic_json(a.output/'AB_RECORDER_CHECK.json',{'passed':passed,'actual_PyBullet_runs':2,
        'motor_controls_per_run':6,'state_max_errors':errors,'all_actual_policy_RGB_byte_identical':equal_RGB,
        'executed_actions_equal':equal_actions,'outcome_flags_equal':equal_outcomes,
        'histories':histories,'fixed_neutral_actions_only_for_non_mutation_test_NOT_VLA_SR':True})
    if not passed:raise ValueError('Recorder changed observation/state: inspect A/B receipt before use')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
