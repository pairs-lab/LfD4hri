"""CPU plant/codec replay of real completed old-S traces, NOT new policy SR."""
import argparse
from pathlib import Path
import numpy as np

from .common import ROOT,OUT,read,atomic_json,TASKS


def main(a):
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.tasks import select_tasks,randomize_rl_scene
    from VLA.scene64_conditioning import model_input
    from VLA.new_vlas.evaluate import png
    from VLA.privileged_native_server import decode_image
    source=ROOT/'VLA/audits/continuous_r86_20261021/vlanext_s/dev40_fit8192'
    plan=read(source/'PLAN.json');protocol=read(plan['protocol'])
    profile=read(protocol['observation_profile_file'])
    seed=20283001;rng=np.random.default_rng(seed)
    tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
    recorded=read(source/f'seed{seed}'/'trajectory.json')
    env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',
        image_size=384,control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,
        use_semantic_success=True,success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=protocol['success']['semantic_thresholds'],render_observations=False)
    summaries=[]
    try:
        for task in tasks:
            obs=env.reset(task);trace=recorded[task.name]
            for camera in ('front','wrist'):
                rgb=env.render(camera)
                if rgb.shape!=(384,384,3) or not np.array_equal(rgb,decode_image(png(rgb))):
                    raise ValueError('Actual camera/PNG native wire roundtrip mismatch')
            for i,entry in enumerate(trace):
                if not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['pre_action_scene64'],np.float32)):
                    raise ValueError(f'CPU replay pre-state mismatch {task.name}/{i}')
                obs,_,done,info=env.step(np.asarray(entry['action'],np.float32))
                if (not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['post_action_scene64'],np.float32))
                    or bool(info['success'])!=entry['success'] or info.get('safety_violation')!=entry['safety_violation']):
                    raise ValueError(f'CPU replay post-state/outcome mismatch {task.name}/{i}')
                if done!=(i==len(trace)-1):raise ValueError('CPU replay termination timing differs')
            summaries.append({'task':task.name,'recorded_controls_verified':len(trace),
                'all64_pre_post_elements_equal_without_tolerance':True,
                'original_stop_outcome_and_safety_equal':True,'actual_RGB_codec_roundtrips':2})
    finally:env.close()
    if {s['task'] for s in summaries}!=set(TASKS):raise ValueError('Incomplete four-task replay')
    atomic_json(a.output,{'passed':True,'source':str(source),'seed':seed,'tasks':summaries,
        'new_neural_trials':0,'no_policy_or_Oracle_queried':True,
        'purpose':'exact plant/reset-history/codec validation, NOT model readiness or SR'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
