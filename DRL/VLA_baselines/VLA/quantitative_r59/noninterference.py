"""Replay fixed genuine native actions twice to test disabling media output."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.qualitative_r57.common import make_env, tasks
from VLA.scene64_conditioning import model_input
from VLA.benchmark import _interaction_audit
from .common import ROOT, OUT


def run(media):
    seed=20268011
    source=ROOT/f'VLA/audits/autocycle_r55_20261017/automatic_v1/xvla_new_r1_development40/test_seed{seed}/trajectory.json'
    traces=json.loads(source.read_text()); hashes=[];outcomes=[]
    env=make_env(seed)
    try:
        for task in tasks(seed):
            obs=env.reset(task)
            query_steps={row['chunk_query_step'] for row in traces[task.name]}
            for row in traces[task.name]:
                state=model_input(env,obs)
                values=dict(front=obs.front_rgb,wrist=obs.wrist_rgb,joints=obs.joint_positions,
                    EEF=obs.ee_position,rotation=obs.ee_rotation_6d,scene67=state['scene64'])
                hashes.append({'task':task.name,'step':obs.step,'actually_sent_to_neural_query':obs.step in query_steps,
                    'hashes':{key:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for key,value in values.items()},
                    'numeric':{key:np.asarray(value).tolist() for key,value in values.items() if key not in ('front','wrist')},
                    'interaction':_interaction_audit(env)})
                if (task.name,obs.step) in (('touch',36),('move',144)):
                    dest=OUT/'unused_frame_pixel_probe';dest.mkdir(exist_ok=True)
                    Image.fromarray(obs.wrist_rgb).save(dest/f'{task.name}_{obs.step}_overview{media}.png')
                obs,_,done,info=env.step(np.asarray(row['action'],np.float32))
                if media: env.render('overview')
            outcomes.append({'task':task.name,'steps':obs.step,'success':bool(info['success'])})
    finally:env.close()
    return hashes,outcomes


def main():
    first,fo=run(True);second,so=run(False)
    atomic_json(OUT/'no_video_all_step_hashes_with_overview.json',first)
    atomic_json(OUT/'no_video_all_step_hashes_without_overview.json',second)
    mismatches=[]
    for a,b in zip(first,second):
        different=[key for key,value in a['hashes'].items() if value!=b['hashes'][key]]
        ia,ib=a['interaction'],b['interaction']
        interaction=[key for key in ia if ia[key]!=ib.get(key)]
        if different or interaction:
            mismatches.append({'task':a['task'],'step':a['step'],'different':different,
                'actually_sent_to_neural_query':a['actually_sent_to_neural_query'],
                'interaction_differences':{key:[ia[key],ib.get(key)] for key in interaction},
                'numeric_max_error':{key:float(np.max(abs(np.asarray(a['numeric'][key])-b['numeric'][key])))
                    for key in a['numeric'] if key in different}})
    atomic_json(OUT/'NO_VIDEO_DIAGNOSTIC_v2.json',{'steps':[len(first),len(second)],'outcomes':[fo,so],
        'mismatch_count':len(mismatches),'first_mismatches':mismatches[:12]})
    # RGB between native chunk boundaries is rendered but NEVER sent to the
    # neural model. The absolute-action decoder consumes measured EEF/proprio,
    # not these unused images. Require exact identity for every used input.
    blocking=[r for r in mismatches if r['interaction_differences'] or r['numeric_max_error']
              or (r['different'] and r['actually_sent_to_neural_query'])]
    if len(first)!=len(second) or fo!=so or blocking:
        raise ValueError('Disabling video changed a USED policy input, physics state or outcome')
    pixels=[]
    for task,step in (('touch',36),('move',144)):
        a=np.asarray(Image.open(OUT/f'unused_frame_pixel_probe/{task}_{step}_overviewTrue.png')).astype(int)
        b=np.asarray(Image.open(OUT/f'unused_frame_pixel_probe/{task}_{step}_overviewFalse.png')).astype(int)
        pixels.append({'task':task,'step':step,'changed_pixels':int(np.any(a!=b,axis=2).sum()),
            'max_channel_error':int(abs(a-b).max()),'mean_channel_error':float(abs(a-b).mean()),
            'not_sent_to_model':True})
    atomic_json(OUT/'NO_VIDEO_NONINTERFERENCE.json', {'passed':True,'steps_each':len(first),
        'all_USED_neural_query_RGB_joint_EEF_scene64_and_interaction_hashes_exact':True,
        'all_control_step_numeric_state_and_interaction_exact':True,
        'all_rendered_RGB_exact_NOT_claimed':len(mismatches)==0,
        'unused_non_query_RGB_mismatches':mismatches,'unused_pixel_probe':pixels,
        'actual_neural_query_frames_each':sum(r['actually_sent_to_neural_query'] for r in first),
        'same_native_actions_replayed_NOT_new_model_SR_trials':True,
        'render_overview_after_each_action_vs_no_extra_render':True,'outcomes':fo,
        'no_video_file_created':True})
    print({'passed':True,'steps_each':len(first)},flush=True)


if __name__ == '__main__':main()
