"""Two synchronized HD streams from verified saved geometry, never a rollout.

First run --golden with the original pinned PyBullet binding. The NumPy-enabled
render process then must match lossless RGB and segmentation for all goldens.
The only streams produced are wide_2 and presentation handeye. No fake six-view
coverage is claimed. All source captures/checkpoints remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pybullet as p

from VLA.common.artifact_io import atomic_json as write
from VLA.common.provenance import sha256_file as digest
from VLA.qualitative_r57.common import make_env,reset_put
from VLA.qualitative_recorder_r51 import apply_render_state,camera_stats
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.camera_editions import writer,projected_box
from env_eval.handeye_camera import handeye_geometry
from utils.camera import CINEMATIC_VIEWS

PINNED='43bacc0f4d1bdae2e05698a98cb3b0fe03238127c7c606992f55593e6841e2af'
OUTPUT=Path('/home/uet/Downloads/qualitative_comparison_redesign')


def read(path):
    return json.loads(Path(path).read_text())


def main(args):
    root=args.output/'additional_recovery_source'
    capture=root/(args.method+'_capture')
    # Version 1 was rejected by visual review: loading EGL after reset made
    # matching but empty images. Preserve that diagnostic, never reuse it.
    gold=root/(args.method+'_golden_v2');dest=gold if args.golden else root/(args.method+'_render_v2')
    if (dest/('probe.json' if args.golden else 'render_complete.json')).exists():return
    dest.mkdir(parents=True,exist_ok=False)
    binding=digest(p.__file__)
    if args.golden and binding!=PINNED:raise ValueError('Goldens must use actual original binding')
    states=read(capture/'recorded_physics_states.json');initial=read(capture/'initial_scene.json')
    settings=read(Path('env_eval/camera_editions.json'))
    event=next((e for e in read(capture/'content_evidence.json')['events'] if e['mode']=='verified_in_transit_drop'),None)
    middle=event['event_frame'] if event else len(states)//2
    indices=(0,middle,len(states)-1)
    reference={} if args.golden else {(r['frame'],r['role']):r for r in read(gold/'probe.json')['rows']}
    env=make_env(initial['seed']);plugin=-1;streams={};records=[];matches=[];goldrows=[]
    try:
        module=importlib.util.find_spec('eglRenderer')
        plugin=p.loadPlugin(module.origin,'_eglRendererPlugin',physicsClientId=env.client_id)
        if plugin<0:raise ValueError('EGL unavailable; do not substitute renderer')
        # Visual shapes must be registered AFTER the hardware renderer exists.
        reset_put(env,initial['seed'])
        context=SimpleNamespace(renderer=p.ER_BULLET_HARDWARE_OPENGL)
        if not args.golden:
            for role,size in [('wide_2',settings['wide_resolution']),('handeye',settings['handeye_resolution'])]:
                streams[role]=writer(dest/(role+'.mp4'),size,30)
        for i,state in enumerate(states):
            if args.golden and i not in indices:continue
            errors=apply_render_state(state['geometry'])
            if errors['position_m']>1e-12 or errors['joint_rad']>1e-12 or errors['quaternion_component']>1e-12:
                raise ValueError('Saved geometry was not restored exactly')
            row={k:state[k] for k in ('physics_tick','simulation_time_s','control_index','correct_attached',
                                     'target_contact_force_N','bilateral','object_position','pad_center')}
            row.update(frame=i,encoded_pts_s=i/30,wide=[{} for _ in range(6)])
            cfgs=[('wide_2',CINEMATIC_VIEWS[1],settings['wide_resolution']),
                  ('handeye',handeye_geometry(env,settings['handeye']),settings['handeye_resolution'])]
            for role,cfg,size in cfgs:
                rgb,seg=CinematicRollout.frame(context,cfg,*size)
                if float(np.std(rgb.astype(float),axis=(0,1)).max())<1.:
                    raise ValueError('Empty/constant presentation image: reject render')
                if role=='wide_2' and not np.any(np.asarray(seg)>=0):
                    raise ValueError('Wide camera sees no scene geometry')
                maskhash=hashlib.sha256(np.asarray(seg,np.int64).tobytes()).hexdigest()
                if args.golden:
                    path=dest/f'frame{i:05d}_{role}.png';Image.fromarray(rgb).save(path)
                    goldrows.append({'frame':i,'role':role,'segmentation_sha256':maskhash,'image_sha256':digest(str(path))})
                else:
                    if (i,role) in reference:
                        expected=np.asarray(Image.open(gold/f'frame{i:05d}_{role}.png'))
                        if not np.array_equal(rgb,expected) or maskhash!=reference[i,role]['segmentation_sha256']:
                            raise ValueError('Alternative render binding differs from actual pinned golden')
                        matches.append({'frame':i,'role':role,'RGB_exact':True,'segmentation_exact':True})
                    streams[role].send(rgb)
                stats=camera_stats(seg,env)
                if role=='wide_2':
                    row['wide'][1]={**stats,
                        'projected_pick_box':projected_box(env,env.pick_id,cfg,*size),
                        'projected_place_box':projected_box(env,env.place_id,cfg,*size) if env.place_id is not None else None}
                else:row['handeye']={**stats,'camera':cfg}
            if not args.golden:records.append(row)
            if i%60==0:print({'method':args.method,'golden':args.golden,'frame':i,'total':len(states)},flush=True)
        if args.golden:
            write(dest/'probe.json',{'binding_sha256':binding,'rows':goldrows,'pinned_binding':True})
        else:
            if len(matches)!=len(reference):raise ValueError('Not every golden was visited')
            write(dest/'frames.json',records)
            write(dest/'renderer.json',{'actual_renderer':'PyBullet_EGL_hardware_OpenGL',
                'source_capture':str(capture),'source_state_sha256':digest(str(capture/'recorded_physics_states.json')),
                'source_physics_binding_sha256':PINNED,'render_binding_sha256':binding,
                'all_per_rollout_golden_exact':True,'per_rollout_golden_matches':matches,
                'EGL_loaded_before_reset':True,'constant_RGB_rejected_each_frame':True,
                'no_steps_during_recorded_state_rendering':True,'no_interpolation':True,
                'actual_streams':['wide_2','handeye'],'frames':len(records)})
            for name in ('initial_scene.json','content_evidence.json','replay_proof.json','outcome.json'):
                if (capture/name).exists():shutil.copy2(capture/name,dest/name)
    finally:
        for stream in streams.values():stream.close()
        if plugin>=0:p.unloadPlugin(plugin,physicsClientId=env.client_id)
        env.close()
    if not args.golden:write(dest/'render_complete.json',{'closed':True,'goldens_passed':True,'frames':len(records)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--method',choices=('XVLA','Oracle'),required=True)
    parser.add_argument('--golden',action='store_true')
    main(parser.parse_args())
