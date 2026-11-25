"""Check alternate binding only in a NON-STEPPING image replica against gold.

The physics/observations/neural-action replay always uses the original pinned
SIM binary. This probe permits a NumPy-enabled rendering binding only after
actual restored geometry, camera extrinsics and lossless RGB match checks.
"""
import argparse
import hashlib
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import apply_render_state
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.handeye_camera import handeye_geometry
from utils.camera import CINEMATIC_VIEWS
from .common import make_env,reset_put,read


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    states=read(a.capture/'recorded_physics_states.json');initial=read(a.capture/'initial_scene.json')
    settings=read(Path('env_eval/camera_editions.json'));env=make_env(initial['seed']);plugin=-1;rows=[]
    try:
        module=importlib.util.find_spec('eglRenderer');plugin=p.loadPlugin(module.origin,'_eglRendererPlugin',physicsClientId=env.client_id)
        if plugin<0:raise ValueError('No EGL')
        reset_put(env,initial['seed']);ctx=SimpleNamespace(renderer=p.ER_BULLET_HARDWARE_OPENGL)
        for index in a.frames:
            errors=apply_render_state(states[index]['geometry'])
            for role,cfg,size in [('handeye',handeye_geometry(env,settings['handeye']),settings['handeye_resolution']),
                    *[(f'wide_{i}',cfg,settings['wide_resolution']) for i,cfg in enumerate(CINEMATIC_VIEWS,1)]]:
                start=time.monotonic();rgb,seg=CinematicRollout.frame(ctx,cfg,*size);elapsed=time.monotonic()-start
                name=f'frame{index:05d}_{role}.png';cv2.imwrite(str(a.output/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                row={'frame':index,'role':role,'render_seconds':elapsed,'geometry_errors':errors,'camera':cfg,
                    'segmentation_sha256':hashlib.sha256(np.asarray(seg,np.int64).tobytes()).hexdigest()}
                if a.reference:
                    golden=cv2.imread(str(a.reference/name));actual=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    if golden is None:raise ValueError('Missing golden RGB')
                    diff=np.abs(golden.astype(np.int16)-actual.astype(np.int16))
                    row.update(RGB_exact=bool(np.array_equal(golden,actual)),RGB_MAE=float(diff.mean()),
                        RGB_max=int(diff.max()),pixels_with_any_channel_difference=int(np.any(diff,axis=2).sum()))
                rows.append(row)
        result={'binding':p.__file__,'binding_sha256':sha256_file(p.__file__),'numpy_enabled':bool(p.isNumpyEnabled()),
            'physics_policy_replay_unchanged_separate_process':True,'no_steps_between_restore_and_image':True,
            'rows':rows,'all_RGB_exact':all(r.get('RGB_exact',False) for r in rows)}
        atomic_json(a.output/'probe.json',result);print(result,flush=True)
    finally:
        if plugin>=0:p.unloadPlugin(plugin,physicsClientId=env.client_id)
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[0,270,599]);parser.add_argument('--reference',type=Path)
    main(parser.parse_args())
