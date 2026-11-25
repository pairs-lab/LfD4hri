"""Quick actual-state event review; not a substitute for full-stream audit."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.qualitative_recorder_r51 import apply_render_state,physical_state,canonical_hash
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.handeye_camera import handeye_geometry
from utils.camera import CINEMATIC_VIEWS
from .common import make_env,reset_put,read


def main(a):
    initial=read(a.capture/'initial_scene.json');states=read(a.capture/'recorded_physics_states.json')
    env=make_env(initial['seed']);plugin=-1
    review=a.capture/'event_previews';review.mkdir(exist_ok=True)
    try:
        module=importlib.util.find_spec('eglRenderer')
        if module is None:raise RuntimeError('No EGL renderer')
        plugin=p.loadPlugin(module.origin,'_eglRendererPlugin',physicsClientId=env.client_id)
        if plugin<0:raise RuntimeError('EGL load failed')
        reset_put(env,initial['seed'])
        if canonical_hash(physical_state(env))!=initial['measured_state_sha256']:raise ValueError('Preview replica initial state differs')
        settings=read(Path('env_eval/camera_editions.json'))
        context=SimpleNamespace(renderer=p.ER_BULLET_HARDWARE_OPENGL)
        files=[]
        for index in a.frames:
            state=states[index];errors=apply_render_state(state['geometry']);panels=[]
            for cfg in [*CINEMATIC_VIEWS[:3],handeye_geometry(env,settings['handeye'])]:
                rgb,_=CinematicRollout.frame(context,cfg,1280,720)
                panels.append(cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(640,360),interpolation=cv2.INTER_AREA))
            sheet=np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
            path=review/f'frame_{index:05d}.jpg';cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,96])
            files.append({'file':str(path),'source_frame':index,'physics_tick':state['physics_tick'],
                'simulation_time_s':state['physics_tick']/240.,'geometry_restore_errors':errors})
        atomic_json(review/'manifest.json',{'actual_recorded_states_NOT_animation':True,
            'views':['front','side','workspace','presentation_handeye'],'review_only_NOT_principal_videos':True,
            'files':files})
    finally:
        if plugin>=0:p.unloadPlugin(plugin,physicsClientId=env.client_id)
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',required=True)
    main(parser.parse_args())
