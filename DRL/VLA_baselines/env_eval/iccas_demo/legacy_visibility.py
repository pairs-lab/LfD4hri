"""Backfill missing per-object segmentation on the two original native pilots.

This is a real measured frame-zero replay, checked against encoded wide pixels;
it does not edit videos, physical trajectories or task outcomes.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from VLA.common.artifact_io import atomic_json
from env_eval.oracle_cinematic import CinematicRollout
from types import SimpleNamespace
from utils.camera import CINEMATIC_VIEWS
from .assets import ICCASEnvironment
from .qualify import OUT,ROBOTS
from .render import apply_frame,VIEWS
from .audit import decode_frame


def main():
    for robot,label in ROBOTS.items():
        folder=OUT/label/'scene01';selection=json.loads((folder/'PHYSICS.json').read_text())
        source=Path(selection['qualified_directory']);schema=json.loads((source/'REPLAY_SCHEMA.json').read_text())
        arrays=np.load(source/'frame_states.npz');layout=json.loads((folder/'layout.json').read_text())
        cache=OUT/'_camera_sources'/label/'scene01';vis=json.loads((cache/'visibility.json').read_text())
        with ICCASEnvironment(robot,OUT/'_derived_assets'/label,True) as env:
            from .scenes import get_scene
            env.reset(get_scene(1).task());apply_frame(env,schema,arrays,0)
            camera=SimpleNamespace(renderer=env.demo_renderer);errors={};counts={}
            for view in VIEWS:
                rgb,seg=CinematicRollout.frame(camera,CINEMATIC_VIEWS[view-1],1920,1080)
                encoded=cv2.cvtColor(decode_frame(folder/f'view_{view}.mp4',0),cv2.COLOR_BGR2RGB)
                _,x,y,w,h=layout[str(view)]['rect'];mask=np.ones((1080,1920),bool)
                mask[y-4:y+h+4,x-4:x+w+4]=False
                error=float(np.abs(rgb.astype(np.float32)-encoded.astype(np.float32))[mask].mean());errors[str(view)]=error
                if error>=3:raise ValueError('Legacy pilot wide frame not reproduced exactly enough for segmentation backfill')
                ids=seg&((1<<24)-1);row={name:int((ids==body).sum()) for name,body in env.object_ids.items()}
                counts[str(view)]=row;vis[0]['wide'][str(view)]['per_object_pixels']=row
            total={name:sum(c[name] for c in counts.values()) for name in env.object_ids}
            if not all(n>=20 for n in total.values()):raise ValueError('A pilot object is not actually visible')
        proof={'passed':True,'measured_frame':0,'raw_per_view_object_pixels':counts,
            'per_object_pixels_across_four_views':total,'encoded_wide_mean_pixel_errors':errors,
            'videos_trajectories_and_outcomes_not_modified':True}
        atomic_json(folder/'INITIAL_OBJECT_VISIBILITY_AUDIT.json',proof)
        vis[0]['per_object_pixel_provenance']='exact_frame0_state_replay_checked_against_all_four_encoded_wide_views'
        atomic_json(cache/'visibility.json',vis)
        audit_path=folder/'AUDIT.json'
        if audit_path.exists():
            check=json.loads(audit_path.read_text());check['initial_per_object_pixels_across_four_views']=total
            check['all_requested_objects_rendered']=True;check['initial_pixel_measurement_proof']=str(folder/'INITIAL_OBJECT_VISIBILITY_AUDIT.json')
            atomic_json(audit_path,check)
        print({'robot':robot,**proof},flush=True)


if __name__=='__main__':main()
