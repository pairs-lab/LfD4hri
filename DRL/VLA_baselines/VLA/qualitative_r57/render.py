"""Seven matching HD streams from measured states, in a non-stepping replica."""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import (HDRecorder,apply_render_state,physical_state,canonical_hash,
    select_layouts,audit_videos,decode_at,SETTINGS)
from env_eval.camera_editions import composite
from env_eval.handeye_camera import handeye_geometry
from .common import read,make_env,reset_put


def render(capture):
    capture=capture.resolve()
    initial=read(capture/'initial_scene.json');seed=initial['seed']
    if (capture/'handeye.mp4').exists():raise FileExistsError('Keep completed/failed render attempts; use a new capture path')
    env=make_env(seed);recorder=None
    try:
        recorder=HDRecorder(env,capture);recorder.open()
        task,obs=reset_put(env,seed)
        if canonical_hash(physical_state(env))!=initial['measured_state_sha256']:
            raise ValueError('HD replica measured initial world differs')
        image,_=recorder.frame(handeye_geometry(env,recorder.settings['handeye']),recorder.settings['handeye_resolution'])
        if image.std()<1.:raise ValueError('Empty EGL image: refuse blank footage')
        cv2.imwrite(str(capture/'review/reset_handeye.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        maxima={'position_m':0.,'quaternion_component':0.,'joint_rad':0.}
        states=read(capture/'recorded_physics_states.json')
        for state in states:
            errors=apply_render_state(state['geometry'])
            maxima={k:max(maxima[k],errors[k]) for k in maxima}
            recorder.capture(state['physics_tick'],state['control_index'],recorded=state)
        atomic_json(capture/'renderer.json',{'actual_renderer':'PyBullet_EGL_hardware_OpenGL',
            'render_OS_pid':os.getpid(),'no_steps_during_recorded_state_rendering':True,
            'no_generated_or_interpolated_poses':True,'render_replica_max_errors':maxima,
            'replica_quaternion_normalization_tolerance_not_bit_exact':1e-12,
            'pybullet_binary_sha256':sha256_file(p.__file__),'camera_settings_sha256':sha256_file(SETTINGS),
            'native_observation_and_inference_not_modified':True})
    finally:
        if recorder is not None:recorder.close()
        env.close()


def main(a):
    render(a.capture)
    if not a.render_only:
        frames=read(a.capture/'frames.json');layouts=select_layouts(frames)
        atomic_json(a.capture/'layout.json',layouts)
        settings=read(SETTINGS)
        for view in range(1,7):composite(a.capture,a.capture,view,layouts[str(view)],settings)
        if (a.capture/'replay_proof.json').exists():proof=read(a.capture/'replay_proof.json')
        else:
            outcome=read(a.capture/'outcome.json')
            proof={'physics_ticks':outcome['physics_ticks'],'benchmark_region_success':False,
                'safety_violation':outcome['error'],'every_after_action_full_physical_state_and_action_exact':False}
        audit=audit_videos(a.capture,proof)
        if not audit['media_decode_sync_pass']:raise ValueError('Encoded media failed full-decode/sync gate')
        event=read(a.capture/'content_evidence.json') if (a.capture/'content_evidence.json').exists() else {'events':[]}
        review_indices=set()
        for e in event['events']:
            review_indices.update(range(e['before_frame'],min(len(frames),e['after_frame']+1),2))
        # Dense event review sheets are derivatives, NEVER added to footage.
        for index in sorted(review_indices):
            images=[]
            for name in ('view_1','view_2','view_3','handeye'):
                frame=decode_at(a.capture/(name+'.mp4'),index)
                images.append(cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA))
            sheet=np.concatenate((np.concatenate(images[:2],axis=1),np.concatenate(images[2:],axis=1)),axis=0)
            cv2.imwrite(str(a.capture/'review'/f'event_{index:05d}.jpg'),sheet)
        atomic_json(a.capture/'manifest.sha256.json',{f.name:sha256_file(f) for f in a.capture.glob('*.mp4')})
        atomic_json(a.capture/'media_status.json',{'status':'full_decode_sync_PASS_manual_event_review_required',
            'streams':7,'principal_method_rollouts':1,'dense_event_review_frames':sorted(review_indices),
            'qualitative_success_NOT_assumed_from_benchmark':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true')
    main(parser.parse_args())
