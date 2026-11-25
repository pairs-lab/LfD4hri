"""Fast non-stepping replica, with per-rollout lossless RGB/segmentation gates."""
import argparse
import hashlib
import os
import time
from pathlib import Path
import cv2
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import HDRecorder,apply_render_state,SETTINGS
from .common import make_env,reset_put,read


def accept_published_cache(output,capture):
    """A secondary may publish atomically while primary prepares goldens.

    Never overwrite it or mistake a ready cache for a broken render. A partial
    directory is STILL an error; only a fully closed matching capture qualifies.
    """
    marker=output/'render_complete.json'
    if not marker.exists():return False
    complete=read(marker);metadata=read(output/'renderer.json')
    if not (complete.get('all_writers_closed') and complete.get('all_golden_RGB_and_seg_checks_passed')
            and metadata.get('all_per_rollout_golden_exact')):
        raise ValueError('Published output lacks complete golden/writer proof')
    if (metadata['source_state_sha256']!=sha256_file(capture/'recorded_physics_states.json')
            or sha256_file(output/'initial_scene.json')!=sha256_file(capture/'initial_scene.json')):
        raise ValueError('Published output belongs to another physical capture')
    return True


def wait_for_active_prefetch(output,capture):
    """Do not allocate another EGL replica for an already active owned stage."""
    if output.parent.name!='_renders' or output.exists():return False
    root=output.parent.parent;staging=root/'_pre_renders'/output.name
    if not staging.exists():return False
    owner=None;child=None
    for status in root.glob('prefetch_v*/status.json'):
        value=read(status);candidate=value.get('child')
        if not candidate:continue
        try:argv=Path(f'/proc/{candidate}/cmdline').read_bytes().split(b'\0')
        except OSError:continue
        arguments=[a.decode() for a in argv if a]
        if ('VLA.qualitative_r58.render' in arguments and str(staging.resolve()) in arguments
                and str(capture.resolve()) in arguments):
            owner=value['pid'];child=candidate;break
    if child is None:return False
    print({'waiting_for_matching_active_prefetch_without_second_EGL':child,'output':str(output)},flush=True)
    deadline=time.monotonic()+900
    def live(pid):
        try:return bool(Path(f'/proc/{pid}/cmdline').read_bytes())
        except OSError:return False
    while time.monotonic()<deadline:
        if accept_published_cache(output,capture):return True
        if not live(owner) and not live(child):
            # An orphaned completed stage is recoverable; never replace any
            # primary directory, even an empty one created by another owner.
            if accept_published_cache(staging,capture):
                from .prefetch import publish_without_replace
                publish_without_replace(staging,output)
                return accept_published_cache(output,capture)
            return False
        time.sleep(2)
    raise TimeoutError('Matching prefetch did not finish; keep artifacts, no duplicate GPU allocation')


class GatedRecorder(HDRecorder):
    def __init__(self,*args,golden,**kwargs):
        super().__init__(*args,**kwargs)
        self.golden=golden;self.refs={(r['frame'],r['role']):r for r in read(golden/'probe.json')['rows']}
        self.role_index=0;self.matches=[]

    def frame(self,cfg,size):
        rgb,seg=super().frame(cfg,size)
        index=len(self.frames);role='handeye' if self.role_index==0 else f'wide_{self.role_index}'
        self.role_index=(self.role_index+1)%7;key=(index,role)
        if key in self.refs:
            expected=cv2.cvtColor(cv2.imread(str(self.golden/f'frame{index:05d}_{role}.png')),cv2.COLOR_BGR2RGB)
            sha=hashlib.sha256(np.asarray(seg,np.int64).tobytes()).hexdigest()
            if not np.array_equal(rgb,expected) or sha!=self.refs[key]['segmentation_sha256']:
                raise ValueError(f'Render binding RGB/seg differs from pinned golden: {key}')
            self.matches.append({'frame':index,'role':role,'RGB_exact':True,'segmentation_exact':True})
        return rgb,seg


def main(a):
    if wait_for_active_prefetch(a.output,a.capture):
        print({'completed_prefetch_reused_without_duplicate_EGL':str(a.output)},flush=True)
        return
    try:a.output.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        if accept_published_cache(a.output,a.capture):
            print({'matching_closed_secondary_cache_reused_without_overwrite':str(a.output)},flush=True)
            return
        raise
    if not p.isNumpyEnabled():raise ValueError('This entry point requires the separately gated NumPy render binding')
    initial=read(a.capture/'initial_scene.json');states=read(a.capture/'recorded_physics_states.json')
    env=make_env(initial['seed']);recorder=None;start=time.monotonic()
    try:
        recorder=GatedRecorder(env,a.output,golden=a.golden);recorder.open();reset_put(env,initial['seed'])
        errors={'position_m':0.,'quaternion_component':0.,'joint_rad':0.}
        for state in states:
            row=apply_render_state(state['geometry']);errors={k:max(errors[k],row[k]) for k in errors}
            recorder.capture(state['physics_tick'],state['control_index'],recorded=state)
        if len(recorder.matches)!=len(recorder.refs):raise ValueError('Not all per-rollout golden checks were visited')
        atomic_json(a.output/'renderer.json',{
            'actual_renderer':'PyBullet_EGL_hardware_OpenGL','source_capture':str(a.capture.resolve()),
            'source_state_sha256':sha256_file(a.capture/'recorded_physics_states.json'),
            'render_binding_sha256':sha256_file(p.__file__),'NumPy_transfer_enabled':True,
            'render_binding_differs_from_physics_binding_EXPLICITLY':True,
            'not_used_for_policy_observations_or_physical_execution':True,
            'source_physics_binding_sha256':'43bacc0f4d1bdae2e05698a98cb3b0fe03238127c7c606992f55593e6841e2af',
            'camera_settings_sha256':sha256_file(SETTINGS),'per_rollout_golden_RGB_and_segmentation_matches':recorder.matches,
            'all_per_rollout_golden_exact':True,'render_replica_max_errors':errors,
            'no_steps_between_restore_and_video_capture':True,'no_interpolation_or_extra_physics_ticks':True,
            'frames':len(states),'wall_seconds':time.monotonic()-start})
    finally:
        if recorder is not None:recorder.close()
        env.close()
    for name in ('initial_scene.json','content_evidence.json','replay_proof.json','protocol.json','outcome.json','release_protocol.json'):
        if (a.capture/name).exists():os.link(a.capture/name,a.output/name)
    atomic_json(a.output/'render_complete.json',{'all_writers_closed':True,'all_golden_RGB_and_seg_checks_passed':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','output','golden'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
