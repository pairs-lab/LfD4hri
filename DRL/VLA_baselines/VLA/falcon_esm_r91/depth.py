"""Admit old synchronized depth and replay new recovery labels without teaching."""
import argparse
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.observation_profile import camera_geometry
from VLA.recovery_quality_r90.data import Data as RecoveryData
from VLA.recovery_quality_r90.audit import OUT as R90
from VLA.dataset_replay_context import audit_environment,reset_source_episode
from VLA.simstate_scene64 import scene_state
from VLA.falcon_depth_pilot_r72 import metric_depth
from .common import OUT, read
from .data import DEPTH, geometry


def capture(env,obs):
    cfg = env.observation_profile['cameras']['front']
    eye,target,up = camera_geometry(env,cfg,'front')
    view = p.computeViewMatrix(eye.tolist(),target.tolist(),up.tolist())
    projection = p.computeProjectionMatrixFOV(float(cfg['fov_degrees']),1.,.01,3.)
    result = p.getCameraImage(384,384,viewMatrix=view,projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=env.client_id)
    rgb = np.asarray(result[2],np.uint8).reshape(384,384,4)[...,:3]
    if not np.array_equal(rgb,obs.front_rgb):
        raise ValueError('Depth render is not at the exact native RGB camera/state')
    depth = metric_depth(np.asarray(result[3],np.float32).reshape(384,384))
    K,cv = geometry(view,projection)
    if not np.isfinite(depth).all() or depth.min()<=0:
        raise ValueError('Invalid actual metric depth')
    return depth,np.asarray(view),np.asarray(projection),K,cv


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'DEPTH_ADMISSION.json').exists(): raise FileExistsError('Already admitted; do not recollect blindly')
    data = RecoveryData()
    original = read(DEPTH/'AUDIT.json')
    if not original['passed_pairing_replay'] or original['episodes']!=2185 or original['frames']!=119495:
        raise ValueError('Original complete TRAIN paired replay did not pass')
    old_rows = {r['qualified_row']:r for r in original['rows']}
    rows = []
    for ei in range(2185):
        row=old_rows[ei];source=data.rows[ei]
        if row['source_sha256']!=source['source_sha256'] or row['frames']!=source['frames']:
            raise ValueError('RGB-D source/count mismatch')
        if sha256_file(row['file'])!=row['sha256']:
            raise ValueError('Original depth payload hash changed')
        with np.load(row['file'],allow_pickle=False) as values:
            if values['front_depth_z_m'].shape!=(source['frames'],384,384):
                raise ValueError('Original full depth shape mismatch')
            np.testing.assert_array_equal(values['source_frame_indices'],np.arange(source['frames']))
        rows.append({'file':row['file'],'sha256':row['sha256'],'frames':row['frames'],
            'source_sha256':source['source_sha256'],'actual_pre_action_pair_replay_passed':True})
        if ei%100==0: atomic_json(OUT/'DEPTH_STATUS.json',{'stage':'admitting_original_TRAIN_pairs','rows':ei+1})
    candidate=R90/'near_miss_sweep'
    plan=read(candidate/'plan.json');start_source=Path(plan['source'])
    originals={r['file']:r for r in read(start_source/'index.json')}
    corrections={str(candidate/r['file']):r for r in read(candidate/'index.json') if r.get('candidate_replay_pass')}
    folder=OUT/'recovery_depth';folder.mkdir(exist_ok=True)
    for ei in range(2185,len(data.rows)):
        source=data.rows[ei];meta=corrections[source['source']]
        dest=folder/f'row{ei:04d}.npz'
        # No new expert action generation: exact replay of the admitted label
        # suffix, with the recorded intervention excluded from training labels.
        env=audit_environment(start_source,robot='ur5e',image_size=384,control_hz=10,
            seed=0,required_success_steps=10000,use_semantic_success=False)
        depths=[];views=[];projections=[]
        try:
            obs=reset_source_episode(env,start_source,originals[meta['source_file']])
            for action in meta['perturbation_actions_NOT_labels']:
                obs,_,_,info=env.step(np.asarray(action['action'],np.float32))
                if info.get('safety_violation'):raise ValueError('Recovery startup safety differs')
            env.step_count=env.success_streak=0;env._last_success=False
            obs=env.observe()
            with np.load(source['source'],allow_pickle=False) as value:
                reference_front=value['front_rgb'];reference_wrist=value['wrist_rgb']
                raw=value['audit_pre_action_PRIVILEGED_state64'];actions=value['canonical_actions']
            for fi,action in enumerate(actions):
                np.testing.assert_allclose(scene_state(env,obs),raw[fi],rtol=0,atol=1e-6)
                # A bit-exact new same-machine replay is expected. Never accept
                # wrong images just because the numeric target pose matches.
                np.testing.assert_array_equal(obs.front_rgb,reference_front[fi])
                np.testing.assert_array_equal(obs.wrist_rgb,reference_wrist[fi])
                depth,view,projection,_,_=capture(env,obs)
                depths.append(depth);views.append(view);projections.append(projection)
                obs,_,_,info=env.step(action)
                if info.get('safety_violation'):raise ValueError('Admitted recovery label replay unsafe')
        finally:env.close()
        if dest.exists(): raise FileExistsError('Partial candidate retained; inspect instead of overwrite')
        np.savez_compressed(dest,front_depth_z_m=np.stack(depths),view_world_to_OpenGL_camera=np.asarray(views),
            projection_OpenGL=np.asarray(projections),source_frame_indices=np.arange(len(depths)))
        rows.append({'file':str(dest),'sha256':sha256_file(dest),'frames':len(depths),
            'source_sha256':source['source_sha256'],'actual_pre_action_pair_replay_passed':True})
        atomic_json(OUT/'DEPTH_STATUS.json',{'stage':'replaying_admitted_recovery_TRAIN','rows':len(rows),
            'frames':sum(r['frames'] for r in rows),'not_neural_SR':True})
    atomic_json(OUT/'DEPTH_INDEX.json',rows)
    atomic_json(OUT/'DEPTH_ADMISSION.json',{'passed':True,'split':'train','episodes':len(rows),
        'frames':sum(r['frames'] for r in rows),'depth_index_sha256':sha256_file(OUT/'DEPTH_INDEX.json'),
        'raw_RGB_not_duplicated':True,'no_final_or_CONF_data':True,'new_expert_labels_generated':False,
        'all_pre_action_RGB_depth_and_state_pairing_replayed':True})


if __name__=='__main__': main()
