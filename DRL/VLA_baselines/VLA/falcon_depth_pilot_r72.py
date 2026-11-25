"""Replay4 TRAIN demonstrations and audit synchronized metric depth/PCD.

Not neural evaluation or admitted FALCON training data. No data is fabricated
from object bounding-box meshes; depth comes from the actual policy camera.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.observation_profile import camera_geometry
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.family_transfer_r54.common import DATA
from VLA.evaluate_smolvla_v9_source import reset_observation_errors, validate_reset_observation
from VLA.simstate_scene64 import scene_state


def metric_depth(buffer, near=.01, far=3.):
    return far*near/(far-(far-near)*buffer)


def cloud(buffer, view, projection):
    h,w=buffer.shape; yy,xx=np.indices((h,w))
    clip=np.stack((2*(xx+.5)/w-1,1-2*(yy+.5)/h,2*buffer-1,np.ones_like(buffer)),axis=-1)
    v=np.asarray(view).reshape(4,4).T; proj=np.asarray(projection).reshape(4,4).T
    world=clip.reshape(-1,4)@np.linalg.inv(proj@v).T
    world=world[:,:3]/world[:,3,None]
    return world.reshape(h,w,3).astype(np.float32)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    index=json.loads((DATA/'index.json').read_text())
    rows=[next(r for r in index if r['task']==task and r['source_kind']=='complete_original')
          for task in ('touch','pick','move','pick_place')]
    report=[]; rng=np.random.default_rng(202672)
    for row in rows:
        source=Path(row['source']); root=source.parent
        original=next(r for r in json.loads((root/'index.json').read_text()) if r['file']==source.name)
        if sha256_file(source)!=row['source_sha256']: raise ValueError('TRAIN source changed')
        with np.load(row['cache'],allow_pickle=False) as f: numbers={k:f[k] for k in f.files}
        with np.load(source,allow_pickle=False) as f:
            reference={k:f[k] for k in ('front_rgb','wrist_rgb','joint_positions','gripper_closure')}
        env=audit_environment(root,robot='ur5e',image_size=384,control_hz=10,seed=0,
            required_success_steps=1,use_semantic_success=False)
        buffers=[]; clouds=[]; views=[]; projections=[]; metrics=[]
        try:
            obs=reset_source_episode(env,root,original)
            cfg=env.observation_profile['cameras']['front']
            for fi, command in enumerate(numbers['canonical7']):
                measured=scene_state(env,obs)
                error=float(np.abs(measured-numbers['scene64'][fi]).max())
                if error>1e-6: raise ValueError(f'Original TRAIN pre-action replay drift at{fi}: {error}')
                errors=reset_observation_errors(obs,env.normalized_gripper_closure(),
                    {k:v[fi:fi+1] for k,v in reference.items()})
                validate_reset_observation(errors)
                geometry=camera_geometry(env,cfg,'front')
                view=p.computeViewMatrix(*[v.tolist() for v in geometry])
                proj=p.computeProjectionMatrixFOV(float(cfg['fov_degrees']),1.,.01,3.)
                result=p.getCameraImage(384,384,viewMatrix=view,projectionMatrix=proj,
                    renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                    physicsClientId=env.client_id)
                rgb=np.asarray(result[2],np.uint8).reshape(384,384,4)[:,:,:3]
                buffer=np.asarray(result[3],np.float32).reshape(384,384)
                if not np.array_equal(rgb,obs.front_rgb): raise ValueError('Depth camera does not match actual policy RGB exactly')
                seg=np.asarray(result[4],np.int64).reshape(384,384)
                world=cloud(buffer,view,proj); depth=metric_depth(buffer)
                # A physically meaningful independent check in metric units.
                camera=np.c_[world.reshape(-1,3),np.ones(384*384)]@np.asarray(view).reshape(4,4)
                z_error=float(np.abs(-camera[:,2].reshape(384,384)-depth).max())
                if z_error>1e-3: raise ValueError('Depth buffer/metric/world-camera convention differs')
                target=(seg&((1<<24)-1))==env.pick_id
                if target.any():
                    low,high=map(np.asarray,p.getAABB(env.pick_id,physicsClientId=env.client_id))
                    points=world[target]
                    aabb_error=float(np.maximum(low-points,points-high).clip(0).max())
                    if aabb_error>.008: raise ValueError('Projected visual target is inconsistent with actual world AABB')
                else: aabb_error=None
                flat=world.reshape(-1,3)
                keep=((flat[:,0]>-.5)&(flat[:,0]<.5)&(flat[:,1]>-.9)&(flat[:,1]<0)&
                      (flat[:,2]>-.02)&(flat[:,2]<.6)&(buffer.ravel()<.99999))
                ids=np.flatnonzero(keep)
                if len(ids)<2048: raise ValueError('Too few measured workspace surface points')
                chosen=rng.choice(ids,2048,replace=False)
                points=np.c_[flat[chosen],rgb.reshape(-1,3)[chosen]/255.].astype(np.float32)
                buffers.append(depth);clouds.append(points);views.append(view);projections.append(proj)
                metrics.append({'frame':fi,'RGB_exactly_same_as_policy':True,'scene64_error':error,
                    'metric_depth_camera_Z_error_m':z_error,'target_AABB_error_m':aabb_error,
                    'target_pixels':int(target.sum()),'available_workspace_points':int(len(ids))})
                if fi in (0,len(numbers['canonical7'])-1):
                    Image.fromarray(rgb).save(a.output/f'{row["task"]}_t{fi}_RGB.png')
                    grey=np.uint8(np.clip((depth-.2)/1.2,0,1)*255)
                    Image.fromarray(grey).save(a.output/f'{row["task"]}_t{fi}_DEPTH_VISUALIZATION.png')
                obs,_,done,info=env.step(command)
                if info.get('safety_violation'): raise ValueError('Safety abort in historical TRAIN replay')
                if done and fi!=len(numbers['canonical7'])-1: raise ValueError('Historical termination differs')
            file=a.output/(row['task']+'_PAIRED_DEPTH_CANDIDATE.npz')
            np.savez_compressed(file,front_depth_z_m=np.stack(buffers),world_point_cloud_xyzrgb=np.stack(clouds),
                view_world_to_OpenGL_camera=np.asarray(views),projection_OpenGL=np.asarray(projections),
                source_frame_indices=np.arange(len(buffers)))
            report.append({'task':row['task'],'frames':len(buffers),'source':str(source),
                'source_sha256':row['source_sha256'],'file':str(file),'sha256':sha256_file(file),'measurements':metrics})
            atomic_json(a.output/'progress.json',{'completed_TRAIN_episodes':len(report),'rows':report,
                'FALCON_training_admitted':False,'native_model_SR_measured':False})
        finally: env.close()
    atomic_json(a.output/'AUDIT.json',{'passed_depth_pairing_and_physical_replay':True,'episodes':4,
        'frames':sum(r['frames'] for r in report),'rows':report,'native_model_SR_measured':False,
        'FALCON_training_admitted':False,'depth_is_actual_camera_render_not_synthetic_geometry':True,
        'point_cloud_frame':'metric_world_XYZ_RGB01; fixed workspace crop; uniform2048 without replacement',
        'remaining_gates':['same pairing audit over full qualified TRAIN bank',
            'explicit FALCON native coordinate/crop/normalization adapter + native forward/gradient/reload',
            'actual model fine-tuning and closed-loop evaluation']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
