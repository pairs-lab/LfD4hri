"""Full qualified TRAIN paired-depth candidate with exact recovery replay.

Offline replay only; never evaluates a learned model or changes its scoring.
Every frame must reproduce saved physical state and established RGB limits.
"""
import argparse
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.observation_profile import camera_geometry
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.dataset_replay_context import audit_environment,reset_source_episode
from VLA.evaluate_smolvla_v9_source import reset_observation_errors,validate_reset_observation
from VLA.family_transfer_r54.common import DATA,read
from VLA.falcon_depth_pilot_r72 import metric_depth,cloud
from VLA.simstate_scene64 import scene_state


@lru_cache(maxsize=8)
def index(root):return {r['file']:r for r in read(Path(root)/'index.json') if r.get('file')}


def source_context(row):
    root=Path(row['source']).parent;item=index(str(root))[Path(row['source']).name]
    if row['source_kind']=='complete_original':return root,item,[],item
    origin=Path(read(root/'plan.json')['source']);base=index(str(origin))[item['source_file']]
    if row['source_kind']=='pose_corrective':
        actions=[r['action'] for r in item['perturbation_actions_NOT_labels']]
    elif row['source_kind']=='fresh_corrective':actions=item['recovery_start_actions_not_training_labels']
    else:raise ValueError('Unsupported qualified source role')
    return origin,base,actions,item


def reset(row):
    root,original,actions,item=source_context(row)
    # Offline rendering must not stop early at an old teacher's more strict
    # near-surface endpoint. This changes replay termination bookkeeping only;
    # every actual physical state is compared to the recorded reference.
    env=audit_environment(root,robot='ur5e',image_size=384,control_hz=10,seed=0,
        required_success_steps=10000,use_semantic_success=False)
    try:
        obs=reset_source_episode(env,root,original)
        for action in actions:
            obs,_,_,info=env.step(np.asarray(action,np.float32))
            if info.get('safety_violation'):raise ValueError('Historical startup prefix hits physical safety guard')
        if actions:
            env.step_count=env.success_streak=0;env._last_success=False
            if item.get('success_contract'):env.dist_threshold,env.obj_threshold=recorded_distance_thresholds(item)
            obs=env.observe()
        return env,obs
    except Exception:
        env.close();raise


def select(rows,mode):
    if mode=='all':return list(enumerate(rows))
    result=[]
    for task in ('touch','pick','move','pick_place'):
        for kind,recovery in (('fresh_corrective',None),('pose_corrective','wrist_tilt'),('pose_corrective','closed_empty')):
            result.append(next((i,r) for i,r in enumerate(rows) if r['task']==task and r['source_kind']==kind and
                (recovery is None or r.get('recovery_kind')==recovery)))
    return result


def main(a):
    if a.output.exists():raise FileExistsError('Preserve old candidate and failures')
    a.output.mkdir(parents=True)
    rows=read(DATA/'index.json');selected=select(rows,a.mode)
    bound={str(p.resolve()):sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('falcon_depth_pilot_r72.py'),DATA/'index.json',DATA/'stats.json')}
    atomic_json(a.output/'PLAN.json',{'mode':a.mode,'selected_rows':[i for i,_ in selected],
        'qualified_TRAIN_episodes':2185,'qualified_TRAIN_frames':119495,'candidate_only_NOT_admitted':True,
        'robot_SR_measured':False,'no_heldout_or_final800_used':True,'code_and_index':bound,
        'offline_replay_success_streak10000_does_not_change_deployed_evaluation':True})
    outputs=[];current=None
    try:
        for ei,row in selected:
            current=ei
            for path,digest in bound.items():
                if sha256_file(path)!=digest:raise ValueError('Bound replay code/index changed')
            source=Path(row['source']);cache=DATA/row['cache']
            if sha256_file(source)!=row['source_sha256'] or sha256_file(cache)!=row['cache_sha256']:
                raise ValueError('Qualified RGB/action or numeric source changed')
            with np.load(cache,allow_pickle=False) as f:raw=f['scene64'];commands=f['canonical7']
            with np.load(source,allow_pickle=False) as f:
                reference={k:f[k] for k in ('front_rgb','wrist_rgb','joint_positions','gripper_closure')}
            env,obs=reset(row);depths=[];clouds=[];views=[];metrics=[];rng=np.random.default_rng(202673+ei)
            try:
                cfg=env.observation_profile['cameras']['front'];proj=p.computeProjectionMatrixFOV(float(cfg['fov_degrees']),1.,.01,3.)
                for fi,action in enumerate(commands):
                    error=float(np.abs(scene_state(env,obs)-raw[fi]).max())
                    if error>1e-6:raise ValueError(f'row{ei} frame{fi} physical state drift {error}')
                    errors=reset_observation_errors(obs,env.normalized_gripper_closure(),
                        {k:v[fi:fi+1] for k,v in reference.items()});validate_reset_observation(errors)
                    geometry=camera_geometry(env,cfg,'front');view=p.computeViewMatrix(*[v.tolist() for v in geometry])
                    frame=p.getCameraImage(384,384,viewMatrix=view,projectionMatrix=proj,
                        renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                        physicsClientId=env.client_id)
                    rgb=np.asarray(frame[2],np.uint8).reshape(384,384,4)[:,:,:3]
                    if not np.array_equal(rgb,obs.front_rgb):raise ValueError('Actual front RGB/depth camera pair differs')
                    buffer=np.asarray(frame[3],np.float32).reshape(384,384);world=cloud(buffer,view,proj)
                    depth=metric_depth(buffer)
                    camera=np.c_[world.reshape(-1,3),np.ones(384*384)]@np.asarray(view).reshape(4,4)
                    zerr=float(np.abs(-camera[:,2].reshape(384,384)-depth).max())
                    if zerr>1e-3:raise ValueError('Metric depth/projected camera-Z mismatch')
                    flat=world.reshape(-1,3)
                    keep=((flat[:,0]>-.5)&(flat[:,0]<.5)&(flat[:,1]>-.9)&(flat[:,1]<0)&
                        (flat[:,2]>-.02)&(flat[:,2]<.6)&(buffer.ravel()<.99999))
                    ids=np.flatnonzero(keep)
                    if len(ids)<2048:raise ValueError('Insufficient actual workspace points')
                    chosen=rng.choice(ids,2048,replace=False)
                    depths.append(depth);clouds.append(np.c_[flat[chosen],rgb.reshape(-1,3)[chosen]/255.].astype(np.float32));views.append(view)
                    metrics.append({'frame':fi,'scene64_max_abs_error':error,'metric_depth_Z_error_m':zerr,
                        'RGB_pair_bit_exact':True,'original_sensor_error':errors})
                    obs,_,_,info=env.step(action)
                    if info.get('safety_violation'):raise ValueError('Safety abort in original physics replay')
            finally:env.close()
            dest=a.output/f'paired_depth_{ei:05d}.npz';temp=dest.with_suffix('.npz.pending')
            with temp.open('xb') as f:np.savez_compressed(f,front_depth_z_m=np.stack(depths),world_point_cloud_xyzrgb=np.stack(clouds),
                view_world_to_OpenGL_camera=np.asarray(views),projection_OpenGL=np.asarray(proj),source_frame_indices=np.arange(len(depths)))
            os.replace(temp,dest)
            outputs.append({'qualified_row':ei,'source_kind':row['source_kind'],'task':row['task'],'frames':len(depths),
                'source':str(source),'source_sha256':row['source_sha256'],'file':str(dest),'sha256':sha256_file(dest),
                'max_state_error':max(m['scene64_max_abs_error'] for m in metrics),
                'max_depth_Z_error_m':max(m['metric_depth_Z_error_m'] for m in metrics)})
            atomic_json(a.output/'progress.json',{'completed':False,'episodes':len(outputs),'planned':len(selected),
                'frames':sum(r['frames'] for r in outputs),'rows':outputs,'admitted_for_FALCON_training':False})
        passed=len(outputs)==len(selected)
        atomic_json(a.output/'AUDIT.json',{'passed_pairing_replay':passed,'mode':a.mode,'episodes':len(outputs),
            'frames':sum(r['frames'] for r in outputs),'rows':outputs,'admitted_for_FALCON_training':False,
            'remaining_gates':['native FALCON point-cloud frame/normalization and privileged-state adapter',
                'actual native forward/backward/optimizer/reload memory qualification','finetune and heldout neural trials'],
            'point_cloud_format':'world XYZ meters and RGB01; fixed measured-workspace crop; random2048 without replacement',
            'learned_VLA_SR_measured':False,'source_and_code':bound})
        atomic_json(a.output/'progress.json',{'completed':True,'episodes':len(outputs),'planned':len(selected),
            'frames':sum(r['frames'] for r in outputs),'admitted_for_FALCON_training':False})
    except Exception as e:
        atomic_json(a.output/'ERROR.json',{'error':str(e),'qualified_row':current,'completed_episodes':len(outputs),
            'candidate_not_admitted':True,'no_failed_rows_silently_filtered':True});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('recovery12','all'),default='recovery12');main(parser.parse_args())
