"""Candidate observed-surface point sampler, not an action/geometric planner.

Uses current sensor segmentation permitted in the privileged setting. Keeps
native centroid/radius normalization; provides its inverse transform for a
future conditioning adapter. Never creates unseen points from object meshes.
"""
import argparse
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.common import read
from VLA.falcon_depth_pilot_r72 import cloud


def sample(depth,view,projection,pick,place,robot,rng):
    near,far=.01,3.
    buffer=far*(depth-near)/(depth*(far-near))
    world=cloud(buffer,view,projection);flat=world.reshape(-1,3)
    valid=((flat[:,0]>-.5)&(flat[:,0]<.5)&(flat[:,1]>-.9)&(flat[:,1]<0)&
        (flat[:,2]>-.02)&(flat[:,2]<.6)&(buffer.ravel()<.99999))
    # Match the native filter order: current-cloud centroid,1m outlier filter,
    # zero centering and max-radius scaling BEFORE selecting2048 points.
    center=flat[valid].mean(0);valid &= np.linalg.norm(flat-center,axis=1)<1.
    center=flat[valid].mean(0);radius=float(np.linalg.norm(flat[valid]-center,axis=1).max())
    if radius<1e-6 or int(valid.sum())<2048:raise ValueError('No valid measured cloud')
    chosen=[];used=np.zeros(len(flat),bool)
    def take(mask,n):
        ids=np.flatnonzero(valid&mask&~used)
        if len(ids):
            selected=rng.choice(ids,min(n,len(ids)),replace=False);used[selected]=True;chosen.extend(selected.tolist())
    take(pick.ravel(),128);take(place.ravel(),128);take(robot.ravel(),256)
    foreground=(flat[:,2]>.008)&~robot.ravel()&~pick.ravel()&~place.ravel()
    take(foreground,1024)
    take(np.ones(len(flat),bool),2048-len(chosen))
    ids=np.asarray(chosen,np.int64)
    if len(ids)!=2048 or len(np.unique(ids))!=2048:raise ValueError('Never fabricate/duplicate an observed point')
    points=((flat[ids]-center)/radius).astype(np.float32)
    reconstruction=float(np.abs(points*radius+center-flat[ids]).max())
    receipt={'target_visible_workspace_pixels':int((pick.ravel()&valid).sum()),
        'sampled_target_points':int(pick.ravel()[ids].sum()),
        'sampled_place_points':int(place.ravel()[ids].sum()),
        'sampled_robot_points':int(robot.ravel()[ids].sum()),
        'near_table_fraction':float(np.mean(flat[ids,2]<.006)),
        'centroid_world_m':center.tolist(),'radius_m':radius,
        'inverse_normalization_max_abs_error_m':reconstruction,
        'privileged_mask_sampling_declared':True,'geometry_from_actual_depth_only':True,
        'native_input_shape':[2048,3],'no_waypoints_or_actions_computed':True}
    return points,ids,receipt


def projected_mask_count(points,view,projection,mask):
    h,w=mask.shape
    v=np.asarray(view).reshape(4,4).T;proj=np.asarray(projection).reshape(4,4).T
    clip=np.c_[points,np.ones(len(points))]@(proj@v).T
    ndc=clip[:,:3]/clip[:,3,None]
    x=np.rint((ndc[:,0]+1)*w/2-.5).astype(int);y=np.rint((1-ndc[:,1])*h/2-.5).astype(int)
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)
    return int(mask[y[inside],x[inside]].sum())


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);pilot=read(a.source/'AUDIT.json');rows=[]
    for row in pilot['rows']:
        with np.load(row['source'],allow_pickle=False) as f:
            masks={k:f[k] for k in ('front_pick_mask','front_place_mask','front_robot_mask')}
        with np.load(row['file'],allow_pickle=False) as f:
            depth=f['front_depth_z_m'];views=f['view_world_to_OpenGL_camera'];projections=f['projection_OpenGL'];old=f['world_point_cloud_xyzrgb']
        rng=np.random.default_rng(202674)
        for fi in range(len(depth)):
            points,ids,r=sample(depth[fi],views[fi],projections[fi],masks['front_pick_mask'][fi],
                masks['front_place_mask'][fi],masks['front_robot_mask'][fi],rng)
            rows.append({'task':row['task'],'frame':fi,**r,
                'baseline_uniform_sampled_target_points':projected_mask_count(old[fi,:,:3],views[fi],projections[fi],masks['front_pick_mask'][fi]),
                'baseline_near_table_fraction':float(np.mean(old[fi,:,2]<.006))})
    visible=[r for r in rows if r['target_visible_workspace_pixels']>0]
    summary={'frames':len(rows),'actual_visible_target_frames':len(visible),
        'target_point_count_median_uniform':float(np.median([r['baseline_uniform_sampled_target_points'] for r in visible])),
        'target_point_count_median_privileged_sampling':float(np.median([r['sampled_target_points'] for r in visible])),
        'near_table_fraction_mean_uniform':float(np.mean([r['baseline_near_table_fraction'] for r in rows])),
        'near_table_fraction_mean_privileged_sampling':float(np.mean([r['near_table_fraction'] for r in rows])),
        'inverse_normalization_max_error_m':max(r['inverse_normalization_max_abs_error_m'] for r in rows),
        'native_FALCON_training_admitted':False,'neural_model_SR_measured':False,
        'remaining':'Integrate this explicit privileged sampler and center/radius metadata into a qualified native FALCON forward/loss/gradient/reload path.',
        'rows':rows}
    atomic_json(a.output/'POINT_ALLOCATION_AUDIT.json',summary)
    print({k:v for k,v in summary.items() if k!='rows'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
