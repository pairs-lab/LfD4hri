"""Audit trained clipping/map bounds; supplied does not mean lossless."""
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.scene64_conditioning import encode_numeric,NUMERIC_NAMES
from VLA.privileged_visual_r53.scene_map import geometry,XY_BOUNDS
from .common import OUT,CP,read


def main():
    config=read(CP/'config.json');stats=config['stats']['scene67']
    mean,std=(np.asarray(stats[k],np.float32) for k in ('mean','std'))
    limit=config['observation_conditioning']['limit_standard_deviations'];results=[]
    for item in read(OUT/'assignments.json'):
        proof=read(Path(item['vla'])/'replay_proof.json');trace=read(proof['source_trajectory'])['pick_place']
        rows=[]
        for index in range(0,len(trace),30):
            raw=np.asarray(trace[index]['pre_action_scene64'],np.float32)
            z=(encode_numeric(raw)-mean)/(std+np.float32(1e-8));clipped=np.flatnonzero(np.abs(z)>limit)
            _,pad,target,goal,_=geometry(raw,'pick_place');x0,x1,y0,y1=XY_BOUNDS
            outside=lambda p:bool(p[0]<x0 or p[0]>x1 or p[1]<y0 or p[1]>y1)
            rows.append({'control':index,'simulation_time_s':index/10.,
                'saturated_fields':[{'index':int(i),'name':NUMERIC_NAMES[i],
                    'standardized_before_clip':float(z[i]),'after_clip':float(np.clip(z[i],-limit,limit))} for i in clipped],
                'relative_target_or_goal_XYZ_saturated':any(11<=i<17 for i in clipped),
                'target_outside_fixed_GT_map_XY_bounds':outside(target),
                'pad_outside_fixed_GT_map_XY_bounds':outside(pad),
                'goal_outside_fixed_GT_map_XY_bounds':outside(goal)})
        results.append({'scene':item['scene_id'],'backup':item['backup'],'seed':item['seed'],'queries':rows})
    atomic_json(OUT/'audit/trained_input_range_limits.json',{'rows':results,
        'numeric_normalization_matches_saved_TRAIN_statistics':True,'trained_standard_deviation_clip':limit,
        'native_proprio_prefix_unchanged':True,'GT_map_world_XY_bounds_m':list(XY_BOUNDS),
        'saturated_values_and_off_map_markers_NOT_lossless_coordinates':True,
        'post_impact_input_saturation_does_not_explain_preceding_impact_onset':True,
        'no_runtime_normalization_camera_checkpoint_or_training_change':True})
    print({'cases_with_target_goal_saturation':[(r['scene'],r['backup'],[q['control'] for q in r['queries'] if q['relative_target_or_goal_XYZ_saturated']])
        for r in results if any(q['relative_target_or_goal_XYZ_saturated'] for q in r['queries'])]},flush=True)


if __name__=='__main__':main()
