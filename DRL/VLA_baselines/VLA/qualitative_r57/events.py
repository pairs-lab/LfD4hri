"""Conservative evidence classifier: observable events, not invented cognition."""
import argparse
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from .common import read


def classify(frames,dense):
    if not frames:return {'valid':False,'reason':'no real scheduled video frames'}
    result={'valid':True,'events':[],'full_put_complete':False,
        'cognitive_misidentification_or_memory_loss_proven':False}
    times=np.asarray([r['simulation_time_s'] for r in frames])
    xyz=np.asarray([r['object_position'] for r in frames]);initial=xyz[0,2]
    held=np.asarray([r['correct_verified_hold'] for r in frames],bool)
    goal=np.asarray([r['goal_position'] for r in frames])
    error=np.linalg.norm(xyz[:,:2]-goal[:,:2],axis=1)
    speed=np.asarray([np.linalg.norm(r['object_linear_velocity']) for r in frames])
    support=np.asarray([r['destination_support_force_N'] for r in frames])
    command=np.asarray([r['gripper_command'] for r in frames])
    attached=np.asarray([r.get('correct_attached',r['correct_verified_hold']) for r in frames],bool)
    stable=(~attached)&(error<=.045)&(speed<.025)&(support>.005)&(command<-.2)
    enough_hold=int(held.sum())>=3
    # Require a continuous0.5s real support/settle interval after an actual hold.
    for i in range(15,len(frames)):
        if enough_hold and stable[i-15:i+1].all() and held[:i-15].any():
            result['full_put_complete']=True;result['stable_completion_frame']=i;break
    # Losing bilateral contact while the same fixed grasp constraint still
    # holds the object is NOT a drop. Require actual attachment loss.
    drops=np.flatnonzero(attached[:-1]&~attached[1:])+1
    for i in drops:
        stop=int(min(len(frames),int(i)+10))
        if (held[:i].any() and xyz[:i,2].max()>initial+.03 and error[max(i-1,0)]>.08
            and xyz[i:stop,2].min()<xyz[i-1,2]-.015):
            result['events'].append({'mode':'verified_in_transit_drop','event_frame':int(i),
                'time_s':float(times[i]),'before_frame':max(0,int(i)-6),'after_frame':stop-1,
                'actual_correct_hold_before_drop':True,'away_from_goal_XY_m':float(error[i-1]),
                'measured_fall_m':float(xyz[i-1,2]-xyz[i:stop,2].min()),
                'event_is_not_deliberate_at_goal_release':True})
    if dense:
        velocities=np.asarray([r['object_velocity'] for r in dense]);positions=np.asarray([r['object_position'] for r in dense])
        forces=np.asarray([r['target_force_N'] for r in dense]);dense_held=np.asarray([r['correct_verified_hold'] for r in dense])
        impacts=[]
        for i in np.flatnonzero((forces>1.)&(~dense_held)):
            stop=min(len(dense),i+60)
            displacement=float(np.linalg.norm(positions[i:stop]-positions[i],axis=1).max())
            maximum_speed=float(np.linalg.norm(velocities[i:stop],axis=1).max())
            if displacement>.03 and maximum_speed>.25:
                t=dense[i]['simulation_time_s'];frame=int(np.argmin(abs(times-t)))
                impacts.append((int(i),{'mode':'contact_confirmed_impact_displacement_NOT_automatically_elastic_bounce',
                    'event_frame':frame,'time_s':t,'before_frame':max(0,frame-6),
                'after_frame':int(min(len(frames)-1,frame+12)),'force_N':float(forces[i]),
                    'object_displacement_within250ms_m':displacement,'maximum_object_speed_m_per_s':maximum_speed,
                    'no_verified_hold_at_impact':True}))
        # A small early contact must not hide a later, visually much stronger
        # ejection. Keep the greatest measured displacement per contact burst,
        # separating bursts by more than50ms at the frozen240Hz physics rate.
        bursts=[]
        for tick,event in impacts:
            if not bursts or tick-bursts[-1][-1][0]>12:bursts.append([])
            bursts[-1].append((tick,event))
        result['events'].extend(max(burst,key=lambda pair:pair[1]['object_displacement_within250ms_m'])[1]
            for burst in bursts)
    result['held_frames']=int(held.sum());result['maximum_lift_m']=float(xyz[:,2].max()-initial)
    result['terminal_goal_XY_m']=float(error[-1]);result['terminal_object_speed_m_per_s']=float(speed[-1])
    result['no_drop_claim_without_verified_hold_and_fall']=True
    result['no_slam_claim_from_safety_abort_alone']=True
    result['manual_video_event_review_REQUIRED']=True
    return result


def main(a):
    frames=read(a.capture/'recorded_physics_states.json')
    result=classify(frames,read(a.capture/'physics_event_ticks.json'))
    if (a.capture/'outcome.json').exists():
        outcome=read(a.capture/'outcome.json');error=outcome.get('error') or ''
        if error.startswith('TimeoutError') and frames[-1]['simulation_time_s']>=21.9:
            result['events'].append({'mode':'controlled_Oracle_transport_timeout',
                'event_frame':len(frames)-1,'time_s':frames[-1]['simulation_time_s'],
                'before_frame':max(0,len(frames)-61),'after_frame':len(frames)-1,
                'actual_fault_events':outcome['fault_events'],
                'scripted_controller_constraint_NOT_intrinsic_RL_failure':True})
        elif ('No verified bilateral grasp' in error and outcome['fault_script']=='approach_overshoot'
              and outcome['fault_events'] and result['held_frames']==0):
            t=outcome['fault_events'][0]['time_s'];idx=min(len(frames)-1,max(0,round(t*30)-1))
            result['events'].append({'mode':'controlled_Oracle_approach_miss_without_grasp',
                'event_frame':idx,'time_s':t,'before_frame':max(0,idx-6),'after_frame':len(frames)-1,
                'actual_fault_events':outcome['fault_events'],'no_drop_or_slam_claim':True,
                'scripted_controller_fault_NOT_intrinsic_RL_failure':True})
    if (a.capture/'replay_proof.json').exists():
        proof=read(a.capture/'replay_proof.json');controls=proof['controls']
        trace=read(proof['source_trajectory'])['pick_place']
        start=None
        for i,row in enumerate(controls):
            absent=row['actual_policy_visibility']['wrist']['target_pixels']==0
            if absent and start is None:start=i
            if start is not None and (not absent or i==len(controls)-1):
                end=i if absent else i-1
                if end-start>=29:
                    poses=np.asarray([r['after']['pad_grasp_center_xyz_m'] for r in trace[start:end+1]])
                    path=float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum())
                    net=float(np.linalg.norm(poses[-1]-poses[0]))
                    if net<.03 and path<.10:
                        requested=np.asarray([r['requested_action'] for r in controls[start:end+1]])
                        executed=np.asarray([r['executed_action'] for r in controls[start:end+1]])
                        suppressed=float(((requested[:,2]<-.005)&(executed[:,2]>requested[:,2]+.003)).mean())
                        idx=min(len(frames)-1,round(start*3))
                        result['events'].append({'mode':'policy_wrist_target_absent_and_measured_pose_stall_NOT_memory_loss',
                            'event_frame':idx,'time_s':start/10.,'before_frame':max(0,idx-6),
                            'after_frame':min(len(frames)-1,3*(end+1)-1),'duration_s':(end-start+1)/10.,
                            'TCP_path_m':path,'TCP_net_displacement_m':net,
                            'fraction_downward_actions_limited_by_shared_clearance_guard':suppressed,
                            'GT_still_supplied_at_native_queries':True,'memory_loss_not_proven':True})
                start=None
    result['source_capture']=str(a.capture.resolve())
    atomic_json(a.capture/'content_evidence.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);main(parser.parse_args())
