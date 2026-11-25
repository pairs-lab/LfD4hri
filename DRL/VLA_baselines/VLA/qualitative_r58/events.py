"""Extend observed acquisition-miss evidence, without invented cognition."""
import argparse
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.qualitative_r57 import events as prior
from .common import read


def empty_lift_events(frames, proof, evidence):
    """A modest incidental push is not a successful lift of the object.

    Use separate vertical evidence instead of rejecting a failed acquisition
    solely because the peach slid a few centimetres. This changes only event
    classification, never simulation, recorded commands or task success.
    """
    if (proof['benchmark_region_success'] or proof['safety_violation']
            or evidence['held_frames'] or not frames):
        return []
    pad=np.asarray([r['pad_center'] for r in frames],float)
    obj=np.asarray([r['object_position'] for r in frames],float)
    closed=np.asarray([r['gripper_command']>.5 for r in frames])
    near=np.flatnonzero((np.linalg.norm(pad-obj,axis=1)<.065)&closed)
    for idx in near:
        stop=min(len(frames),int(idx)+120)
        lifted=int(idx)+int(np.argmax(pad[idx:stop,2]))
        tcp_lift=float(pad[lifted,2]-pad[idx,2])
        object_lift=float(np.max(obj[idx:lifted+1,2])-obj[idx,2])
        separation=float(pad[lifted,2]-obj[lifted,2])
        if tcp_lift>=.08 and object_lift<.02 and separation>=.08:
            return [{'mode':'empty_gripper_lift_after_failed_acquisition_no_object_lift',
                'event_frame':int(idx),'time_s':frames[idx]['simulation_time_s'],
                'before_frame':max(0,int(idx)-6),'after_frame':lifted,
                'TCP_lift_m':tcp_lift,'object_vertical_lift_during_empty_lift_m':object_lift,
                'vertical_pad_object_separation_at_lift_m':separation,
                'target_displacement_m':float(np.linalg.norm(obj-obj[0],axis=1).max()),
                'verified_hold_frames':0,'incidental_object_sliding_is_not_lifting':True,
                'no_claim_that_model_believed_grasp_succeeded':True}]
    return []


def main(a):
    prior.main(a);result=read(a.capture/'content_evidence.json')
    frames=read(a.capture/'recorded_physics_states.json')
    if (a.capture/'replay_proof.json').exists():
        proof=read(a.capture/'replay_proof.json');controls=proof['controls']
        xy=np.asarray([r['pad_center'][:2] for r in frames])-np.asarray([r['object_position'][:2] for r in frames])
        minimum=float(np.linalg.norm(xy,axis=1).min())
        positions=np.asarray([r['object_position'] for r in frames])
        movement=float(np.linalg.norm(positions-positions[0],axis=1).max())
        closed=[i for i,r in enumerate(controls) if r['requested_action'][6]>.5]
        if (not proof['benchmark_region_success'] and not proof['safety_violation'] and result['held_frames']==0
                and minimum>.08 and movement<.01 and len(closed)>=5):
            first=closed[0];idx=min(len(frames)-1,first*3)
            result['events'].append({'mode':'clear_planar_acquisition_miss_empty_closure_no_recovery',
                'event_frame':idx,'time_s':first/10.,'before_frame':max(0,idx-6),'after_frame':len(frames)-1,
                'minimum_pad_target_XY_over_whole_rollout_m':minimum,
                'requested_close_controls':len(closed),'target_displacement_m':movement,
                'verified_correct_hold_frames':0,'lateral_miss_precedes_any_terminal_guard_stall':True,
                'no_inference_of_memory_loss_or_wrong_identity':True})
        # A visible failed acquisition followed by lifting an EMPTY gripper is
        # distinct from dropping/releasing an already carried object.
        result['events'].extend(empty_lift_events(frames,proof,result))
    else:
        outcome=read(a.capture/'outcome.json')
        error=outcome.get('error') or ''
        if (result['held_frames']==0 and 'No verified bilateral grasp' in error
                and outcome['fault_script'] in ('persistent_miss','closed_jaw_approach')):
            result['events'].append({'mode':'controlled_Oracle_failed_acquisition_empty_closure',
                'event_frame':len(frames)-1,'time_s':frames[-1]['simulation_time_s'],
                'before_frame':max(0,len(frames)-61),'after_frame':len(frames)-1,
                'fault_events':outcome['fault_events'],'scripted_Oracle_NOT_intrinsic_learned_RL_failure':True})
    atomic_json(a.capture/'content_evidence.json',result)
    print({'capture':str(a.capture),'full_put':result['full_put_complete'],'modes':[e['mode'] for e in result['events']]},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True)
    main(parser.parse_args())
