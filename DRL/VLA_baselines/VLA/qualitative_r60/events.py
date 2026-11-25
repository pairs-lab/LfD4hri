"""Evidence for high-speed ground contact and off-target empty closure."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.qualitative_r58.events import main as prior
from VLA.qualitative_r57.common import read
from VLA.common.artifact_io import atomic_json


def main(a):
    prior(SimpleNamespace(capture=a.capture))
    evidence=read(a.capture/'content_evidence.json')
    frames=read(a.capture/'recorded_physics_states.json')
    dense=read(a.capture/'physics_event_ticks.json')
    if (a.capture/'outcome.json').exists():
        outcome=read(a.capture/'outcome.json')
        if outcome.get('fault_script')=='STAGED_wrong_ground_height':
            xyz=np.array([d['pad_center'] for d in dense]);forces=np.array([d['ground_force_N'] for d in dense])
            vel=np.diff(xyz,axis=0)*240.
            hits=np.flatnonzero(forces>20.)
            if len(hits):
                idx=int(hits[0]);v=float(vel[max(0,idx-4):max(1,idx-1),2].min())
                peak=float(forces[idx:min(len(dense),idx+12)].max())
                event_frame=int(np.argmin([abs(f['simulation_time_s']-dense[idx]['simulation_time_s']) for f in frames]))
                strong=peak>=100 and v<=-.25
                event={'mode':'STAGED_Oracle_Ground_Slam' if strong else 'STAGED_Oracle_Ground_Contact',
                    'event_frame':event_frame,'time_s':dense[idx]['simulation_time_s'],
                    'before_frame':max(0,event_frame-9),'after_frame':min(len(frames)-1,event_frame+24),
                    'incoming_downward_TCP_velocity_m_per_s':v,'peak_ground_force_next50ms_N':peak,
                    'first_contact_XYZ_m':xyz[idx].tolist(),'contact_ticks':int((forces>20).sum()),
                    'ground_slam_force_velocity_gate_pass':bool(strong),
                    'learned_RL_failure_claimed':False,'native_physics_guard_retained':True}
                evidence['events'].append(event)
    if (a.capture/'replay_proof.json').exists():
        proof=read(a.capture/'replay_proof.json')
        trace=read(proof['source_trajectory'])['pick_place']
        closed=[i for i,r in enumerate(trace) if r['action'][6]>.5]
        if closed and not evidence['held_frames'] and not proof['safety_violation'] and not proof['benchmark_region_success']:
            first=closed[0];r=trace[first]
            delta=np.array(r['before']['pad_grasp_center_xyz_m'])-r['before']['object_center_xyz_m']
            obj=np.array([f['object_position'] for f in frames]);motion=float(np.linalg.norm(obj-obj[0],axis=1).max())
            if np.linalg.norm(delta[:2])>.10 and motion<.01:
                frame=min(len(frames)-1,first*3)
                evidence['events'].append({'mode':'native_Target_Miss_then_Failed_Recovery',
                    'event_frame':frame,'time_s':first/10.,'before_frame':max(0,frame-15),
                    'after_frame':len(frames)-1,
                    'first_close_control':first,'first_close_target_XY_separation_m':float(np.linalg.norm(delta[:2])),
                    'first_close_target_vertical_separation_m':float(delta[2]),
                    'object_maximum_displacement_m':motion,'verified_hold_frames':0,
                    'control_count':len(trace),'native_actions_unmodified':True,
                    'source_is_completed_frozen_quantitative_trace_selected_posthoc':True,
                    'semantic_misidentification_and_memory_loss_NOT_proven':True,
                    'GT_remained_available_to_policy':True})
    atomic_json(a.capture/'content_evidence.json',evidence)
    print({'events':evidence['events'],'full_put_complete':evidence['full_put_complete']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);main(p.parse_args())
