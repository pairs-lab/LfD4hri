"""Dense surface-contact evidence; force alone is not a violent slam."""
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from .common import OUT,read


def summarize(capture):
    ticks=read(Path(capture)/'physics_event_ticks.json')
    forces=np.asarray([r['ground_force_N'] for r in ticks]);xyz=np.asarray([r['pad_center'] for r in ticks])
    groups=[]
    for i in np.flatnonzero(forces>1.):
        if not groups or i-groups[-1][-1]>12:groups.append([])
        groups[-1].append(int(i))
    events=[]
    for group in groups:
        start=group[0];peak=max(group,key=lambda i:forces[i]);pre=max(0,start-12);after=min(len(ticks)-1,start+24)
        events.append({'first_contact_time_s':ticks[start]['simulation_time_s'],
            'peak_force_N':float(forces[peak]),'peak_time_s':ticks[peak]['simulation_time_s'],
            'entry_vertical_velocity_50ms_m_per_s':float((xyz[start,2]-xyz[pre,2])/((start-pre)/240.)) if start>pre else None,
            'TCP_vertical_change_100ms_after_contact_m':float(xyz[after,2]-xyz[start,2]),
            'duration_above_1N_s':len(group)/240.,
            'near_first_video_frame':round(ticks[start]['simulation_time_s']*30)-1,
            'near_peak_video_frame':round(ticks[peak]['simulation_time_s']*30)-1})
    return {'peak_gripper_ground_force_N':float(forces.max()),'events':events,
        'verified_correct_hold_ticks_240Hz':sum(bool(r['correct_verified_hold']) for r in ticks),
        'forces_are_PyBullet_solver_measurements_NOT_calibrated_real_robot_forces':True,
        'force_alone_does_not_establish_high_speed_impact_or_root_cause':True}


def main():
    rows=[]
    for row in read(OUT/'assignments.json'):
        if row['VLA_full_put']:continue
        rows.append({'scene':row['scene_id'],'seed':row['seed'],'backup':row['backup'],
            **summarize(row['vla'])})
    atomic_json(OUT/'audit/dense_ground_contact_counterevidence.json',rows)
    print({'ground_contact_counterevidence_rows':len(rows),'violent_slam_NOT_inferred':True},flush=True)


if __name__=='__main__':main()
