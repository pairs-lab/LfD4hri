"""Actual-motion comparisons and success-first profile scoring for feedback Oracle."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter
from env_eval.audit_cinematic import motion_metrics


def feedback_metrics(samples):
    t=np.array([s['time'] for s in samples]);dt=float(np.median(np.diff(t)))
    joints=np.array([s['joints'] for s in samples]);tcp=np.array([s['tcp'] for s in samples])
    velocity=np.diff(joints,axis=0)/dt
    acceleration=np.diff(velocity,axis=0)/dt;jerk=np.diff(acceleration,axis=0)/dt
    smooth=savgol_filter(joints,7,3,axis=0) if len(joints)>=7 else joints
    smooth_jerk=np.diff(smooth,n=3,axis=0)/dt**3
    active=np.array([s['phase'] not in ('initial','final','settle') for s in samples])
    contact=[s for s in samples if s['phase'] in ('grasp','touch','touch_retract','release') or
             (s['phase']=='approach' and s.get('physical_contact',False))]
    times=[s['time'] for s in samples if s['held'] and s['bilateral']]
    first_contact=next((s['time'] for s in samples if s.get('contact_force_n',sum(s['pad_forces']))>.02),None)
    tracking=[np.linalg.norm(np.array(s['commanded_tcp'])-s['tcp']) for s in samples if 'commanded_tcp' in s]
    active_tracking=[np.linalg.norm(np.array(s['commanded_tcp'])-s['tcp']) for s in samples
                     if 'commanded_tcp' in s and s['phase'] in ('approach','grasp','carry','touch','touch_retract')]
    force=max((s.get('contact_force_n',sum(s['pad_forces'])) for s in contact),default=0.)
    # Joint high-frequency residual is measured in rad/s, using the same
    # seven-frame filter and 30 Hz sampling in both old and new runs.
    filtered_velocity=savgol_filter(velocity,7,3,axis=0)
    return {**motion_metrics(samples),
        'actual_joint_jerk_rms_rad_s3':float(np.sqrt(np.mean(jerk**2))),
        'actual_joint_jerk_peak_rad_s3':float(abs(jerk).max()),
        'filtered_joint_jerk_rms_rad_s3':float(np.sqrt(np.mean(smooth_jerk**2))),
        'joint_velocity_high_frequency_rms_rad_s':float(np.sqrt(np.mean((velocity-filtered_velocity)**2))),
        'path_length_m':float(np.linalg.norm(np.diff(tcp,axis=0),axis=1).sum()),
        'duration_seconds':float(t[-1]),
        'contact_to_verified_grasp_seconds':None if not times or first_contact is None else times[0]-first_contact,
        'contact_window_max_force_n':float(force),
        'max_sampled_tracking_error_m':max(tracking,default=None),
        'max_interaction_tracking_error_m':max(active_tracking,default=None)}


def score_run(audit,samples):
    """Model-based profile selection reward; this is not RL training."""
    m=feedback_metrics(samples)
    hard_pass=bool(audit.get('overall_pass') and m['motion_pass'] and
        (m['max_interaction_tracking_error_m'] or 0)<.015 and
        (audit['action']!='touch' or audit['max_touch_force_n']<5.) and
        audit['max_mount_translation_error_m']<(.001 if audit['robot']=='uf850' else .0012) and
        audit['max_mount_rotation_error_deg']<1.5)
    components={'jerk':m['actual_joint_jerk_rms_rad_s3']/10.,
                'high_frequency':m['joint_velocity_high_frequency_rms_rad_s']/.01,
                'duration':m['duration_seconds']/10.,
                'contact_latency':max(0.,m['contact_to_verified_grasp_seconds'] or 0.)/.5,
                'touch_force':audit['max_touch_force_n']/5. if audit['action']=='touch' else 0.}
    cost=1.5*components['jerk']+2.*components['high_frequency']+.5*components['duration']+.4*components['contact_latency']+.5*components['touch_force']
    return {'feasible':hard_pass,'score':(1000. if hard_pass else -1000.)-cost,
            'cost_components':components,'motion':m}


def compare(root,baseline):
    rows=[]
    for p in sorted(root.glob('*/*/*/audit.json')):
        relative=p.parent.relative_to(root);old_path=baseline/relative
        audit=json.loads(p.read_text());new=json.loads(p.with_name('trajectory.json').read_text())
        old=json.loads((old_path/'trajectory.json').read_text())
        rows.append({'scene':str(relative),'old':feedback_metrics(old),'new':feedback_metrics(new),
                     'selection':score_run(audit,new)})
    return rows
