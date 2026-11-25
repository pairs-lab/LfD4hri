"""Experimental task-independent actuator governor, NOT a grasp controller.

Only measured finger joint lag is read. Existing forces, carrier criteria,
safety thresholds, robot/object assets and neural Cartesian commands remain.
Not deployed in any live evaluation before its independent qualification.
"""
import numpy as np
import pybullet as p

ACTIVATION_RAD=.08
TRACKING_MARGIN_RAD=.02


def limited_target(target, positions, signs, latched=False):
    q=np.asarray(positions,float);sign=np.asarray(signs,float)
    if q.shape!=(6,) or sign.shape!=(6,) or not np.isfinite(np.r_[q,target]).all():
        raise ValueError("Actual finite mimic joint measurements required")
    error=float(np.max(np.abs(q[1:]-sign[1:]*q[0])))
    if error<ACTIVATION_RAD:return float(target),False,error
    followers=q[1:]*sign[1:]
    # Brake the leading main actuator rather than continuing to pull a
    # blocked follower away from the mechanical coupling manifold.
    low=float(followers.max()-TRACKING_MARGIN_RAD);high=float(followers.min()+TRACKING_MARGIN_RAD)
    proposed=float(np.clip(target,low,high)) if low<=high else float((followers.min()+followers.max())/2)
    return float(np.clip(proposed,0.,.849)),abs(proposed-target)>1e-12,error


def install(gripper):
    if getattr(gripper,"_experimental_mimic_governor_installed",False):raise ValueError("Do not install twice")
    original=gripper.apply_motor_control
    gripper.mimic_governor_ticks=0;gripper.mimic_governor_max_before_error=0.
    def apply():
        q=[state[0] for state in p.getJointStates(gripper.body,gripper.JOINTS)]
        latched=gripper.grasp_constraint is not None and gripper._latched is not None
        requested=float(gripper._latched[0] if latched else gripper.target_angle)
        target,active,error=limited_target(requested,q,gripper.SIGNS,latched)
        gripper.mimic_governor_max_before_error=max(gripper.mimic_governor_max_before_error,error)
        if not active:return original()
        gripper.mimic_governor_ticks+=1
        before=gripper._latched.copy() if latched else gripper.command
        try:
            if latched:gripper._latched[0]=target
            else:gripper.command=2*(target-gripper.OPEN_ANGLE)/(gripper.CLOSED_ANGLE-gripper.OPEN_ANGLE)-1
            return original()
        finally:
            if latched:gripper._latched=before
            else:gripper.command=before
    gripper.apply_motor_control=apply;gripper._experimental_mimic_governor_installed=True
