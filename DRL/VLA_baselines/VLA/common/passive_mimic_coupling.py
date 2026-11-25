"""Experimental passive coupling ONLY after excessive measured finger lag.

Same optional Robotiq gear relations/force bound as the existing controller.
No task, object identity, reward, pose proposal or neural command is read.
Not production qualified. Sources whose all-tick lag remains below0.08rad
never engage these extra constraints.
"""
import numpy as np
import pybullet as p
ACTIVATION_RAD=.08


def install(gripper):
    if getattr(gripper,"_passive_mimic_monitor_installed",False):raise ValueError("Install once")
    original=gripper.apply_motor_control
    gripper.mimic_governor_ticks=0;gripper.passive_mimic_gears=[]
    def apply():
        q=np.asarray([s[0] for s in p.getJointStates(gripper.body,gripper.JOINTS)])
        error=float(np.max(np.abs(q[1:]-gripper.SIGNS[1:]*q[0])))
        if error>=ACTIVATION_RAD and not gripper.passive_mimic_gears:
            for joint,sign in zip(gripper.JOINTS[1:],gripper.SIGNS[1:],strict=True):
                gear=p.createConstraint(gripper.body,gripper.JOINTS[0],gripper.body,joint,
                    jointType=p.JOINT_GEAR,jointAxis=[1,0,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
                p.changeConstraint(gear,gearRatio=float(-sign),maxForce=120.,erp=.8)
                gripper.passive_mimic_gears.append(gear)
        if gripper.passive_mimic_gears:gripper.mimic_governor_ticks+=1
        return original()
    gripper.apply_motor_control=apply;gripper._passive_mimic_monitor_installed=True
