"""STAGED Oracle wrong-surface-height fault; native physics guard stays ON.

This is not evidence of a learned RL failure. Only the faulty controller's
estimated surface height/waypoint changes. No external forces, object poses,
motor parameters, safety tolerances, solver, collision filters or assets change.
"""
import argparse
from pathlib import Path
from types import MethodType
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from VLA.qualitative_r57 import oracle as base
from env_eval.oracle_trajectory import PoseCurve


def wrong_surface_tick(self,center,quaternion,command):
    """Existing Oracle servo with an explicitly wrong assumed surface Z.

    Native JointReference jerk/velocity/acceleration limits and the physical
    post_step guard remain unchanged. Unlike the normal servo, this faulty
    controller assumes the floor is 12 cm lower than it really is. It has no
    preview during the faulty straight-down segment.
    """
    _,root_quaternion=p.getBasePositionAndOrientation(self.gripper.body)
    root=np.array(p.getBasePositionAndOrientation(self.gripper.body)[0])
    local=Rotation.from_quat(root_quaternion).inv().apply(self.gripper.grasp_center()-root)
    raw=self.gripper.mount_offset+Rotation.from_quat(self.gripper.mount_rotation).apply(local)
    if self.offset_filter is None:self.offset_filter=raw.copy()
    else:self.offset_filter+=(1-np.exp(-1/self.physics_hz/.025))*(raw-self.offset_filter)
    mount=np.asarray(center)-Rotation.from_quat(quaternion).apply(self.offset_filter)
    pad_local=np.vstack([Rotation.from_quat(root_quaternion).inv().apply(v-root)
        for v in self.gripper.world_pad_vertices()])
    flange_pad=self.gripper.mount_offset+Rotation.from_quat(self.gripper.mount_rotation).apply(pad_local)
    bottom=Rotation.from_quat(quaternion).apply(flange_pad)[:,2].min()
    # Fault is in controller's ground estimate, NOT PyBullet ground geometry.
    mount[2]=max(mount[2],-.123+.0015-bottom)
    ik=np.array(p.calculateInverseKinematics(self.robot_id,self.robot_spec.mount_link,mount,
        targetOrientation=quaternion,maxNumIterations=100,residualThreshold=1e-7)[:6])
    ik=self.q+(ik-self.q+np.pi)%(2*np.pi)-np.pi
    self.last_ik=ik.copy()
    self.q,self.q_velocity,self.reference_acceleration=self.reference.step(ik,None)
    p.setJointMotorControlArray(self.robot_id,self.robot_spec.arm_joint_ids,p.POSITION_CONTROL,
        targetPositions=self.q,targetVelocities=self.q_velocity,forces=[250]*6,
        positionGains=[.16]*6,velocityGains=[1.]*6)
    self.gripper.command=float(command)
    self.gripper.apply_motor_control();p.stepSimulation();self.gripper.post_step()
    self.physics_commands.append({'q':self.q.tolist(),'v':self.q_velocity.tolist(),
        'a':self.reference_acceleration.tolist()})


class GroundSlamOracle(base.ScriptedOracle):
    launch_delay_ticks=0
    site=(.51,-.40)

    def run(self):
        e=self.env
        center=e.gripper.grasp_center();_,q=e.pose()
        top=np.array([*self.site,.34]);q1=np.array(p.getQuaternionFromEuler([0,np.pi,0]))
        self.fault_events.append({'type':'STAGED_wrong_ground_height_controller',
            'actual_floor_top_m':-.003,'assumed_floor_top_m':-.123,
            'site_xy_m':list(self.site),'not_learned_RL_failure':True,
            'physics_guard_unchanged':True,'same_motor_forces_gains_joint_limits':True})
        self.curve(PoseCurve([center,center+[0,0,.05],top+[0,0,.08],top],q,q1,3.0),
            'staged_miss_approach',-1.,1.)
        self.hold(top,q1,.5+self.launch_delay_ticks/e.physics_hz,'pre_impact',1.)
        e.preview_sampler=None
        e.tick=MethodType(wrong_surface_tick,e)
        self.phase='STAGED_fast_ground_descent'
        target=np.array([*self.site,-.105])
        for _ in range(round(3*e.physics_hz)):
            self.step(target,q1,1.)
        return {'success':False,'note':'No verified placement; staged faulty ground approach'}


if __name__=='__main__':
    p1=argparse.ArgumentParser(description=__doc__)
    p1.add_argument('--seed',type=int,default=20269719)
    p1.add_argument('--output',type=Path,required=True)
    p1.add_argument('--paired-capture',type=Path,required=True)
    p1.add_argument('--delay-ticks',type=int,default=0)
    p1.add_argument('--site-x',type=float,default=.51)
    p1.add_argument('--site-y',type=float,default=-.40)
    a=p1.parse_args();a.fault='STAGED_wrong_ground_height'
    GroundSlamOracle.launch_delay_ticks=a.delay_ticks;GroundSlamOracle.site=(a.site_x,a.site_y)
    base.ScriptedOracle=GroundSlamOracle
    base.main(a)
