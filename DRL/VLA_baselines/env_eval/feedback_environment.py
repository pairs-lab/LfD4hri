"""Original robot/gripper assets with a stateful jerk-bounded action executor."""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from env_eval.legacy_gripper_environment import LegacyGripperEnvironment
from env_eval.feedback_motion import JointReference


class FeedbackEnvironment(LegacyGripperEnvironment):
    def reset(self,task):
        result=super().reset(task)
        self.reference=JointReference(self.q,1/self.physics_hz,getattr(self,'joint_jerk',35.))
        self.reference_acceleration=np.zeros(6)
        self.offset_filter=None;self.last_ik=self.q.copy()
        self.physics_commands=[]
        self.gripper.pregrasp_force_limit=4.
        self.gripper.pregrasp_object=self.pick_id
        self.preview_sampler=None
        return result

    def tick(self,center,quaternion,command):
        if command<-.2 and self.gripper.grasp_constraint is not None:
            self.gripper._release_assisted_grasp()
        if self.gripper.grasp_constraint is not None and self.gripper.grasp_constraint!=self.pressure_constraint:
            self.gripper._latched+=.006*self.gripper.SIGNS
            self.pressure_constraint=self.gripper.grasp_constraint
        current,qcur=self.pose()
        root,root_quaternion=p.getBasePositionAndOrientation(self.gripper.body)
        local=Rotation.from_quat(root_quaternion).inv().apply(self.gripper.grasp_center()-root)
        # Do not turn compliance of the separate flange constraint into a
        # positive feedback loop that drives the arm further into the object.
        raw=self.gripper.mount_offset+Rotation.from_quat(self.gripper.mount_rotation).apply(local)
        pad_local=np.vstack([Rotation.from_quat(root_quaternion).inv().apply(vertices-root)
                             for vertices in self.gripper.world_pad_vertices()])
        flange_pad=self.gripper.mount_offset+Rotation.from_quat(self.gripper.mount_rotation).apply(pad_local)
        # Mesh collision AABBs include Bullet's convex margin. Raw STL
        # vertices alone underestimate the lowest physical contact surface.
        vertices_bottom=min(v[:,2].min() for v in self.gripper.world_pad_vertices())
        collision_bottom=min(p.getAABB(self.gripper.body,link)[0][2] for link in self.gripper.PAD_LINKS)
        collision_padding=max(0.,vertices_bottom-collision_bottom)
        self.table_guard_correction=0.
        def guard(flange,orientation):
            flange=np.asarray(flange).copy()
            if self.task.task_id!=0:
                bottom=Rotation.from_quat(orientation).apply(flange_pad)[:,2].min()
                minimum=p.getAABB(self.table_id)[1][2]+.0015+collision_padding-bottom
                self.table_guard_correction=max(self.table_guard_correction,max(0.,minimum-flange[2]))
                flange[2]=max(flange[2],minimum)
            return flange
        if self.offset_filter is None:self.offset_filter=raw.copy()
        else:self.offset_filter+=(1-np.exp(-1/self.physics_hz/.025))*(raw-self.offset_filter)
        mount=np.asarray(center)-Rotation.from_quat(quaternion).apply(self.offset_filter)
        if self.touch_reference is not None:
            mount=self.touch_reference['mount']+np.asarray(center)-self.touch_reference['tcp']
        mount=guard(mount,quaternion)
        ik=np.array(p.calculateInverseKinematics(self.robot_id,self.robot_spec.mount_link,mount,
            targetOrientation=quaternion,maxNumIterations=100,residualThreshold=1e-7)[:6])
        ik=self.q+(ik-self.q+np.pi)%(2*np.pi)-np.pi
        self.last_ik=ik.copy()
        future=None
        if self.preview_sampler is not None:
            horizon=.10;epsilon=.01;targets=[]
            for elapsed in (horizon-epsilon,horizon,horizon+epsilon):
                point,orientation=self.preview_sampler(elapsed)
                flange=np.asarray(point)-Rotation.from_quat(orientation).apply(self.offset_filter)
                if self.touch_reference is not None:
                    flange=self.touch_reference['mount']+np.asarray(point)-self.touch_reference['tcp']
                flange=guard(flange,orientation)
                solution=np.array(p.calculateInverseKinematics(self.robot_id,self.robot_spec.mount_link,flange,
                    targetOrientation=orientation,maxNumIterations=100,residualThreshold=1e-7)[:6])
                targets.append(self.q+(solution-self.q+np.pi)%(2*np.pi)-np.pi)
            before,goal,after=targets
            future=(goal,(after-before)/(2*epsilon),(after-2*goal+before)/epsilon**2,horizon)
        self.q,self.q_velocity,self.reference_acceleration=self.reference.step(ik,future)
        p.setJointMotorControlArray(self.robot_id,self.robot_spec.arm_joint_ids,p.POSITION_CONTROL,
            targetPositions=self.q,targetVelocities=self.q_velocity,forces=[250]*6,
            positionGains=[.16]*6,velocityGains=[1.]*6)
        self.gripper.command=float(command)
        self.gripper.apply_motor_control();p.stepSimulation();self.gripper.post_step()
        self.max_mount_translation_error=self.gripper.max_mount_translation_error
        self.max_mount_rotation_error=self.gripper.max_mount_rotation_error
        self.physics_commands.append({'q':self.q.tolist(),'v':self.q_velocity.tolist(),
                                      'a':self.reference_acceleration.tolist()})
