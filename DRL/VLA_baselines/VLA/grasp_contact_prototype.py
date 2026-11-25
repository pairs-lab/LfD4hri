"""AUDIT-ONLY body-grasp prototype. Not a production VLA controller/profile."""
import numpy as np
import pybullet as p

from VLA.audit_gripper_geometry import transform_points
from VLA.recovery_teacher import RecoveryTeacher


def install_contact_prototype(env, vertices, freeze_contact_joints=False):
    gripper = env.gripper
    def collision_centre():
        points = []
        for link in gripper.PAD_LINKS:
            state = p.getLinkState(gripper.body, link, computeForwardKinematics=True)
            points.append(transform_points(vertices[link, "collision"], state[4], state[5]).mean(axis=0))
        return np.mean(points, axis=0).astype(np.float32)

    def bilateral_candidate():
        candidates = []
        for body in gripper.object_ids:
            contacts = p.getContactPoints(bodyA=gripper.body, bodyB=body)
            contacted = {c[3] for c in contacts if c[9] > .1}
            if not set(gripper.PAD_LINKS) <= contacted:
                continue
            lo, hi = map(np.asarray, p.getAABB(body))
            centre = collision_centre()
            fraction = (centre[2] - lo[2]) / (hi[2] - lo[2])
            if not .25 <= fraction <= .80:
                continue
            candidates.append((float(np.linalg.norm(centre - (lo+hi)/2)), body))
        return min(candidates)[1] if candidates else None
    gripper.grasp_center = collision_centre
    gripper._candidate_between_fingers = bilateral_candidate
    if freeze_contact_joints:
        original_create = gripper._create_assisted_grasp
        original_motor_control = gripper.apply_motor_control
        joint_ids = (1, 3, 5, 6, 8, 10)
        latched = [None]
        def create(body):
            latched[0] = [p.getJointState(gripper.body, i)[0] for i in joint_ids]
            original_create(body)
        def motor_control():
            if gripper.grasp_constraint is None or latched[0] is None:
                original_motor_control()
            else:
                # Do not keep driving fingers through an object tethered to
                # the gripper base. This is a separate audit experiment.
                p.setJointMotorControlArray(gripper.body, joint_ids, p.POSITION_CONTROL,
                    targetPositions=latched[0], forces=(120.,) * 6, positionGains=(.8,) * 6)
        gripper._create_assisted_grasp = create
        gripper.apply_motor_control = motor_control


class MidbodyContactTeacher(RecoveryTeacher):
    def predict(self, observation):
        if self.task.name != "touch" and self.phase in ("descend", "close"):
            state = self.env.debug_state()
            if self.phase == "close" and not state["grasped"] and self.phase_steps >= 15:
                return super().predict(observation)
            target = state["pick_center"].copy()
            # Keep 35mm pads above the table. No object/pose modification.
            target[2] = max(float(target[2]), p.getAABB(self.env.table_id)[1][2] + .0205)
            gripper_command = -1 if self.phase == "descend" else 1
            if self.phase == "descend" and self._near(state["grasp_center"], target, .004):
                self._advance("close")
            elif self.phase == "close" and state["grasped"]:
                self._advance("lift")
            self.phase_steps += 1
            # Recenter during closure: finger kinematics move the physical pad
            # centre even if the wrist is stationary.
            return self._move(state["grasp_center"], target, gripper_command)
        return super().predict(observation)
