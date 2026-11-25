"""Expert-only contact-margin proposal; robot/guards/success remain unchanged."""
import numpy as np
import pybullet as p
from VLA.geometry_clearance_teacher import GeometryClearanceTeacher


def limit_near_contact_motion(action, surface_distance, phase):
    action = np.asarray(action).copy()
    if phase == "descend" and surface_distance is not None and surface_distance <= .010:
        action[:3] = np.clip(action[:3], -.002, .002)
    return action


class RobustTouchTeacher(GeometryClearanceTeacher):
    def reset(self, task):
        super().reset(task)
        if task.name == "touch":
            _, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
            rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
            # A1mm nominal contact margin, NOT a floor/gripper penetration
            # override. Stop at the ORIGINAL first-hit success as before.
            self.touch_surface_local += rotation.T @ np.array([0, 0, -.001])

    def predict(self, observation):
        action = super().predict(observation)
        if self.task.name == "touch" and self.phase == "descend":
            # Slow only the FINAL centimetre, not the entire14cm descent:
            # a blanket2mm cap exhausted the original80-step budget.
            closest = p.getClosestPoints(self.env.gripper.body, self.env.pick_id, distance=.010)
            distance = min((contact[8] for contact in closest), default=None)
            action = limit_near_contact_motion(action, distance, self.phase)
        return action


class GentleForceTouchTeacher(RobustTouchTeacher):
    """Expert-only last-millimetre speed proposal, not learned control."""
    def predict(self, observation):
        action = super().predict(observation)
        if self.task.name == "touch" and self.phase == "descend":
            closest = p.getClosestPoints(self.env.gripper.body, self.env.pick_id, distance=.001)
            if closest:
                action[:3] = np.clip(action[:3], -.000040, .000040)
        return action
