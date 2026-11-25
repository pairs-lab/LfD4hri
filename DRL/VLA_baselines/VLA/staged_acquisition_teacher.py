"""OFFLINE expert labels: align horizontally high, then descend slowly.

Addresses diagonal fast-descent demonstration bias, not learned localization.
Simulator poses are demonstration-only; never a deployed VLA safety/oracle
fallback. The original robot, assets, physics guards and success stay intact.
"""
import numpy as np
import pybullet as p

from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchTeacher


XY_ALIGNMENT_M = .012
YAW_ALIGNMENT_ACTION_RAD = .025
HIGH_TRAVEL_STEP_M = .030
DESCENT_STEP_M = .015
INITIAL_VERTICAL_STEP_M = .020


def horizontal_first_action(current, target, yaw, travel_height, horizontal_done):
    current, target = np.asarray(current, float), np.asarray(target, float)
    if current.shape != (3,) or target.shape != (3,) or not np.isfinite([*current, *target, yaw, travel_height]).all():
        raise ValueError("Invalid staged expert geometry")
    aligned = np.linalg.norm(target[:2]-current[:2]) < XY_ALIGNMENT_M and abs(yaw) < YAW_ALIGNMENT_ACTION_RAD
    horizontal_done = bool(horizontal_done or aligned)
    desired = target.copy()
    if not horizontal_done:
        desired[2] = max(float(travel_height), float(current[2]))
    delta = .8*(desired-current)
    delta[:2] = np.clip(delta[:2], -HIGH_TRAVEL_STEP_M, HIGH_TRAVEL_STEP_M)
    delta[2] = np.clip(delta[2], -DESCENT_STEP_M, DESCENT_STEP_M)
    if not aligned:
        # Even after alignment, correct a new XY/yaw deviation before more Z.
        delta[2] = max(float(delta[2]), 0.)
    return np.r_[delta, 0., 0., np.clip(yaw, -.08, .08), -1.].astype(np.float32), horizontal_done


class StagedAcquisitionTeacher(SurfaceLimitedTouchTeacher):
    def reset(self, task):
        super().reset(task)
        self.horizontal_done = False
        # The high overview/reset pose is not horizontally reachable across
        # the entire table. First lower VERTICALLY to a collision-clear hover,
        # never diagonally into a guessed object location. Then align XY/yaw.
        self.travel_height = max(.22,
            max(p.getAABB(body)[1][2] for body in self.env.object_ids.values())+.08)
        self.initial_hover_reached = False

    def _touch_target(self):
        position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        surface = rotation@self.touch_surface_local+position
        centre = self.env.gripper.grasp_center()
        targets = []
        for vertices in self.env.gripper.world_pad_vertices():
            tip = vertices[vertices[:, 2] <= vertices[:, 2].min()+.001].mean(axis=0)
            target = surface-(tip-centre)
            if -.36 <= target[0] <= .36 and -.79 <= target[1] <= -.21:
                targets.append(target)
        if not targets:
            raise ValueError("No safely reachable physical touch proposal")
        return min(targets, key=lambda point: np.linalg.norm(point[:2]-centre[:2])).copy()

    def predict(self, observation):
        if self.phase == "approach":
            centre = self.env.gripper.grasp_center()
            if not self.initial_hover_reached:
                dz = self.travel_height-float(centre[2])
                if abs(dz) >= .006:
                    return np.asarray([0., 0., np.clip(.8*dz, -INITIAL_VERTICAL_STEP_M, INITIAL_VERTICAL_STEP_M),
                                       0., 0., 0., -1.], dtype=np.float32)
                self.initial_hover_reached = True
            target, yaw = (self._touch_target(), 0.) if self.task.name == "touch" else self.target_and_rotation()
            target[2] = max(float(target[2])+.14,
                max(p.getAABB(body)[1][2] for body in self.env.object_ids.values())+.04)
            action, self.horizontal_done = horizontal_first_action(
                centre, target, yaw, self.travel_height, self.horizontal_done)
            if self.horizontal_done and self._near(centre, target, .015) and abs(yaw) < YAW_ALIGNMENT_ACTION_RAD:
                self._advance("descend")
            self.phase_steps += 1
            return action
        action = super().predict(observation)
        if self.phase == "descend":
            action[2] = max(float(action[2]), -DESCENT_STEP_M)
            target, yaw = (self._touch_target(), 0.) if self.task.name == "touch" else self.target_and_rotation()
            centre = self.env.gripper.grasp_center()
            if (np.linalg.norm(target[:2]-centre[:2]) >= XY_ALIGNMENT_M
                    or abs(yaw) >= YAW_ALIGNMENT_ACTION_RAD):
                action[2] = max(float(action[2]), 0.)
        return action
