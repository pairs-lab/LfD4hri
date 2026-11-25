"""OFFLINE ONLY: rotate above neighbours when neither open finger fits.

Finite, geometric candidates; no scene reroll, task relaxation or learned
fallback. Rotated link AABBs are conservative proposals, not physics proofs.
Keep the old teacher's successful zero-yaw proposal whenever it exists.
"""
import itertools
import numpy as np
import pybullet as p

from VLA.obstacle_aware_touch_teacher import (OpenVerifiedTouchTeacher,
    translated_vertical_sweep_clearance, touch_open_wait_action)
from VLA.staged_acquisition_teacher import (horizontal_first_action,
    XY_ALIGNMENT_M, YAW_ALIGNMENT_ACTION_RAD, INITIAL_VERTICAL_STEP_M)
from VLA.robust_touch_teacher import limit_near_contact_motion
from VLA.surface_limited_touch_teacher import surface_limited_motion


def rotated_aabb(low, high, centre, angle):
    low, high, centre = map(lambda v: np.asarray(v, float), (low, high, centre))
    if any(v.shape != (3,) for v in (low, high, centre)) or not np.isfinite([*low, *high, *centre, angle]).all():
        raise ValueError("Invalid rotated collision bounds")
    if np.any(low > high):
        raise ValueError("Inverted collision bounds")
    corners = np.array(list(itertools.product(*zip(low, high))))
    rotation = yaw_matrix(angle)
    rotated = (corners-centre)@rotation.T+centre
    return rotated.min(axis=0), rotated.max(axis=0)


def yaw_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])


def yaw_touch_proposals(link_boxes, obstacle_boxes, pad_vertices, centre, surface, travel_height):
    centre, surface = np.asarray(centre, float), np.asarray(surface, float)
    scored = []
    # Zero yaw is deliberately first. Nonzero yaw is used only if zero cannot
    # clear the neighbours, preserving the original teacher on prior scenes.
    for angle in (0., np.pi/2, -np.pi/2, np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4, np.pi):
        boxes = [rotated_aabb(low, high, centre, angle) for low, high in link_boxes]
        for index, vertices in enumerate(pad_vertices):
            vertices = (np.asarray(vertices)-centre)@yaw_matrix(angle).T+centre
            tip = vertices[vertices[:, 2] <= vertices[:, 2].min()+.001].mean(axis=0)
            target = surface-(tip-centre)
            if not (-.36 <= target[0] <= .36 and -.79 <= target[1] <= -.21):
                continue
            score = translated_vertical_sweep_clearance(boxes, obstacle_boxes,
                target-centre, travel_height-float(centre[2]))
            scored.append({"pad_index": index, "yaw_world_rad": angle,
                "target": target.tolist(), "sweep_aabb_clearance_m": score if np.isfinite(score) else None})
        safe = [row for row in scored if row["sweep_aabb_clearance_m"] is None
                or row["sweep_aabb_clearance_m"] > .004]
        if angle == 0 and safe:
            return scored, min(safe, key=lambda row: np.linalg.norm(np.asarray(row["target"])[:2]-centre[:2]))
    safe = [row for row in scored if row["sweep_aabb_clearance_m"] is None
            or row["sweep_aabb_clearance_m"] > .004]
    if not safe:
        raise ValueError("No neighbour-clear open-finger proposal in the declared yaw grid")
    return scored, min(safe, key=lambda row: (abs(row["yaw_world_rad"]),
        np.linalg.norm(np.asarray(row["target"])[:2]-centre[:2]), -float(row["sweep_aabb_clearance_m"] or 1e6)))


class YawClearTouchTeacher(OpenVerifiedTouchTeacher):
    def reset(self, task):
        super().reset(task)
        self.touch_jaw_angle = None

    def _touch_target(self):
        if self.touch_pad_index is None:
            position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
            surface = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)@self.touch_surface_local+position
            centre = self.env.gripper.grasp_center()
            links = [p.getAABB(self.env.gripper.body, link) for link in range(-1, p.getNumJoints(self.env.gripper.body))]
            obstacles = [p.getAABB(body) for body in self.env.object_ids.values() if body != self.env.pick_id]
            self.touch_proposals, selected = yaw_touch_proposals(links, obstacles,
                self.env.gripper.world_pad_vertices(), centre, surface, self.travel_height)
            self.touch_pad_index = selected["pad_index"]
            left, right = self.env.gripper.pad_positions()
            axis = right-left
            self.touch_jaw_angle = float(np.arctan2(axis[1], axis[0])+selected["yaw_world_rad"])
        return super()._touch_target()

    def _touch_yaw(self):
        left, right = self.env.gripper.pad_positions()
        axis = right-left
        error = (self.touch_jaw_angle-np.arctan2(axis[1], axis[0])+np.pi)%(2*np.pi)-np.pi
        pose = p.getLinkState(self.env.robot_id, self.env.robot_spec.mount_link, computeForwardKinematics=True)
        tool = np.asarray(p.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
        if abs(tool[2, 2]) < .9:
            raise ValueError("Touch yaw requires unchanged top-down orientation")
        return float(np.clip(error/tool[2, 2], -.08, .08))

    def predict(self, observation):
        if self.task.name != "touch":
            return super().predict(observation)
        if self.env._last_success:
            return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        centre = self.env.gripper.grasp_center()
        if self.phase == "approach":
            if self.touch_pad_index is None:
                wait = touch_open_wait_action(self.env.normalized_gripper_closure())
                if wait is not None:
                    return wait
            if not self.initial_hover_reached:
                dz = self.travel_height-float(centre[2])
                if abs(dz) >= .006:
                    return np.array([0, 0, np.clip(.8*dz, -INITIAL_VERTICAL_STEP_M, INITIAL_VERTICAL_STEP_M),
                                     0, 0, 0, -1], dtype=np.float32)
                self.initial_hover_reached = True
            target = self._touch_target()
            yaw = self._touch_yaw()
            if abs(yaw) >= YAW_ALIGNMENT_ACTION_RAD:
                # Rotate at the clear hover before translating or descending;
                # the physical selected pad retains its identity, including pi.
                return np.array([0, 0, 0, 0, 0, yaw, -1], dtype=np.float32)
            target[2] = max(float(target[2])+.14,
                max(p.getAABB(body)[1][2] for body in self.env.object_ids.values())+.04)
            action, self.horizontal_done = horizontal_first_action(centre, target, yaw,
                self.travel_height, self.horizontal_done)
            if self.horizontal_done and self._near(centre, target, .015):
                self._advance("descend")
            self.phase_steps += 1
            return action
        target, yaw = self._touch_target(), self._touch_yaw()
        action = np.r_[np.clip(.35*(target-centre), -.004, .004), 0, 0, yaw, -1].astype(np.float32)
        if np.linalg.norm(target[:2]-centre[:2]) >= XY_ALIGNMENT_M or abs(yaw) >= YAW_ALIGNMENT_ACTION_RAD:
            action[2] = max(float(action[2]), 0.)
        closest = [c for body in self.env.object_ids.values() for c in
                   p.getClosestPoints(self.env.gripper.body, body, distance=.010)]
        action = limit_near_contact_motion(action, min((c[8] for c in closest), default=None), "descend")
        return surface_limited_motion(action, closest)
