"""OFFLINE ONLY: choose a stable touch finger with whole-gripper clearance.

The old nearest-finger proposal ignores the opposite finger hitting a neighbour.
A translated link-AABB sweep is a conservative proposal, NOT a physics proof.
No robot/controller/success changes and no learned-policy fallback.
"""
import numpy as np
import pybullet as p

from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher, XY_ALIGNMENT_M
from VLA.robust_touch_teacher import limit_near_contact_motion
from VLA.surface_limited_touch_teacher import surface_limited_motion


def box_xy_clearance(low, high, other_low, other_high):
    low, high, other_low, other_high = map(lambda x: np.asarray(x, float),
                                         (low, high, other_low, other_high))
    if any(x.shape != (3,) for x in (low, high, other_low, other_high)):
        raise ValueError("Expected 3D bounds")
    if not np.isfinite([low, high, other_low, other_high]).all():
        raise ValueError("Nonfinite collision bounds")
    if np.any(low > high) or np.any(other_low > other_high):
        raise ValueError("Inverted collision bounds")
    outside = np.maximum(np.maximum(other_low[:2]-high[:2], low[:2]-other_high[:2]), 0.)
    if np.any(outside):
        return float(np.linalg.norm(outside))
    depth = np.minimum(high[:2]-other_low[:2], other_high[:2]-low[:2])
    return -float(depth.min())


def translated_vertical_sweep_clearance(link_boxes, obstacle_boxes, delta, hover_height_delta):
    """Whole gripper, including the unused finger, along the vertical descent."""
    delta = np.asarray(delta, float)
    if delta.shape != (3,) or not np.isfinite([*delta, hover_height_delta]).all():
        raise ValueError("Invalid proposed translation")
    scores = []
    for low, high in link_boxes:
        low, high = np.asarray(low)+delta, np.asarray(high)+delta
        sweep_high = high.copy()
        sweep_high[2] += max(0., float(hover_height_delta)-float(delta[2]))
        for other_low, other_high in obstacle_boxes:
            if low[2] > other_high[2]+.004 or sweep_high[2] < other_low[2]-.004:
                continue
            scores.append(box_xy_clearance(low, high, other_low, other_high))
    return min(scores, default=float("inf"))


class ObstacleAwareTouchTeacher(StagedAcquisitionTeacher):
    def reset(self, task):
        super().reset(task)
        self.touch_pad_index = None
        self.touch_proposals = None

    def _touch_target(self):
        position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        surface = rotation@self.touch_surface_local+position
        centre = self.env.gripper.grasp_center()
        proposals = []
        for index, vertices in enumerate(self.env.gripper.world_pad_vertices()):
            tip = vertices[vertices[:, 2] <= vertices[:, 2].min()+.001].mean(axis=0)
            target = surface-(tip-centre)
            if -.36 <= target[0] <= .36 and -.79 <= target[1] <= -.21:
                proposals.append((index, target))
        if self.touch_pad_index is None:
            links = [p.getAABB(self.env.gripper.body, link) for link in
                     range(-1, p.getNumJoints(self.env.gripper.body))]
            obstacles = [p.getAABB(body) for body in self.env.object_ids.values()
                         if body != self.env.pick_id]
            scored = [(index, target, translated_vertical_sweep_clearance(
                links, obstacles, target-centre, self.travel_height-float(centre[2])))
                      for index, target in proposals]
            self.touch_proposals = [{"pad_index": index, "target": target.tolist(),
                                     "sweep_aabb_clearance_m": score if np.isfinite(score) else None}
                                    for index, target, score in scored]
            safe = [(index, target) for index, target, score in scored if score > .004]
            if not safe:
                raise ValueError("Neither touch finger has a neighbour-clear whole-gripper proposal")
            self.touch_pad_index = min(safe, key=lambda item: np.linalg.norm(item[1][:2]-centre[:2]))[0]
        matches = [target for index, target in proposals if index == self.touch_pad_index]
        if len(matches) != 1:
            raise ValueError("Selected physical touch finger left the unchanged workspace")
        return matches[0].copy()

    def predict(self, observation):
        if self.task.name != "touch" or self.phase == "approach":
            return super().predict(observation)
        if self.env._last_success:
            return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        centre, target = self.env.gripper.grasp_center(), self._touch_target()
        action = np.r_[np.clip(.35*(target-centre), -.004, .004), np.zeros(3), -1].astype(np.float32)
        if np.linalg.norm(target[:2]-centre[:2]) >= XY_ALIGNMENT_M:
            action[2] = max(float(action[2]), 0.)
        closest = [c for body in self.env.object_ids.values() for c in
                   p.getClosestPoints(self.env.gripper.body, body, distance=.010)]
        action = limit_near_contact_motion(action, min((c[8] for c in closest), default=None), "descend")
        return surface_limited_motion(action, closest)


def touch_open_wait_action(closure):
    """Use actual aperture feedback before caching an OPEN-finger footprint."""
    if not np.isfinite(closure) or not 0 <= closure <= 1:
        raise ValueError("Invalid measured gripper closure")
    if closure > .02:
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    return None


class OpenVerifiedTouchTeacher(ObstacleAwareTouchTeacher):
    def predict(self, observation):
        if self.task.name == "touch" and self.phase == "approach" and self.touch_pad_index is None:
            wait = touch_open_wait_action(self.env.normalized_gripper_closure())
            if wait is not None:
                return wait
        return super().predict(observation)
