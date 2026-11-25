"""OFFLINE ONLY: invalidate a grasp plan when its object finishes tipping.

No reset, teleport, guard relaxation, learned-policy coordinates or fallback.
The preserved gelatin failure used a fixed body point/yaw planned while tilted;
rotation into a flat resting pose invalidated that footprint. Retreat vertically
with OPEN before replanning if the current tool is below the clear hover.
"""
import numpy as np
import pybullet as p

from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
from VLA.staged_acquisition_teacher import INITIAL_VERTICAL_STEP_M


def grasp_plan_pose_changed(reference_position, reference_quaternion, position, quaternion):
    values = [np.asarray(v, float) for v in
              (reference_position, reference_quaternion, position, quaternion)]
    if [v.shape for v in values] != [(3,), (4,), (3,), (4,)] or not all(np.isfinite(v).all() for v in values):
        raise ValueError("Finite position/quaternion required")
    first, second = values[1], values[3]
    if min(np.linalg.norm(first), np.linalg.norm(second)) < 1e-6:
        raise ValueError("Invalid quaternion")
    angle = 2 * np.arccos(np.clip(abs(np.dot(first / np.linalg.norm(first),
                                             second / np.linalg.norm(second))), 0, 1))
    return bool(np.linalg.norm(values[2] - values[0]) > .002 or angle > np.deg2rad(3))


class PoseRevalidatedAcquisitionTeacher(CentralSectionClearanceTeacher):
    MAX_REPLANS = 32
    MAX_RETREAT_TICKS = 35

    def reset(self, task):
        super().reset(task)
        self.plan_pose = p.getBasePositionAndOrientation(self.env.pick_id)
        self.replan_events = []
        self.retreat_ticks = 0
        self.pending_replan = False

    def predict(self, observation):
        if self.task.name != "touch" and self.phase in ("approach", "descend"):
            pose = p.getBasePositionAndOrientation(self.env.pick_id)
            changed = grasp_plan_pose_changed(*self.plan_pose, *pose)
            self.pending_replan = bool(self.pending_replan or changed)
            if self.pending_replan:
                if self.env.gripper.grasped_object is not None:
                    raise ValueError("Cannot replan an acquisition while holding an object")
                centre = self.env.gripper.grasp_center()
                clear_height = max(self.travel_height,
                    max(p.getAABB(body)[1][2] for body in self.env.object_ids.values()) + .08)
                if centre[2] < clear_height - .006:
                    self.retreat_ticks += 1
                    if self.retreat_ticks > self.MAX_RETREAT_TICKS:
                        raise ValueError("Bounded OPEN vertical plan-invalidation retreat did not converge")
                    return np.array([0, 0, min(INITIAL_VERTICAL_STEP_M, .8*(clear_height-centre[2])),
                                     0, 0, 0, -1], np.float32)
                if len(self.replan_events) >= self.MAX_REPLANS:
                    raise ValueError("Object never stabilized within the bounded offline replan budget")
                events = self.replan_events
                old_pose = self.plan_pose
                # Reset the EXPERT plan/phase only; NEVER reset the simulator.
                super().reset(self.task)
                self.plan_pose = p.getBasePositionAndOrientation(self.env.pick_id)
                events.append({"step": self.env.step_count, "old_pose": old_pose,
                               "new_pose": self.plan_pose, "new_plan": dict(self.plan)})
                self.replan_events = events
                self.retreat_ticks, self.pending_replan = 0, False
        return super().predict(observation)
