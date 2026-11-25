"""OFFLINE-only higher clear carry proposal; no camera/controller change.

Purpose: test whether raising the held object before horizontal transport
avoids robot occlusion of physical destinations in the same two cameras.
Not a VLA fallback; acceptance needs actual visibility AND strict replay.
"""
import numpy as np
import pybullet as p

from VLA.pose_revalidated_acquisition_teacher import PoseRevalidatedAcquisitionTeacher


def raised_carry_height(object_z, object_bottom, highest_top, extra_clearance=.08):
    if not np.isfinite([object_z, object_bottom, highest_top, extra_clearance]).all() or extra_clearance < 0:
        raise ValueError("Invalid carry geometry")
    return float(object_z + max(0., highest_top + .04 - object_bottom) + extra_clearance)


class RaisedCarryVisibilityTeacher(PoseRevalidatedAcquisitionTeacher):
    def predict(self, observation):
        if self.phase == "transport" and self.task.name in ("move", "pick_place") and self.carry_stage is None:
            state = self.env.debug_state()
            if state["grasped"]:
                top = max(p.getAABB(body)[1][2] for body in self.env.object_ids.values())
                self.carry_height = raised_carry_height(state["pick_position"][2], state["pick_bottom"][2], top)
                self.carry_stage = "raise"
        return super().predict(observation)
