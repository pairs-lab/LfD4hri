"""OFFLINE carry labels: side-view retrieval before goal-directed transport.

Raising alone does not uncover a far goal behind the vertical arm. A bounded
side waypoint, then travel along that clear side, retrieves the SAME cameras'
view. Simulator masks/poses are expert-only, never learned-policy inputs.
"""
import numpy as np
from VLA.raised_carry_visibility_teacher import RaisedCarryVisibilityTeacher
from VLA.common.spatial import FIXED_DESTINATION


def side_waypoints(obj, goal, carry_height):
    obj, goal = np.asarray(obj, float), np.asarray(goal, float)
    if obj.shape != (3,) or goal.shape != (3,) or not np.isfinite([*obj, *goal, carry_height]).all():
        raise ValueError("Invalid carry waypoint geometry")
    side_x = float(np.clip(obj[0] + (-.09 if obj[0] >= 0 else .09), -.28, .28))
    return [np.array([side_x, obj[1], carry_height]), np.array([side_x, goal[1], carry_height])]


class LateralVisibilityCarryTeacher(RaisedCarryVisibilityTeacher):
    def reset(self, task):
        super().reset(task)
        self.visibility_start_checked = False
        self.visibility_waypoints = []

    def predict(self, observation):
        if self.phase == "transport" and self.task.name in ("move", "pick_place") and self.carry_stage == "travel":
            state = self.env.debug_state()
            if not state["grasped"]:
                self.visibility_waypoints = []
                return super().predict(observation)
            if not self.visibility_start_checked:
                self.visibility_start_checked = True
                physical_goal = self.task.place_name not in (None, *FIXED_DESTINATION)
                if physical_goal and all(np.count_nonzero(self.env.target_mask(view, "place")) < 25
                                         for view in ("front", "wrist")):
                    self.visibility_waypoints = side_waypoints(state["pick_position"], state["goal_position"], self.carry_height)
            if self.visibility_waypoints:
                goal = self.visibility_waypoints[0]
                if self._near(state["pick_position"], goal, .012):
                    self.visibility_waypoints.pop(0)
                else:
                    self.phase_steps += 1
                    return self._move(state["grasp_center"], state["grasp_center"] + goal - state["pick_position"], 1)
        return super().predict(observation)
