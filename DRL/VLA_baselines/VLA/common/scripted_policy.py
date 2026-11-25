"""IK waypoint oracle used only to validate physics and generate demonstrations."""

from __future__ import annotations

import numpy as np

from .environment import ManipulationEnv
from .observation import VLAObservation
from .policy import Policy
from .tasks import TaskSpec


class ScriptedPolicy(Policy):
    name = "oracle"
    comparable = False
    input_modalities = ("privileged_simulator_state",)

    def __init__(self, env: ManipulationEnv) -> None:
        self.env = env
        self.task: TaskSpec | None = None
        self.phase = "approach"
        self.phase_steps = 0

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        self.phase = "approach"
        self.phase_steps = 0

    def predict(self, observation: VLAObservation) -> np.ndarray:
        del observation
        if self.task is None:
            raise RuntimeError("Policy reset(task) was not called")
        state = self.env.debug_state()
        obj = state["pick_position"]
        obj_center = state["pick_center"]
        initial_obj = state["initial_pick_position"]
        center = state["grasp_center"]
        grasped = bool(state["grasped"])
        place = state["place_position"]
        goal = state["goal_position"]

        gripper = -1.0
        target = obj_center + np.array((0.0, 0.0, 0.14))

        if self.task.name == "touch":
            if self.phase == "approach" and self._near(center, target, 0.025):
                self._advance("descend")
            if self.phase == "descend":
                target = state["pick_top"]
            return self._move(center, target, gripper)

        if self.phase == "approach":
            if self._near(center, target, 0.025):
                self._advance("descend")
        elif self.phase == "descend":
            target = obj_center + np.array((0.0, 0.0, 0.025))
            if self._near(center, target, 0.018):
                self._advance("close")
        elif self.phase == "close":
            target = center
            gripper = 1.0
            if grasped or self.phase_steps >= 15:
                self._advance("lift")
        elif self.phase == "lift":
            gripper = 1.0
            intermediate_obj_goal = initial_obj + np.array((0.0, 0.0, 0.11))
            target = center + (intermediate_obj_goal - obj)
            if self._near(obj, intermediate_obj_goal, 0.025):
                if self.task.name == "pick":
                    self._advance("transport")
                else:
                    self._advance("transport")
        elif self.phase == "transport":
            gripper = 1.0
            if self.task.name == "pick_place":
                # Travel above all 0--15 cm random_env objects, then descend
                # vertically.  A direct diagonal path can drag the grasped
                # object through a bowl/block that happens to sit between the
                # pick pose and a named corner; that is a demonstration-policy
                # bug, not evidence that a VLA cannot solve the task.
                hover_goal = goal.copy()
                hover_goal[2] = max(float(goal[2]) + 0.14, 0.18)
                target = center + (hover_goal - obj)
                xy_close = np.linalg.norm(obj[:2] - hover_goal[:2]) < 0.015
                if xy_close and abs(float(obj[2] - hover_goal[2])) < 0.025:
                    self._advance("lower")
            else:
                target = center + (goal - obj)
                if self._near(obj, goal, 0.012):
                    self._advance("hold")
        elif self.phase == "hold":
            # Hold the object at the exact reward.xml goal for the same ten
            # consecutive success steps required by individual_reward.py.
            gripper = 1.0
            target = center + (goal - obj)
        elif self.phase == "lower":
            gripper = 1.0
            target = center + (goal - obj)
            if self._near(obj, goal, 0.012):
                self._advance("hold")
        elif self.phase == "release":
            target = center
            gripper = -1.0
        else:
            raise RuntimeError(f"Unknown oracle phase {self.phase}")

        self.phase_steps += 1
        return self._move(center, target, gripper)

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0

    @staticmethod
    def _near(current: np.ndarray, target: np.ndarray, threshold: float) -> bool:
        return float(np.linalg.norm(current - target)) < threshold

    @staticmethod
    def _move(current: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        delta = np.clip((target - current) * 0.65, -0.045, 0.045)
        return np.concatenate((delta, np.zeros(3), np.array((gripper,)))).astype(np.float32)
