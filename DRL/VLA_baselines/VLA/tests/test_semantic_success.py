from __future__ import annotations

import unittest

import numpy as np
import pybullet as p

from VLA.benchmark import _normalized_semantic_gap
from VLA.common.environment import ManipulationEnv
from VLA.common.environment import SEMANTIC_SUCCESS_DEFAULTS
from VLA.common.tasks import TASKS


def move_object_bottom(env: ManipulationEnv, desired_bottom: np.ndarray) -> None:
    state = env.debug_state()
    center = np.asarray(state["pick_position"], dtype=np.float64)
    bottom = np.asarray(state["pick_bottom"], dtype=np.float64)
    orientation = p.getBasePositionAndOrientation(env.pick_id)[1]
    p.resetBasePositionAndOrientation(
        env.pick_id,
        (center + np.asarray(desired_bottom, dtype=np.float64) - bottom).tolist(),
        orientation,
    )


class SemanticSuccessTest(unittest.TestCase):
    def make_env(self) -> ManipulationEnv:
        return ManipulationEnv(
            image_size=32,
            required_success_steps=1,
            use_semantic_success=True,
        )

    def test_pick_is_lift_outcome_not_exact_waypoint(self) -> None:
        with self.make_env() as env:
            env.reset(TASKS["pick"])
            self.assertFalse(env.is_semantic_success())
            desired = env.initial_pick_bottom + np.array((0.0, 0.0, 0.051))
            move_object_bottom(env, desired)
            self.assertTrue(env.is_semantic_success())
            self.assertFalse(env.is_strict_success())
            self.assertGreater(env.object_goal_distance(), env.obj_threshold)

    def test_touch_distance_is_loose_but_bounded(self) -> None:
        with self.make_env() as env:
            env.reset(TASKS["touch"])
            original = env.ee_object_distance
            env.ee_object_distance = lambda: 0.04
            self.assertTrue(env.is_semantic_success())
            env.ee_object_distance = lambda: 0.0401
            self.assertFalse(env.is_semantic_success())
            env.ee_object_distance = original

    def test_move_accepts_correct_region_while_hovering(self) -> None:
        with self.make_env() as env:
            env.reset(TASKS["move"])
            place = np.asarray(env.debug_state()["place_position"], dtype=np.float64)
            move_object_bottom(env, place + np.array((0.075, 0.0, 0.12)))
            self.assertTrue(env.is_semantic_success())
            self.assertFalse(env.is_strict_success())
            self.assertGreater(env.object_goal_distance(), env.obj_threshold)

            move_object_bottom(env, place + np.array((0.081, 0.0, 0.12)))
            self.assertFalse(env.is_semantic_success())

            move_object_bottom(env, place + np.array((0.0, 0.0, 0.201)))
            self.assertFalse(env.is_semantic_success())

            move_object_bottom(env, place + np.array((0.0, 0.0, -0.031)))
            self.assertFalse(env.is_semantic_success())

    def test_pick_place_uses_target_region_not_two_centimetre_sphere(self) -> None:
        with self.make_env() as env:
            env.reset(TASKS["pick_place"])
            place = np.asarray(env.debug_state()["place_position"], dtype=np.float64)
            move_object_bottom(env, place + np.array((0.075, 0.0, 0.09)))
            self.assertTrue(env.is_semantic_success())
            self.assertFalse(env.is_strict_success())
            self.assertGreater(env.object_goal_distance(), env.obj_threshold)

            move_object_bottom(env, place + np.array((0.0, 0.0, 0.101)))
            self.assertFalse(env.is_semantic_success())

            move_object_bottom(env, place + np.array((0.0, 0.0, -0.061)))
            self.assertFalse(env.is_semantic_success())

    def test_tie_break_gap_combines_xy_and_height_from_one_frame(self) -> None:
        info = {
            "semantic_raw_success": False,
            "goal_xy_distance_m": 0.12,
            "goal_vertical_offset_m": 0.30,
        }
        gap = _normalized_semantic_gap("move", info, SEMANTIC_SUCCESS_DEFAULTS)
        expected = np.hypot(0.12 / 0.08 - 1.0, (0.30 - 0.20) / 0.20)
        self.assertAlmostEqual(gap, expected)

        info["semantic_raw_success"] = True
        self.assertEqual(_normalized_semantic_gap("move", info, SEMANTIC_SUCCESS_DEFAULTS), 0.0)


if __name__ == "__main__":
    unittest.main()
