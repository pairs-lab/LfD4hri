import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from VLA.pose_revalidated_acquisition_teacher import (
    PoseRevalidatedAcquisitionTeacher, grasp_plan_pose_changed)


class PoseRevalidatedAcquisitionTests(unittest.TestCase):
    def test_quaternion_sign_is_not_motion(self):
        self.assertFalse(grasp_plan_pose_changed([0, 0, 0], [0, 0, 0, 1],
                                                 [0, 0, 0], [0, 0, 0, -1]))

    def test_large_tipping_invalidates_old_support(self):
        self.assertTrue(grasp_plan_pose_changed([0, 0, 0], [0, 0, 0, 1],
            [0, 0, 0], [0, np.sin(.25), 0, np.cos(.25)]))

    def test_small_settling_does_not_replan(self):
        self.assertFalse(grasp_plan_pose_changed([0, 0, 0], [0, 0, 0, 1],
            [.001, 0, 0], [0, np.sin(.005), 0, np.cos(.005)]))

    def test_translation_invalidates_plan(self):
        self.assertTrue(grasp_plan_pose_changed([0, 0, 0], [0, 0, 0, 1],
            [.003, 0, 0], [0, 0, 0, 1]))

    def test_malformed_pose_rejected(self):
        for bad in ([0, 0, 0, 0], [0, 0, np.nan, 1], [0, 1]):
            with self.assertRaises(ValueError):
                grasp_plan_pose_changed([0, 0, 0], [0, 0, 0, 1], [0, 0, 0], bad)

    def teacher(self):
        teacher = object.__new__(PoseRevalidatedAcquisitionTeacher)
        teacher.task = SimpleNamespace(name="move")
        teacher.phase, teacher.travel_height = "descend", .22
        teacher.plan_pose = ([0, 0, 0], [0, 0, 0, 1])
        teacher.replan_events, teacher.retreat_ticks, teacher.pending_replan = [], 0, False
        teacher.env = SimpleNamespace(pick_id=3, object_ids={"box": 3},
            gripper=SimpleNamespace(grasped_object=None, grasp_center=lambda: np.array([0, -.5, .10])))
        return teacher

    @patch("VLA.pose_revalidated_acquisition_teacher.p.getAABB", return_value=([0, 0, 0], [0, 0, .08]))
    @patch("VLA.pose_revalidated_acquisition_teacher.p.getBasePositionAndOrientation",
           return_value=([.01, 0, 0], [0, 0, 0, 1]))
    def test_invalid_low_plan_first_retreats_vertical_open(self, *_):
        teacher = self.teacher()
        action = teacher.predict(None)
        np.testing.assert_array_equal(action[[0, 1, 3, 4, 5]], np.zeros(5))
        self.assertGreater(action[2], 0)
        self.assertLessEqual(float(action[2]), .02000001)
        self.assertEqual(action[6], -1)
        self.assertTrue(teacher.pending_replan)

    @patch("VLA.pose_revalidated_acquisition_teacher.p.getAABB", return_value=([0, 0, 0], [0, 0, .08]))
    @patch("VLA.pose_revalidated_acquisition_teacher.p.getBasePositionAndOrientation",
           return_value=([.01, 0, 0], [0, 0, 0, 1]))
    def test_bounded_retreat_preserves_failure(self, *_):
        teacher = self.teacher()
        teacher.retreat_ticks = teacher.MAX_RETREAT_TICKS
        with self.assertRaisesRegex(ValueError, "did not converge"):
            teacher.predict(None)


if __name__ == "__main__":
    unittest.main()
