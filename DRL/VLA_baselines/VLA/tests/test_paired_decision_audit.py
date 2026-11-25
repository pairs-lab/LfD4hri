import unittest
import numpy as np
from VLA.paired_decision_audit import paired_decision_frame, SENSOR_FIELDS


class PairedDecisionAuditTests(unittest.TestCase):
    def pair(self):
        first = {key: np.zeros((8, 2)) for key in SENSOR_FIELDS}
        second = {key: value.copy() for key, value in first.items()}
        first["canonical_actions"] = np.zeros((8, 7))
        second["canonical_actions"] = np.zeros((8, 7))
        first["canonical_actions"][:3, 2] = -.02
        second["canonical_actions"][:3, 2] = -.02
        first["canonical_actions"][3:, 0] = .03
        second["canonical_actions"][3:, 0] = -.03
        return first, second

    def test_audits_target_decision_not_shared_vertical_reset(self):
        first, second = self.pair()
        result = paired_decision_frame(first, second)
        self.assertEqual(result["frame"], 3)
        self.assertEqual(result["identical_sensor_prefix_frames"], 4)
        self.assertTrue(result["informative_teacher_xy_separation"])

    def test_later_easy_xy_after_different_yaw_not_selected(self):
        first, second = self.pair()
        first["canonical_actions"][1, 5] = .08
        second["canonical_actions"][1, 5] = -.08
        second["front_rgb"][2] = 1
        result = paired_decision_frame(first, second, horizon=1)
        self.assertEqual(result["frame"], 1)
        self.assertFalse(result["informative_teacher_xy_separation"])
        self.assertEqual(result["first_difference_dimensions"], [5])

    def test_entire_sensor_prefix_not_only_chosen_frame_must_match(self):
        first, second = self.pair()
        second["gripper_closure"][0] = 1
        with self.assertRaisesRegex(ValueError, "common-prefix sensor gripper_closure"):
            paired_decision_frame(first, second)

    def test_identical_or_malformed_teacher_branches_rejected(self):
        first, second = self.pair()
        with self.assertRaisesRegex(ValueError, "No text-dependent"):
            paired_decision_frame(first, first)
        second["canonical_actions"][4, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Finite canonical"):
            paired_decision_frame(first, second)


if __name__ == "__main__":
    unittest.main()
