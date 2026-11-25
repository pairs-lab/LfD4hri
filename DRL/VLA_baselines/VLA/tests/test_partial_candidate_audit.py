import unittest
import numpy as np
from VLA.audit_partial_candidate import audit_arrays, native_preview


class PartialCandidateAuditTest(unittest.TestCase):
    def test_native_preprocessor_returns_list_of_backbone_triples(self):
        rgb, mask = np.zeros((512, 512, 3), np.uint8), np.ones((512, 512), bool)
        convert = lambda image, segmentation: [("smol", rgb, mask)]
        actual_rgb, actual_mask = native_preview(convert, None, None)
        self.assertIs(actual_rgb, rgb)
        self.assertIs(actual_mask, mask)
        with self.assertRaises(ValueError):
            native_preview(lambda a, b: [], None, None)

    def sample(self):
        n = 2
        arrays = {key: np.zeros((n, 384, 384, 3), np.uint8) for key in ("front_rgb", "wrist_rgb")}
        arrays.update({key: np.zeros((n, 6)) for key in ("joint_positions", "next_joint_targets", "next_observed_joint_positions")})
        arrays.update({key: np.zeros(n) for key in ("gripper_closure", "next_gripper_closure", "next_requested_gripper_closure", "next_verified_grasp")})
        arrays["canonical_actions"] = np.zeros((n, 7))
        arrays["ee_position"] = np.zeros((n, 3))
        arrays["ee_rotation_6d"] = np.tile([1., 0., 0., 0., 1., 0.], (n, 1))
        arrays["gripper_open"] = np.ones(n)
        arrays["canonical_actions"][:, 6] = -1
        arrays["prompt"] = np.asarray("Touch the red apple")
        arrays.update({f"{cam}_{role}_mask": np.ones((n, 384, 384), bool)
                       for cam in ("front", "wrist") for role in ("pick", "place", "robot")})
        row = {"steps": n, "task": {"prompt": str(arrays["prompt"])}, "success": True}
        return arrays, row

    def test_valid_aligned_sample(self):
        arrays, row = self.sample()
        self.assertEqual(audit_arrays(arrays, row)["errors"], [])

    def test_invalid_tcp_and_gripper_feedback_are_not_ignored(self):
        arrays, row = self.sample()
        arrays["ee_position"][0, 0] = np.nan
        arrays["ee_rotation_6d"][0, :3] = 0
        arrays["gripper_open"][0] = .2
        errors = audit_arrays(arrays, row)["errors"]
        self.assertIn("ee_position: wrong shape/nonfinite", errors)
        self.assertIn("TCP rotation6d is not an orthonormal column pair", errors)
        self.assertIn("open feedback is not the complement of measured closure", errors)

    def test_misalignment_and_requested_intent_are_not_ignored(self):
        arrays, row = self.sample()
        arrays["joint_positions"][1, 0] = .1
        arrays["next_requested_gripper_closure"][0] = 1
        result = audit_arrays(arrays, row)
        self.assertIn("joint before/after time alignment", result["errors"])
        self.assertIn("requested close intent differs from canonical label", result["errors"])

    def test_masks_only_report_visibility_not_dataset_qualification(self):
        arrays, row = self.sample()
        arrays["front_pick_mask"][:] = False
        arrays["wrist_pick_mask"][:] = False
        result = audit_arrays(arrays, row)
        self.assertEqual(result["visibility"]["pick"]["both_views_invisible_frames"], [0, 1])

    def test_virtual_region_has_no_segmentation_body_and_is_not_marked_blind(self):
        arrays, row = self.sample()
        row["task"]["place_name"] = "bottom right corner"
        arrays["front_place_mask"][:] = False
        arrays["wrist_place_mask"][:] = False
        result = audit_arrays(arrays, row)["visibility"]["place"]
        self.assertFalse(result["mask_applicable"])
        self.assertIsNone(result["either_view_visible_fraction"])
        self.assertEqual(result["both_views_invisible_frames"], [])


if __name__ == "__main__":
    unittest.main()
