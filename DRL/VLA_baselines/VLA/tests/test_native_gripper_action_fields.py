import unittest
import numpy as np
from VLA.common.native_action_fields import native_gripper_closure, copy_requested_gripper_field


class NativeGripperActionFieldTest(unittest.TestCase):
    def test_v9_missing_command_cannot_silently_use_feedback(self):
        with self.assertRaisesRegex(ValueError, "V9 feedback"):
            native_gripper_closure({"next_gripper_closure": np.array([.2]),
                                    "next_verified_grasp": np.array([False])})
    def test_legacy_labels_unchanged(self):
        legacy = {"next_gripper_closure": np.array([0., 1.])}
        np.testing.assert_array_equal(native_gripper_closure(legacy), [0, 1])

    def test_subset_materialization_cannot_drop_v9_missing_command_guard(self):
        archive = {"next_gripper_closure": np.array([.2]),
                   "next_verified_grasp": np.array([False])}
        subset = {"next_gripper_closure": archive["next_gripper_closure"]}
        with self.assertRaisesRegex(ValueError, "V9 feedback"):
            copy_requested_gripper_field(archive, subset)

    def test_subset_materialization_rejects_nonfinite_v9_command(self):
        archive = {"next_gripper_closure": np.array([.2]),
                   "next_bilateral_pad_contact": np.array([False]),
                   "next_requested_gripper_closure": np.array([np.nan])}
        subset = {"next_gripper_closure": archive["next_gripper_closure"]}
        with self.assertRaisesRegex(ValueError, "finite"):
            copy_requested_gripper_field(archive, subset)

    def test_feedback_cannot_replace_requested_close(self):
        source = {"next_gripper_closure": np.array([.2, .6]),
                  "next_requested_gripper_closure": np.array([0., 1.])}
        materialized = {"next_gripper_closure": source["next_gripper_closure"]}
        copy_requested_gripper_field(source, materialized)
        np.testing.assert_array_equal(native_gripper_closure(materialized), [0, 1])
        np.testing.assert_array_equal(materialized["next_gripper_closure"], [.2, .6])
