import copy
import unittest

import numpy as np

from VLA.audit_training_readiness import CLOSED_HOVER, reset_example_role


class RecoveryRoleTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "variant": 0, "task": {"name": "pick", "prompt": "Pick the blue block up"},
            "recovery_start_kind": CLOSED_HOVER,
            "recovery_start_actions_not_training_labels": [[0, 0, 0, 0, 0, 0, 1]],
            "closed_hover_recovery_checks": {key: True for key in (
                "initial_measured_closed_empty", "first_corrective_label_open",
                "first_post_action_not_false_grasp", "terminal_verified_grasp")},
            "recovery_start_evidence": {"kind": CLOSED_HOVER, "actual_closure": .97,
                "grasped_object": None, "contact_count": 0, "surface_clearance_m": .1,
                "object_clearance_m": .1, "object_poses_teleported": False,
                "startup_controls_are_training_labels": False,
                "near_body_or_natural_failure_recovery_claimed": False},
        }
        self.reset = {"canonical_source": "source", "paired_scene_source": "scene.npz"}
        self.cache = {"source": ("hash", {"scene.npz": {"task": copy.deepcopy(self.item["task"])}})}
        self.data = {"gripper_closure": np.array([.97]),
                     "canonical_actions": np.array([[0, 0, 0, 0, 0, 0, -1]]),
                     "next_grasped_object": np.array([False]),
                     "next_verified_grasp": np.array([True])}

    def role(self):
        return reset_example_role(self.item, self.reset, self.cache, self.data)

    def test_actual_standalone_recovery_is_not_a_text_pair(self):
        self.assertEqual(self.role(), "closed_hover_recovery")

    def test_raw_and_near_body_pairs_keep_distinct_checks(self):
        self.assertEqual(reset_example_role({}, {}, {}, {}), "raw")
        self.item["recovery_start_kind"] = "closed_empty_near_object_missed_close"
        self.assertEqual(self.role(), "counterfactual_pair")
        self.item.pop("recovery_start_kind")
        self.assertEqual(self.role(), "counterfactual_pair")

    def test_no_provenance_or_retargeted_instruction_cannot_exempt_pair(self):
        with self.assertRaises(ValueError):
            reset_example_role(self.item, {}, self.cache, self.data)
        self.item["task"]["prompt"] = "Pick the red block up"
        with self.assertRaises(ValueError):
            self.role()

    def test_missing_or_false_collection_check_is_rejected(self):
        for value in (False, 1, None):
            self.item["closed_hover_recovery_checks"]["first_corrective_label_open"] = value
            with self.assertRaises(ValueError):
                self.role()

    def test_measured_empty_and_open_feedback_not_metadata_only(self):
        for key, bad in (("gripper_closure", np.array([.2])),
                         ("canonical_actions", np.array([[0, 0, 0, 0, 0, 0, 1]])),
                         ("next_grasped_object", np.array([True])),
                         ("next_verified_grasp", np.array([False]))):
            good = self.data[key]
            self.data[key] = bad
            with self.assertRaises(ValueError):
                self.role()
            self.data[key] = good

    def test_startup_missing_nonfinite_or_sideways_is_rejected(self):
        for controls in ([], [[0] * 6], [[float("nan"), 0, 0, 0, 0, 0, 1]],
                         [[.02, 0, 0, 0, 0, 0, 1]]):
            self.item["recovery_start_actions_not_training_labels"] = controls
            with self.assertRaises(ValueError):
                self.role()

    def test_contact_teleport_and_clearance_are_not_bypassed(self):
        for key, bad in (("contact_count", 1), ("object_poses_teleported", True),
                         ("object_clearance_m", .001), ("actual_closure", .99)):
            good = self.item["recovery_start_evidence"][key]
            self.item["recovery_start_evidence"][key] = bad
            with self.assertRaises(ValueError):
                self.role()
            self.item["recovery_start_evidence"][key] = good


if __name__ == "__main__":
    unittest.main()
