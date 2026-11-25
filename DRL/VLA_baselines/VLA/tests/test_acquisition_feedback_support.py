import unittest
import numpy as np
from VLA.audit_acquisition_feedback_support import episode_support, stable_manifest


def fixture():
    return {"canonical_actions": np.zeros((3, 7)), "gripper_closure": np.zeros(3),
        "next_gripper_closure": np.zeros(3), "next_requested_gripper_closure": np.full(3, .5),
        "next_grasped_object": np.zeros(3, bool), "next_verified_grasp": np.zeros(3, bool),
        "joint_positions": np.zeros((3, 6)), "next_observed_joint_positions": np.zeros((3, 6))}


class AcquisitionFeedbackTest(unittest.TestCase):
    def test_live_manifest_progress_is_not_a_contract_change(self):
        old = {"robot": "ur5e", "episodes": 10, "frames": 100, "index_sha256": "a"}
        new = {"robot": "ur5e", "episodes": 11, "frames": 110, "index_sha256": "b"}
        self.assertEqual(stable_manifest(old, True), stable_manifest(new, True))
        self.assertNotEqual(stable_manifest(old, False), stable_manifest(new, False))
        self.assertNotEqual(stable_manifest(old, True), stable_manifest({**new, "robot": "other"}, True))

    def test_requested_is_not_measured_aperture(self):
        value = fixture()
        value["canonical_actions"][:, 6] = 1
        value["next_requested_gripper_closure"][:] = 1
        report = episode_support(value, "pick")
        self.assertFalse(report["errors"])
        self.assertEqual(report["requested_vs_measured_closure_max_gap"], 1)

    def test_shifted_sensor_feedback_is_rejected(self):
        value = fixture()
        value["next_observed_joint_positions"][0, 0] = .01
        self.assertIn("adjacent_measured_feedback_time_mismatch", episode_support(value, "touch")["errors"])

    def test_acquisition_transition_is_not_empty_lift(self):
        value = fixture()
        value["canonical_actions"][0, [2, 6]] = [.01, 1]
        value["next_requested_gripper_closure"][0] = 1
        value["next_grasped_object"][0] = True
        value["next_verified_grasp"][0] = True
        self.assertEqual(episode_support(value, "pick")["close_empty_lift_candidate_frames"], [])

    def test_empty_lift_is_preserved_for_diagnosis(self):
        value = fixture()
        value["canonical_actions"][0, [2, 6]] = [.01, 1]
        value["next_requested_gripper_closure"][0] = 1
        self.assertEqual(episode_support(value, "move")["close_empty_lift_candidate_frames"], [0])

    def test_hold_aperture_is_actual_sensor_not_requested_close(self):
        value = fixture()
        value["next_verified_grasp"][:] = True
        value["next_grasped_object"][:] = True
        value["gripper_closure"][:] = .4
        value["next_gripper_closure"][:] = .4
        report = episode_support(value, "pick")
        self.assertEqual(report["verified_before_measured_closure_range"], [.4, .4])

    def test_reopen_empty_uses_measured_before_state(self):
        value = fixture()
        value["gripper_closure"][:] = .95
        value["next_gripper_closure"][:] = .95
        value["canonical_actions"][:, 6] = -1
        value["next_requested_gripper_closure"][:] = 0
        self.assertEqual(episode_support(value, "pick")["reopen_empty_frames"], 3)


if __name__ == "__main__":
    unittest.main()
