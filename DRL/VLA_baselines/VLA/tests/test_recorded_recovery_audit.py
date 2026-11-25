import unittest
from types import SimpleNamespace

import numpy as np

from VLA.audit_recorded_safety_batch import summarize
from VLA.probe_wrong_object_recovery import WrongHeldOpeningProbe


class RecordedRecoveryTests(unittest.TestCase):
    def environment(self):
        gripper = SimpleNamespace(grasped_object=9, grasp_center=lambda: np.array([0., -.5, .1]))
        return SimpleNamespace(gripper=gripper, pick_id=3, measured_gripper_closure=lambda: .6)

    def test_wrong_hold_open_before_any_retreat_or_horizontal_motion(self):
        teacher = WrongHeldOpeningProbe(self.environment())
        teacher.reset(SimpleNamespace(name="pick"))
        action = teacher.predict(None)
        np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, -1])

    def test_actual_wrong_hold_required_not_empty_or_correct(self):
        for held in (None, 3):
            env = self.environment()
            env.gripper.grasped_object = held
            with self.assertRaises(ValueError):
                WrongHeldOpeningProbe(env).reset(SimpleNamespace(name="pick"))

    def test_unheld_retreat_slow_vertical_still_open(self):
        env = self.environment()
        teacher = WrongHeldOpeningProbe(env)
        teacher.reset(SimpleNamespace(name="pick"))
        env.gripper.grasped_object = None
        action = teacher.predict(None)
        self.assertGreater(action[2], 0)
        self.assertLessEqual(float(action[2]), .015000001)
        np.testing.assert_array_equal(action[[0, 1, 3, 4, 5, 6]], [0, 0, 0, 0, 0, -1])

    def test_unexpected_regrasp_and_nonconvergent_open_fail(self):
        env = self.environment()
        teacher = WrongHeldOpeningProbe(env)
        teacher.reset(SimpleNamespace(name="pick"))
        env.gripper.grasped_object = 4
        with self.assertRaises(ValueError): teacher.predict(None)
        env.gripper.grasped_object = 9
        teacher.open_steps = 40
        with self.assertRaises(ValueError): teacher.predict(None)

    def report(self):
        return {"passed_exact_recorded_replay": True, "task": "touch", "steps": 59,
            "maximum_replay_errors": {"joint_rad": 0},
            "feedback": [{"safety_violation": "mimic error", "pre_rollback_diagnostics": {
                "contacts": [{"normal_force_n": 140., "other_body_name": "orange orange", "is_requested_object": False},
                             {"normal_force_n": .001, "other_body_name": "green pear", "is_requested_object": True}]}}]}

    def test_loaded_contact_uses_recorded_newtons_and_does_not_count_tiny_contact(self):
        row = summarize(self.report())
        self.assertTrue(row["distractor_loaded_contact"])
        self.assertFalse(row["requested_body_loaded_contact"])
        self.assertEqual(row["loaded_contact_body_names"], ["orange orange"])

    def test_failed_reproduction_cannot_be_used_for_causal_signature(self):
        report = self.report()
        report["passed_exact_recorded_replay"] = False
        with self.assertRaises(ValueError): summarize(report)


if __name__ == "__main__":
    unittest.main()
