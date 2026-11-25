import copy
import unittest

from VLA.analyze_diagnostic50 import analyze_trial


class FailureTraceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.result = {"name": "pick", "prompt": "Pick the red block up", "success": False, "steps": 1}
        self.trace = [{"step": 1, "action": [0, 0, 0, 0, 0, 0, -1],
            "interaction_after_audit_only": {"privileged_policy_input": False,
                "held_object": None, "correct_object_held": False,
                "pad_grasp_center_xyz_m": [.15, 0, .04], "object_center_xyz_m": [0, 0, .02]}}]

    def test_no_close_and_localization_offset_recorded_not_model_blame(self):
        r = analyze_trial(self.result, self.trace)
        self.assertEqual(set(r["failure_signatures"]), {
            "no_close_request_in_grasp_task", "never_held_requested_object", "low_approach_offset_over5cm"})
        self.assertAlmostEqual(r["min_xy_error_at_low_approach_m"], .15)

    def test_touch_does_not_require_grasp_close(self):
        self.result["name"] = "touch"
        self.assertEqual(analyze_trial(self.result, self.trace)["failure_signatures"], ["low_approach_offset_over5cm"])

    def test_wrong_object_is_not_empty_gripper(self):
        self.trace[0]["action"][6] = 1
        self.trace[0]["interaction_after_audit_only"]["held_object"] = "blue block"
        r = analyze_trial(self.result, self.trace)
        self.assertIn("held_wrong_object", r["failure_signatures"])
        self.assertNotIn("close_requested_without_hold_after_step", r["failure_signatures"])
        self.assertEqual(r["wrong_hold_steps"], 1)

    def test_success_does_not_receive_failure_classifications(self):
        self.result["success"] = True
        self.assertEqual(analyze_trial(self.result, self.trace)["failure_signatures"], [])

    def test_close_empty_lift_is_an_observed_signature(self):
        self.trace[0]["action"][2] = .01
        self.trace[0]["action"][6] = 1
        self.assertIn("close_and_lift_without_hold_after_step", analyze_trial(self.result, self.trace)["failure_signatures"])

    def test_partial_neural_close_uses_the_actual_controller_deadband(self):
        self.trace[0]["action"][6] = .3
        r = analyze_trial(self.result, self.trace)
        self.assertEqual(r["close_requests"], 1)
        self.assertEqual(r["canonical_close_request_threshold"], .2)
        self.assertNotIn("no_close_request_in_grasp_task", r["failure_signatures"])
        self.assertIn("close_requested_without_hold_after_step", r["failure_signatures"])
        self.trace[0]["action"][6] = .2
        self.assertEqual(analyze_trial(self.result, self.trace)["close_requests"], 0)

    def test_no_privileged_input_or_empty_trace_allowed(self):
        with self.assertRaises(ValueError):
            analyze_trial(self.result, [])
        t = copy.deepcopy(self.trace)
        t[0]["interaction_after_audit_only"]["privileged_policy_input"] = True
        with self.assertRaises(ValueError):
            analyze_trial(self.result, t)


if __name__ == "__main__":
    unittest.main()
