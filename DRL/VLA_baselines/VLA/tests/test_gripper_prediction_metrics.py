import unittest
import numpy as np
from VLA.smolvla.audit_checkpoint_predictions import gripper_diagnostic_fields, summarize


class GripperMetricTest(unittest.TestCase):
    def test_negative_sign_in_deadband_does_not_reopen_closed_hand(self):
        result = summarize([{"predicted": [0, 0, 0, 0, 0, 0, -.1],
            "expected": [0, 0, 0, 0, 0, 0, -1], "commanded_closure_before_action": 1}])
        self.assertEqual(result["gripper_sign_accuracy"], 1)
        self.assertEqual(result["gripper_applied_command_agreement"], 0)
        self.assertEqual(result["gripper_deadband_fraction"], 1)

    def test_deadband_correctly_retains_existing_closed_command(self):
        result = summarize([{"predicted": [0, 0, 0, 0, 0, 0, .1],
            "expected": [0, 0, 0, 0, 0, 0, 1], "commanded_closure_before_action": 1}])
        self.assertEqual(result["gripper_applied_command_agreement"], 1)

    def episode(self, measured, requested=None):
        episode = {"gripper_closure": np.array(measured),
                   "next_grasped_object": np.zeros(len(measured), dtype=bool)}
        if requested is not None:
            episode["next_requested_gripper_closure"] = np.array(requested)
        return episode

    def test_v9_deadband_retains_close_even_when_measured_aperture_is_partial(self):
        fields = gripper_diagnostic_fields(self.episode([0., .3], [1., 1.]), {}, 1, body_v9=True)
        self.assertEqual(fields["commanded_closure_before_action"], 1.)
        self.assertEqual(fields["measured_closure_before_action"], .3)
        row = {"predicted": [0, 0, 0, 0, 0, 0, -.1],
               "expected": [0, 0, 0, 0, 0, 0, 1], **fields}
        self.assertEqual(summarize([row])["gripper_applied_command_agreement"], 1)

    def test_v9_closed_empty_proxy_requires_fully_closed_feedback(self):
        episode = self.episode([.9, .95], [1., 1.])
        self.assertFalse(gripper_diagnostic_fields(episode, {}, 0, body_v9=True)["closed_empty_audit_only"])
        self.assertTrue(gripper_diagnostic_fields(episode, {}, 1, body_v9=True)["closed_empty_audit_only"])
        episode["next_grasped_object"][0] = True
        self.assertFalse(gripper_diagnostic_fields(episode, {}, 1, body_v9=True)["closed_empty_audit_only"])

    def test_v9_initial_command_defaults_open_independently_of_measurement(self):
        fields = gripper_diagnostic_fields(self.episode([.97], [0.]), {}, 0, body_v9=True)
        self.assertEqual(fields["commanded_closure_before_action"], 0.)

    def test_v9_initial_command_replays_startup_request_with_deadband(self):
        item = {"recovery_start_actions_not_training_labels": [
            [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -.1]]}
        episode = self.episode([.97], [0.])
        self.assertEqual(gripper_diagnostic_fields(episode, item, 0, body_v9=True)["commanded_closure_before_action"], 1.)
        item["recovery_start_actions_not_training_labels"].append([0, 0, 0, 0, 0, 0, -1])
        self.assertEqual(gripper_diagnostic_fields(episode, item, 0, body_v9=True)["commanded_closure_before_action"], 0.)

    def test_v9_missing_or_invalid_request_history_fails_closed(self):
        for episode in (self.episode([.3]), self.episode([.3], [np.nan]),
                        self.episode([.3], [2.]), self.episode([.3], [0., 1.])):
            with self.subTest(episode=episode), self.assertRaises(ValueError):
                gripper_diagnostic_fields(episode, {}, 0, body_v9=True)

    def test_v9_invalid_startup_action_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Invalid startup action"):
            gripper_diagnostic_fields(self.episode([.3], [0.]),
                {"recovery_start_actions_not_training_labels": [[1.]]}, 0, body_v9=True)

    def test_legacy_state_and_closed_empty_threshold_are_preserved(self):
        fields = gripper_diagnostic_fields(self.episode([.6], [0.]), {}, 0, body_v9=False)
        self.assertEqual(fields["commanded_closure_before_action"], .6)
        self.assertTrue(fields["closed_empty_audit_only"])
