import unittest
from unittest.mock import patch
from dataclasses import replace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.privileged_prompt import augment_prompt, parse_feedback, PrivilegedPromptPolicy


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.state = np.array([.1, .2, .3, .4, .5, .6, .7, -.3, .1, 1.1, .8,
                               .03, -.04, -.1, .2, -.25, .04, 1], dtype=np.float32)

    def test_roundtrip_and_quantization(self):
        state = self.state.copy(); state[0] = .123456
        prompt = augment_prompt("Pick the red block", state)
        self.assertLessEqual(np.max(np.abs(parse_feedback(prompt) - state)), .000051)
        self.assertTrue(prompt.startswith("Pick the red block;"))
        self.assertNotIn("reward", prompt)
        self.assertNotIn("waypoint", prompt)

    def test_every_causal_field_affects_input(self):
        original = augment_prompt("Move the apple", self.state)
        for i in range(18):
            state = self.state.copy(); state[i] += .01 if i != 17 else -1
            self.assertNotEqual(original, augment_prompt("Move the apple", state))

    def test_bad_values_and_duplicates_rejected(self):
        for bad in ([1] * 17, [np.nan] * 18, [np.inf] * 18):
            with self.assertRaises(ValueError):
                augment_prompt("pick", bad)
        with self.assertRaises(ValueError):
            augment_prompt(augment_prompt("pick", self.state), self.state)

    def test_output_is_exact_native_output_not_a_controller(self):
        action = np.arange(7, dtype=np.float32)
        class Native:
            name = "openvla"
            def predict(self, obs):
                self.observation = obs
                return action
            def reset(self, task): pass
            def close(self): pass
        obs = VLAObservation(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8),
                             self.state[:6], .01, .7, np.zeros(3), np.zeros(6), "pick", 0)
        native = Native(); policy = PrivilegedPromptPolicy(object(), native)
        with patch("VLA.privileged_prompt.feedback_state", return_value=self.state):
            self.assertIs(policy.predict(obs), action)
        self.assertIs(native.observation.front_rgb, obs.front_rgb)
        self.assertIs(native.observation.joint_positions, obs.joint_positions)
        self.assertEqual(obs.prompt, "pick")
        np.testing.assert_array_equal(parse_feedback(native.observation.prompt), self.state)


if __name__ == "__main__":
    unittest.main()
