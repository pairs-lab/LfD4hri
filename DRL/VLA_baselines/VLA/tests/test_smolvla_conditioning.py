from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from VLA.audit_smolvla_conditioning import controlled_prediction, effect, interventions
from VLA.common.observation import VLAObservation


class ConditioningTests(unittest.TestCase):
    def observation(self):
        return VLAObservation(np.ones((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 2, dtype=np.uint8),
                              np.zeros(6), .085, 1., np.zeros(3), np.zeros(6), "Pick the red apple", 0)

    def test_interventions_do_not_mutate_original_or_proprioception(self):
        obs = self.observation()
        variants = interventions(obs, "Pick the yellow banana")
        for variant in variants.values():
            self.assertIs(variant.joint_positions, obs.joint_positions)
            self.assertIs(variant.ee_position, obs.ee_position)
        self.assertTrue(np.all(obs.front_rgb == 1))
        self.assertTrue(np.all(obs.wrist_rgb == 2))
        self.assertTrue(np.all(variants["both_blackout"].wrist_rgb == 0))
        self.assertEqual(variants["alternate_target_text"].prompt, "Pick the yellow banana")

    def test_bad_alternative_is_rejected(self):
        for text in ("", " ", self.observation().prompt):
            with self.assertRaises(ValueError):
                interventions(self.observation(), text)

    def test_changed_text_still_uses_baseline_noise_reset(self):
        baseline = SimpleNamespace(prompt="Pick the red apple")
        class Policy:
            def reset(self, task): self.seed_prompt = task.prompt
            def predict(self, observation): return np.zeros(7)
        policy = Policy()
        sensor = SimpleNamespace()
        controlled_prediction(policy, baseline, replace(self.observation(), prompt="Pick the yellow banana"), sensor, .4)
        self.assertEqual(policy.seed_prompt, baseline.prompt)
        self.assertEqual(sensor.closure, .4)

    def test_invalid_predictions_fail_closed(self):
        class Policy:
            def reset(self, task): pass
            def predict(self, observation): return np.full(7, np.nan)
        with self.assertRaises(ValueError):
            controlled_prediction(Policy(), object(), self.observation(), SimpleNamespace(), .4)

    def test_effects_keep_translation_rotation_and_grip_units_separate(self):
        result = effect([3., 4., 0., 0., 0., 2., -1.], np.zeros(7))
        self.assertEqual(result, {"translation_l2_m": 5., "rotation_l2_rad": 2., "gripper_abs_difference": 1.})


if __name__ == "__main__":
    unittest.main()
