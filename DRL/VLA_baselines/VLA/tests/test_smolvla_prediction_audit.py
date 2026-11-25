import unittest

from VLA.smolvla.audit_checkpoint_predictions import summarize


class PredictionSummaryTest(unittest.TestCase):
    def test_exact_actions(self):
        action = [.01, -.02, .03, 0, 0, 0, -1]
        result = summarize([{"predicted": action, "expected": action}])
        self.assertEqual(result["translation_axis_rmse_m"], 0)
        self.assertEqual(result["gripper_sign_accuracy"], 1)
        self.assertAlmostEqual(result["translation_cosine_mean"], 1)

    def test_stationary_actions_have_no_direction(self):
        action = [0, 0, 0, 0, 0, 0, 1]
        result = summarize([{"predicted": action, "expected": action}])
        self.assertIsNone(result["translation_cosine_mean"])
        self.assertEqual(result["moving_direction_samples"], 0)

    def test_opposite_actions(self):
        result = summarize([{
            "predicted": [.06, 0, 0, 0, 0, 0, 1],
            "expected": [-.03, 0, 0, 0, 0, 0, -1],
        }])
        self.assertEqual(result["gripper_sign_accuracy"], 0)
        self.assertEqual(result["translation_would_clip_fraction"], 1)
        self.assertAlmostEqual(result["translation_cosine_mean"], -1)


if __name__ == "__main__":
    unittest.main()
