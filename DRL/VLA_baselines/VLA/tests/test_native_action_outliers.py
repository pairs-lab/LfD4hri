import unittest
import numpy as np
from VLA.audit_native_action_outliers import error_metrics


class NativeActionOutlierTests(unittest.TestCase):
    def test_zero_and_signed_tail(self):
        zero = np.zeros(7)
        self.assertEqual(error_metrics(zero, zero)["translation_max_abs_m"], 0)
        actual = zero.copy()
        actual[0], actual[5], actual[6] = -.031, -.04, 2
        result = error_metrics(actual, zero)
        self.assertEqual(result["translation_max_abs_m"], .031)
        self.assertEqual(result["rotation_max_abs_rad"], .04)
        self.assertEqual(result["gripper_abs_error"], 2)

    def test_l2_does_not_hide_multiple_axis_errors(self):
        actual = np.r_[np.ones(3)*.01, np.zeros(4)]
        result = error_metrics(actual, np.zeros(7))
        self.assertAlmostEqual(result["translation_l2_m"], np.sqrt(3)*.01)

    def test_invalid_vectors_fail(self):
        for actual in (np.zeros(6), np.ones((1, 7)), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                error_metrics(actual, np.zeros(7))


if __name__ == "__main__":
    unittest.main()
