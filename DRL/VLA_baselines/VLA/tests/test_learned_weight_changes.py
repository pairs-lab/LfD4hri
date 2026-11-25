import unittest
import numpy as np
from VLA.smolvla.audit_learned_weight_changes import difference_metrics


class LearnedWeightChangesTest(unittest.TestCase):
    def test_exact_unchanged_and_changed_weights(self):
        base = np.zeros((2, 3), dtype=np.float32)
        self.assertEqual(difference_metrics(base, base)["changed_values"], 0)
        learned = base.copy()
        learned[0, 0] = .01
        report = difference_metrics(base, learned)
        self.assertEqual(report["changed_values"], 1)
        self.assertAlmostEqual(report["changed_fraction"], 1/6)
        self.assertAlmostEqual(report["max_abs_change"], .01)

    def test_bad_shapes_empty_and_nonfinite_are_rejected(self):
        for a, b in ((np.zeros(2), np.zeros(3)), ([], []), ([float("nan")], [0]),
                     ([0], [float("inf")])):
            with self.assertRaises(ValueError):
                difference_metrics(a, b)
