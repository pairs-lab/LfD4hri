import json
from pathlib import Path
import unittest

import numpy as np

from VLA.new_vlas.privileged_inputs import append_numeric67
from VLA.stable_native_r44.conditioning import bounded_append, LIMIT


class BoundedObservationTest(unittest.TestCase):
    def setUp(self):
        root = Path('VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts')
        self.stats = json.loads((root/'stats.json').read_text())['scene67']
        row = json.loads((root/'index.json').read_text())[0]
        with np.load(root/row['cache']) as z:
            self.raw = z['scene64'][0].copy()

    def test_original_native_prefix_is_bit_exact_for_both_families(self):
        for width in (8, 20):
            prefix = np.linspace(-100., 100., width).astype(np.float32)
            value = bounded_append(prefix, self.raw, self.stats)
            np.testing.assert_array_equal(value[:width], prefix)
            self.assertEqual(value.shape, (width+67,))
            self.assertLessEqual(float(np.max(abs(value[width:]))), LIMIT)

    def test_all_in_range_values_unchanged_not_action_correction(self):
        prefix = np.ones(20, np.float32)
        original = append_numeric67(prefix, self.raw, self.stats)
        bounded = bounded_append(prefix, self.raw, self.stats)
        within = abs(original[20:]) <= LIMIT
        np.testing.assert_array_equal(original[20:][within], bounded[20:][within])
        self.assertEqual(self.raw[4], np.float32(self.raw[4]))

    def test_small_real_pose_error_cannot_become_500_sigma_input(self):
        self.raw[4] += .15
        original = append_numeric67(np.zeros(20), self.raw, self.stats)
        bounded = bounded_append(np.zeros(20), self.raw, self.stats)
        self.assertGreater(original[24], 400.)
        self.assertEqual(bounded[24], LIMIT)

    def test_wrong_body_feedback_slot_is_not_removed(self):
        self.raw[30], self.raw[31], self.raw[17] = 1., 1., 0.
        bounded = bounded_append(np.zeros(20), self.raw, self.stats)
        self.assertEqual(bounded[50], 1.)
        self.assertEqual(bounded[51], 1.)


if __name__ == '__main__':
    unittest.main()
