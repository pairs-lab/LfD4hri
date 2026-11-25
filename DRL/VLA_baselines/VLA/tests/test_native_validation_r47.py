"""CPU tests of diagnostic plumbing; test doubles are NOT model results."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from VLA.fit_xvla_validation_r47 import monitor_rows
from VLA.new_vlas.action_contracts import pack_native20


class HeldoutMonitorTest(unittest.TestCase):
    def setUp(self):
        root = Path('VLA/audits/new_vlas_quality_r44_20261016/heldout_validation8_native_fixture')
        self.item = json.loads((root/'requests_with_LOCAL_gold.json').read_text())[3]
        self.stats = json.loads(Path('VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts/stats.json').read_text())
        self.images = {'front': np.zeros((16, 16, 3), np.uint8), 'wrist': np.zeros((16, 16, 3), np.uint8)}
        self.prediction = np.array(self.item['native_gold_LOCAL_ONLY'], np.float32)

    def run_monitor(self, prediction=None):
        from types import SimpleNamespace
        def predict(sample):
            self.seen = sample
            return self.prediction if prediction is None else prediction
        return monitor_rows(SimpleNamespace(predict=predict), self.stats, [self.item], [self.images])

    def test_neural_sample_contains_no_gold_future_phase_or_task_success(self):
        self.run_monitor()
        self.assertEqual(set(self.seen), {'front', 'wrist', 'instruction', 'state'})
        self.assertEqual(self.seen['instruction'], self.item['payload']['instruction'])
        self.assertEqual(self.seen['state'].shape, (87,))
        self.assertLessEqual(float(np.max(abs(self.seen['state'][20:]))), 5.)
        p = self.item['payload']
        prefix = pack_native20(np.asarray(p['ee_position']), np.asarray(p['ee_rotation_6d']), np.asarray(p['scene64'][6]))
        np.testing.assert_array_equal(self.seen['state'][:20], prefix)

    def test_future_gold_does_not_change_inference_inputs(self):
        self.run_monitor()
        before = copy.deepcopy(self.seen)
        self.item['native_gold_LOCAL_ONLY'] = (self.prediction+.001).tolist()
        self.run_monitor()
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(before[key], self.seen[key])
            else:
                self.assertEqual(before[key], self.seen[key])

    def test_command_error_is_measured_without_a_success_rate(self):
        rows = self.run_monitor()
        self.assertEqual(rows[0]['mean30_translation_L2_m'], 0.)
        self.assertEqual(rows[0]['mean30_gripper_accuracy'], 1.)
        self.assertNotIn('success', rows[0])
        self.assertNotIn('success_rate', rows[0])

    def test_malformed_or_nonfinite_predictions_fail_closed(self):
        with self.assertRaises(ValueError): self.run_monitor(self.prediction[:1])
        broken = self.prediction.copy()
        broken[0, 0] = np.nan
        with self.assertRaises(ValueError): self.run_monitor(broken)


if __name__ == '__main__':
    unittest.main()
