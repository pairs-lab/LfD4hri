import unittest

import numpy as np
import torch

from VLA.smolvla.balanced_sampling import BalancedInitialSampler, frame_weights


class BalancedSamplingTest(unittest.TestCase):
    def index(self):
        return [{"task": {"name": name}, "steps": length}
                for name, length in zip(("touch", "pick", "move", "pick_place"), (3, 5, 7, 9))]

    def sampler(self):
        lengths = [item["steps"] for item in self.index()]
        return BalancedInitialSampler(np.r_[0, np.cumsum(lengths)[:-1]], np.cumsum(lengths),
                                      source_index=self.index(), initial_probability=.3, seed=42)

    def test_exact_task_and_initial_mass(self):
        weights = frame_weights(self.index(), .3)
        start = 0
        for item in self.index():
            end = start + item["steps"]
            self.assertAlmostEqual(float(weights[start:end].sum()), .25)
            self.assertAlmostEqual(float(weights[start]), .075)
            start = end

    def test_determinism_resume_and_global_rng_unchanged(self):
        rng = torch.get_rng_state().clone()
        first, repeat = self.sampler(), self.sampler()
        expected = list(first)
        self.assertEqual(expected, list(repeat))
        repeat.load_state_dict({"epoch": 0, "start_index": 11})
        self.assertEqual(expected[11:], list(repeat))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_source_offset_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            BalancedInitialSampler([0] * 4, [4] * 4, source_index=self.index())

    def test_target_balance_does_not_overweight_repeated_classes(self):
        rows = [{"task": {"name": task, "pick_name": target}, "steps": 3}
                for task in ("touch", "pick", "move", "pick_place")
                for target in ("rare", "common", "common", "common")]
        weights = frame_weights(rows, .3, target_balanced=True)
        for task_number in range(4):
            start = task_number * 12
            self.assertAlmostEqual(float(weights[start:start + 3].sum()), .125)
            self.assertAlmostEqual(float(weights[start + 3:start + 12].sum()), .125)


if __name__ == "__main__":
    unittest.main()
