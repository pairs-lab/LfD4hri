import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from VLA.smolvla.audit_simstate_predictions import saved_normalization, diagnostic_frames


class SavedSimStateNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.projection = self.root/"projection"
        self.projection.mkdir()
        self.checkpoint = self.root/"checkpoint"
        self.checkpoint.mkdir()
        self.stats = {k: (np.arange(18, dtype=np.float32)+1).tolist()
                      for k in ("mean", "std", "min", "max")}
        (self.projection/"stats.json").write_text(json.dumps(self.stats))

    def save(self, dimensions=18, error=0.):
        values = {"observation.state."+k: torch.tensor(v[:dimensions])+error
                  for k, v in self.stats.items()}
        save_file(values, self.checkpoint/"policy_preprocessor_step_5_normalizer_processor.safetensors")

    def test_exact_saved_train18(self):
        self.save()
        report = saved_normalization(self.checkpoint, self.projection)
        self.assertTrue(report["passed"])
        self.assertFalse(report["validation_statistics_used"])

    def test_stale_seven_state_rejected(self):
        self.save(dimensions=7)
        with self.assertRaisesRegex(ValueError, "TRAIN18"):
            saved_normalization(self.checkpoint, self.projection)

    def test_different_statistics_rejected(self):
        self.save(error=.1)
        with self.assertRaisesRegex(ValueError, "TRAIN18"):
            saved_normalization(self.checkpoint, self.projection)

    def test_missing_statistics_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete saved"):
            saved_normalization(self.checkpoint, self.projection)


class InformativeFrameSelectionTests(unittest.TestCase):
    def test_rare_xy_decision_not_missed_by_uniform_frames(self):
        actions = np.zeros((100, 7), dtype=np.float32)
        actions[:, 2] = -.02
        actions[17, 0] = .004
        actions[23, 1] = .05
        selected = diagnostic_frames(actions)
        self.assertIn(17, selected)
        self.assertIn(23, selected)
        self.assertIn(0, selected)
        self.assertIn(99, selected)
        self.assertEqual(len(selected), len(set(selected)))

    def test_pure_vertical_frames_not_labeled_xy_informative(self):
        actions = np.zeros((40, 7), dtype=np.float32)
        actions[:, 2] = -.02
        np.testing.assert_array_equal(diagnostic_frames(actions), [0, 13, 26, 39])


if __name__ == "__main__":
    unittest.main()
