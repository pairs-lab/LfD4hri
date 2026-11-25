import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from VLA.smolvla.audit_projection_adapter_weights import audit_weights


class ProjectionWeightPreservationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)/"base.safetensors"
        self.saved = Path(self.folder.name)/"saved.safetensors"
        self.weights = {"model.state_proj.weight": torch.arange(64, dtype=torch.float32).reshape(2, 32),
                        "frozen_bf16": torch.tensor([0., 1., 2.], dtype=torch.bfloat16),
                        "frozen_fp32": torch.tensor([3., 4.])}
        save_file(self.weights, self.base)

    def candidate(self):
        return {k: v.clone() for k, v in self.weights.items()}

    def check(self, weights):
        save_file(weights, self.saved)
        return audit_weights(self.base, self.saved)

    def test_native_bf16_and_new_columns(self):
        weights = self.candidate()
        weights["model.state_proj.weight"][:, 7:18] += 10
        self.assertTrue(self.check(weights)["passed"])

    def test_frozen_bf16_value_rejected(self):
        weights = self.candidate()
        weights["frozen_bf16"][1] += 1
        with self.assertRaisesRegex(ValueError, "supposedly frozen"):
            self.check(weights)

    def test_signed_zero_bits_rejected(self):
        weights = self.candidate()
        weights["frozen_bf16"][0] = -0.
        with self.assertRaisesRegex(ValueError, "supposedly frozen"):
            self.check(weights)

    def test_original_state_columns_rejected(self):
        weights = self.candidate()
        weights["model.state_proj.weight"][0, 6] += 1
        with self.assertRaisesRegex(ValueError, "state/padding"):
            self.check(weights)

    def test_unused_padding_rejected(self):
        weights = self.candidate()
        weights["model.state_proj.weight"][0, 18] += 1
        with self.assertRaisesRegex(ValueError, "state/padding"):
            self.check(weights)

    def test_silent_cast_rejected(self):
        weights = self.candidate()
        weights["frozen_bf16"] = weights["frozen_bf16"].float()
        with self.assertRaisesRegex(ValueError, "downcast"):
            self.check(weights)


if __name__ == "__main__":
    unittest.main()
