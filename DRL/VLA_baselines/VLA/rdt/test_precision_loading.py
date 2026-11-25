"""Run in the isolated RDT runtime; no GPU or large checkpoint is required."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from VLA.rdt import server  # Pinned upstream modules.
from VLA.rdt.checkpoint_loading import load_checkpoint
from models.rdt_runner import RDTRunner


class RDTPrecisionLoadingTest(unittest.TestCase):
    def test_float_after_bf16_loading_cannot_restore_finetuned_values(self):
        config = {
            "action_dim": 128, "pred_horizon": 5,
            "rdt": {"hidden_size": 64, "depth": 1, "num_heads": 4},
            "lang_adaptor": "linear", "img_adaptor": "linear", "state_adaptor": "linear",
            "lang_token_dim": 16, "img_token_dim": 16, "state_token_dim": 128,
            "max_lang_cond_len": 4, "img_cond_len": 6,
            "noise_scheduler": {"num_train_timesteps": 20, "num_inference_timesteps": 2,
                                "beta_schedule": "squaredcos_cap_v2", "prediction_type": "sample",
                                "clip_sample": False},
        }
        source = RDTRunner(action_dim=128, pred_horizon=5, config=config,
            lang_token_dim=16, img_token_dim=16, state_token_dim=128,
            max_lang_cond_len=4, img_cond_len=6, dtype=torch.float32)
        with torch.no_grad():
            source.model.x_pos_embed[0, 0, 0] = 1.001234
        expected = source.model.x_pos_embed[0, 0, 0].item()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            torch.save(source.state_dict(), root / "pytorch_model.bin")
            legacy = RDTRunner.from_pretrained(str(root), local_files_only=True).float()
            self.assertNotEqual(legacy.model.x_pos_embed[0, 0, 0].item(), expected)
            fixed, _ = load_checkpoint(root, 5, dtype=torch.float32)
            self.assertEqual(fixed.model.x_pos_embed[0, 0, 0].item(), expected)
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, fixed.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
