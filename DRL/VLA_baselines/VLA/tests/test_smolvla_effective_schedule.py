import json
import math
from pathlib import Path
import tempfile
import unittest

from VLA.smolvla.audit_scheduler_effective import (
    audit_checkpoint, effective_schedule, main, stock_lr_multiplier,
)


class EffectiveScheduleTest(unittest.TestCase):
    def checkpoint(self, root, total=16, accumulation=8, warmup=0, decay=8000):
        cp = root / "checkpoints" / f"{total:06d}" / "pretrained_model"
        state = cp.parent / "training_state"
        cp.mkdir(parents=True)
        state.mkdir()
        policy = {"type": "smolvla", "scheduler_warmup_steps": warmup,
                  "scheduler_decay_steps": decay, "scheduler_decay_lr": 2.5e-6,
                  "optimizer_lr": 1e-4}
        cfg = {"policy": policy, "steps": total,
               "accelerator": {"gradient_accumulation": {"steps": accumulation}},
               "scheduler": {"type": "cosine_decay_with_warmup", "num_warmup_steps": warmup,
                             "num_decay_steps": decay, "peak_lr": 1e-4, "decay_lr": 2.5e-6}}
        actual = effective_schedule(total, accumulation, warmup, decay)["effective_contract"]
        lr = 1e-4 * stock_lr_multiplier(total, actual["warmup_microsteps"],
                                       actual["decay_microsteps"], 1e-4, 2.5e-6)
        payloads = {cp / "config.json": policy, cp / "train_config.json": cfg,
                    state / "scheduler_state.json": {"last_epoch": total, "base_lrs": [1e-4], "_last_lr": [lr]},
                    state / "optimizer_param_groups.json": [{"lr": lr, "initial_lr": 1e-4, "params": [0]}]}
        for path, data in payloads.items():
            path.write_text(json.dumps(data))
        return cp

    def test_smoke_pass_does_not_hide_autoscale(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit_checkpoint(self.checkpoint(Path(directory)))
        self.assertTrue(report["passed"])
        self.assertFalse(report["configured_contract_is_effective"])
        self.assertEqual(report["configured_contract"]["decay_optimizer_updates"], 1000)
        self.assertEqual(report["effective_contract"]["decay_optimizer_updates"], 2)
        self.assertAlmostEqual(report["parameter_groups"][0]["saved_optimizer_lr"], 2.5e-6)

    def test_500_pilot_effective_warmup_and_decay(self):
        report = effective_schedule(8000, 16, 800, 16000)
        self.assertEqual(report["effective_contract"], {"warmup_microsteps": 400,
            "decay_microsteps": 8000, "warmup_optimizer_updates": 25, "decay_optimizer_updates": 500})
        self.assertFalse(report["configured_contract_is_effective"])

    def test_explicit_run_length_avoids_autoscale(self):
        report = effective_schedule(8000, 16, 800, 8000)
        self.assertTrue(report["configured_contract_is_effective"])
        self.assertEqual(report["effective_contract"]["warmup_optimizer_updates"], 50)

    def test_autoscaled_warmup_is_int_truncated_not_rounded(self):
        value = effective_schedule(16000, 16, 1600, 480000)
        self.assertEqual(value["effective_contract"]["warmup_microsteps"], 53)
        self.assertEqual(value["effective_contract"]["warmup_optimizer_updates"], 53 / 16)

    def test_stock_cosine_uses_global_step(self):
        expected = (1 - .025) * .5 * (1 + math.cos(math.pi * 800 / 8000)) + .025
        self.assertAlmostEqual(stock_lr_multiplier(800, 800, 8000, 1e-4, 2.5e-6), expected)
        self.assertNotEqual(expected, 1.0)
        self.assertAlmostEqual(stock_lr_multiplier(0, 800, 8000, 1e-4, 2.5e-6), 1 / 801)

    def test_incomplete_saved_schedule_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = self.checkpoint(Path(directory))
            path = cp.parent / "training_state/scheduler_state.json"
            data = json.loads(path.read_text())
            data["last_epoch"] = 15
            path.write_text(json.dumps(data))
            report = audit_checkpoint(cp)
        self.assertFalse(report["passed"])
        self.assertIn("declared completed microsteps", report["failures"][0])

    def test_wrong_optimizer_lr_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = self.checkpoint(Path(directory))
            path = cp.parent / "training_state/optimizer_param_groups.json"
            data = json.loads(path.read_text())
            data[0]["lr"] = 1e-4
            path.write_text(json.dumps(data))
            self.assertFalse(audit_checkpoint(cp)["passed"])

    def test_wrong_saved_scheduler_lr_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = self.checkpoint(Path(directory))
            path = cp.parent / "training_state/scheduler_state.json"
            data = json.loads(path.read_text())
            data["_last_lr"] = [1e-4]
            path.write_text(json.dumps(data))
            self.assertFalse(audit_checkpoint(cp)["passed"])

    def test_policy_config_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = self.checkpoint(Path(directory))
            path = cp / "config.json"
            data = json.loads(path.read_text())
            data["scheduler_decay_steps"] = 16
            path.write_text(json.dumps(data))
            self.assertFalse(audit_checkpoint(cp)["passed"])

    def test_nonfinite_lr_and_partial_accumulation_rejected(self):
        with self.assertRaises(ValueError):
            stock_lr_multiplier(0, 0, 16, float("nan"), 2.5e-6)
        with self.assertRaises(ValueError):
            effective_schedule(15, 8, 0, 8000)

    def test_cli_refuses_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cp = self.checkpoint(root)
            output = root / "existing.json"
            output.write_text("preserve")
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", str(cp), "--output", str(output)])
            self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
