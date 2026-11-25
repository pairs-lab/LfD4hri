import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.run_cross_vla120 import SEEDS, TASKS, model_specs, summarize_model
from VLA.cross_vla120_trial import task_json, validate_regime
from VLA.common.tasks import select_tasks


class CrossVLA120Tests(unittest.TestCase):
    def test_budget_is_120_for_each_model_not_120_total(self):
        self.assertEqual(len(SEEDS), 30)
        self.assertEqual(len(set(SEEDS)), 30)
        self.assertEqual(len(SEEDS) * len(TASKS), 120)
        self.assertEqual(len(model_specs()), 6)
        self.assertEqual(len(model_specs()) * len(SEEDS) * len(TASKS), 720)

    def test_pending_inference_is_null_not_zero_success(self):
        value = summarize_model({}, model_specs()[0], [], "queued")
        self.assertIsNone(value["success_rate_for_completed120"])
        self.assertIsNone(value["partial_success_rate"])
        self.assertEqual(value["completed_trials"], 0)

    def test_full_denominator_and_safety_failures_are_retained(self):
        rows = [{"seed": seed, "task": task, "success": seed == SEEDS[0],
                 "safety_violation": "floor_guard" if seed == SEEDS[-1] else None,
                 "error": None} for seed in SEEDS for task in TASKS]
        value = summarize_model({}, model_specs()[0], rows, "completed120")
        self.assertAlmostEqual(value["success_rate_for_completed120"], 4 / 120)
        self.assertEqual(value["safety_aborts"], 4)
        self.assertEqual(value["by_task"]["pick"]["completed"], 30)

    def test_integrator_error_cannot_be_counted_as_task_failure(self):
        rows = [{"seed": 1, "task": "pick", "success": False, "error": "bad shape"}]
        value = summarize_model({}, model_specs()[0], rows, "integration_error")
        self.assertEqual(value["completed_trials"], 0)
        self.assertIsNone(value["partial_success_rate"])

    def test_task_tuple_schema_compares_to_its_persisted_json(self):
        for task in select_tasks("all"):
            self.assertEqual(task_json(task), json.loads(json.dumps(task.to_dict())))
            self.assertEqual(task_json(task), json.loads(json.dumps(task_json(task))))

    def test_zero_shot_cannot_relabel_bound_target_weights(self):
        with patch("VLA.cross_vla120_trial.checkpoint_profile", return_value={"bound": True}):
            with self.assertRaises(ValueError):
                validate_regime("pi0", "official_zero_shot_v9", "old_finetuned", {})

    def test_pi0_requires_exact_declared_official_base(self):
        with patch("VLA.cross_vla120_trial.checkpoint_profile", return_value=None):
            with self.assertRaises(ValueError):
                validate_regime("pi0", "official_zero_shot_v9", "old_finetuned", {})
            validate_regime("pi0", "official_zero_shot_v9", "gs://openpi-assets/checkpoints/pi0_base", {})

    def test_qualified_v9_never_accepts_unbound_checkpoint(self):
        with patch("VLA.cross_vla120_trial.checkpoint_profile", return_value=None):
            with self.assertRaises(ValueError):
                validate_regime("rdt", "qualified_v9_postfit", "published/base", {})

    def test_front_only_models_do_not_masquerade_as_smol_postfit(self):
        regimes = {s["model"]: s["adaptation_regime"] for s in model_specs()}
        self.assertEqual(regimes["smolvla"], "qualified_v9_postfit")
        self.assertEqual(regimes["rdt"], "qualified_v9_postfit")
        self.assertEqual(regimes["openvla"], "official_zero_shot_v9")
        self.assertEqual(regimes["cogact"], "official_zero_shot_v9")


if __name__ == "__main__":
    unittest.main()
