import json
from pathlib import Path
import tempfile
import unittest
from VLA.prepare_v9_candidate_fit import additional_collection_ready, collection_ready, require_visual_receipt, fit_recipe_tokens
from VLA.common.provenance import sha256_file


class CandidateFitWorkflowTest(unittest.TestCase):
    def test_additional_live_prefix_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source/"collection_state.json").write_text(json.dumps({"status": "in_progress"}))
            self.assertFalse(additional_collection_ready(source, 38))
            (source/"collection_state.json").write_text(json.dumps({"status": "stopped_error_NOT_training"}))
            with self.assertRaisesRegex(RuntimeError, "preserved a failure"):
                additional_collection_ready(source, 38)

    def test_legacy_complete_additional_source_needs_exact_counts_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.assertFalse(additional_collection_ready(source, 1))
            rows = [{"task": {"name": name}, "success": True, "safety_violation": None}
                    for name in ("touch", "pick", "move", "pick_place")]
            index = source/"index.json"
            index.write_text(json.dumps(rows))
            (source/"dataset_manifest.json").write_text(json.dumps({"episodes": 4, "index_sha256": sha256_file(index)}))
            self.assertTrue(additional_collection_ready(source, 1))
            with self.assertRaises(RuntimeError):
                additional_collection_ready(source, 2)

    def test_explicit_refit_recipe_preserves_old_defaults(self):
        self.assertEqual(fit_recipe_tokens(), ["--initial-frame-sampling-probability", "0.3",
                                             "--learning-rate", "0.0001", "--batch-size", "1"])

    def test_new_recipe_forwarding_is_explicit(self):
        self.assertEqual(fit_recipe_tokens(.10, 5e-5, 2),
                         ["--initial-frame-sampling-probability", "0.1",
                          "--learning-rate", "5e-05", "--batch-size", "2"])

    def test_invalid_refit_recipe_is_not_silently_defaulted(self):
        for recipe in ((0., 1e-4, 1), (1., 1e-4, 1), (.1, float("nan"), 1),
                       (float("inf"), 1e-4, 1), (.1, -1e-4, 1), (.1, 1e-4, 0),
                       (.1, 1e-4, True)):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                fit_recipe_tokens(*recipe)

    def test_stopped_collection_never_triggers_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.assertFalse(collection_ready(source, 1))
            state = source/"collection_state.json"
            state.write_text(json.dumps({"status": "in_progress"}))
            self.assertFalse(collection_ready(source, 1))
            state.write_text(json.dumps({"status": "stopped_on_collection_exception"}))
            with self.assertRaisesRegex(RuntimeError, "preserved a failure"):
                collection_ready(source, 1)

    def test_completed_flag_alone_is_not_a_quality_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            rows = [{"task": {"name": name}, "success": True, "safety_violation": None}
                    for name in ("touch", "pick", "move", "pick_place")]
            index = source/"index.json"
            index.write_text(json.dumps(rows))
            state = {"status": "completed_candidate_not_qualified", "completed_episodes": 4,
                     "index_sha256": sha256_file(index)}
            (source/"collection_state.json").write_text(json.dumps(state))
            self.assertTrue(collection_ready(source, 1))
            rows[0]["safety_violation"] = "floor"
            index.write_text(json.dumps(rows))
            with self.assertRaises(RuntimeError):
                collection_ready(source, 1)

    def test_visual_review_binds_both_native_and_source_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, native, audit = root/"source", root/"native", root/"source_audit"
            for path in (source, native, audit):
                path.mkdir()
            (source/"index.json").write_text("[]")
            for path in (native/"audit.json", audit/"audit.json"):
                path.write_text("{}")
            images = [native/"native_inputs_contact_sheet.png", audit/"camera_contact_sheet.png"]
            for path in images:
                path.write_bytes(b"fake-unit-test-image")
            receipt = {"source_index_sha256": sha256_file(source/"index.json"),
                "native_report_sha256": sha256_file(native/"audit.json"), "source_audit_sha256": sha256_file(audit/"audit.json"),
                "reviewer": "codex_actual_image_inspection", "tasks_reviewed": ["touch", "pick", "move", "pick_place"],
                "acceptable_for_bounded_fit": True, "images": [{"path": str(p), "sha256": sha256_file(p)} for p in images]}
            path = root/"receipt.json"
            path.write_text(json.dumps(receipt))
            require_visual_receipt(path, source, native, audit/"audit.json")
            images[0].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image changed"):
                require_visual_receipt(path, source, native, audit/"audit.json")
