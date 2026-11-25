import json
from pathlib import Path
import tempfile
import unittest

from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.publish_exact_image_review import REQUIRED_STAGES, TASKS, verified_receipt


def fixture(root):
    source, native, audit, preview = (root/n for n in ("combined_candidate", "native_smol512", "combined_source_audit", "preview"))
    for p in (source, native, audit, preview):
        p.mkdir()
    (source/"ep.npz").write_bytes(b"unit fixture payload, not robot data")
    (source/"index.json").write_text(json.dumps([{"file": "ep.npz"}]))
    (source/"dataset_manifest.json").write_text("{}")
    (root/"workflow_plan.json").write_text("{}")
    for p in (preview/"native.png", native/"native_inputs_contact_sheet.png"):
        p.write_bytes(b"unit fixture native image; NOT actual review")
    (audit/"camera_contact_sheet.png").write_bytes(b"unit fixture source image")
    (native/"audit.json").write_text("{}")
    report = {"passed": True, "failures": [], "episodes": 1,
              "source_payload_fingerprint": source_payload_fingerprint(source)}
    (audit/"audit.json").write_text(json.dumps(report))
    replay = {"passed": True, "failures": [], "all_episodes": True, "native_success_mode": "strict",
              "summaries": {m: {"episodes": 1, "successes": 1} for m in ("canonical", "native")}}
    (root/"combined_full_physics_replay.json").write_text(json.dumps(replay))
    approval = {"reviewer": "codex_actual_image_inspection", "scope": "root_review_of_existing_exact_bytes",
                "tasks_reviewed": sorted(TASKS), "acceptable_for_bounded_fit": True, "workflow": str(root),
                "images": [{"inspected_path": str(original), "workflow_path": str(dest), "sha256": sha256_file(original)}
                    for original, dest in ((preview/"native.png", native/"native_inputs_contact_sheet.png"),
                        (audit/"camera_contact_sheet.png", audit/"camera_contact_sheet.png"))]}
    for key, path in {"source_index_sha256": source/"index.json", "source_manifest_sha256": source/"dataset_manifest.json",
        "source_audit_sha256": audit/"audit.json", "native_report_sha256": native/"audit.json",
        "workflow_plan_sha256": root/"workflow_plan.json"}.items():
        approval[key] = sha256_file(path)
    req = {"source": str(source), "required_images": [r["workflow_path"] for r in approval["images"]],
           "receipt": str(root/"visual_review_receipt.json"),
           **{k: approval[k] for k in ("source_index_sha256", "native_report_sha256", "source_audit_sha256")}}
    state = {"status": "awaiting_actual_visual_review", "stage": "review", "final_test_used": False,
             "run_training_requested": True, "history": [{"stage": n, "exit_code": 0} for n in sorted(REQUIRED_STAGES)]}
    return approval, req, state


class ExactImageReviewTest(unittest.TestCase):
    def test_exact_bytes_after_all_gates_can_publish_bound_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            result = verified_receipt(a, root, r, s)
            self.assertFalse(result["data_sufficiency_established"])
            self.assertFalse(result["learned_success_claimed"])
            self.assertEqual(len(result["images"]), 2)

    def test_changed_workflow_image_is_not_approved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            (root/"native_smol512/native_inputs_contact_sheet.png").write_bytes(b"unseen changed bytes")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                verified_receipt(a, root, r, s)

    def test_changed_original_inspected_image_is_not_approved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            (root/"preview/native.png").write_bytes(b"changed original")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                verified_receipt(a, root, r, s)

    def test_early_or_failed_workflow_does_not_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            for status in ("running", "stopped_no_silent_retry"):
                with self.assertRaisesRegex(ValueError, "review barrier"):
                    verified_receipt(a, root, r, {**s, "status": status})
            s["history"] = [x for x in s["history"] if x["stage"] != "full_physics_replay"]
            with self.assertRaisesRegex(ValueError, "upstream"):
                verified_receipt(a, root, r, s)

    def test_partial_or_failed_replay_does_not_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            p = root/"combined_full_physics_replay.json"
            original = json.loads(p.read_text())
            for patch in ({"all_episodes": False}, {"passed": False}, {"native_success_mode": "semantic"},
                          {"summaries": {m: {"episodes": 1, "successes": 0} for m in ("canonical", "native")}}):
                p.write_text(json.dumps({**original, **patch}))
                with self.assertRaisesRegex(ValueError, "strict replay"):
                    verified_receipt(a, root, r, s)

    def test_unreviewed_camera_or_report_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            with self.assertRaisesRegex(ValueError, "both mandatory"):
                verified_receipt({**a, "images": a["images"][:1]}, root, r, s)
            (root/"native_smol512/audit.json").write_text('{"unseen":"report"}')
            with self.assertRaisesRegex(ValueError, "native_report_sha256"):
                verified_receipt(a, root, r, s)

    def test_changed_unselected_payload_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            (root/"combined_candidate/ep.npz").write_bytes(b"different source bytes")
            with self.assertRaisesRegex(RuntimeError, "payload content/hash differs"):
                verified_receipt(a, root, r, s)

    def test_changed_preprocessor_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            config = root/"inspected_config.json"
            config.write_text("{}")
            a["bound_files"] = {str(config): sha256_file(config)}
            verified_receipt(a, root, r, s)
            config.write_text('{"image_size":123}')
            with self.assertRaisesRegex(ValueError, "preprocessing/configuration"):
                verified_receipt(a, root, r, s)

    def test_individually_reviewed_native_frame_requires_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, r, s = fixture(root)
            original, destination = root/"preview/detail.png", root/"native_smol512/detail.png"
            original.write_bytes(b"unit reviewed individual frame")
            destination.write_bytes(original.read_bytes())
            a["images"].append({"inspected_path": str(original), "workflow_path": str(destination), "sha256": sha256_file(original)})
            self.assertEqual(len(verified_receipt(a, root, r, s)["images"]), 3)
            destination.write_bytes(b"changed individual frame")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                verified_receipt(a, root, r, s)


if __name__ == "__main__":
    unittest.main()
