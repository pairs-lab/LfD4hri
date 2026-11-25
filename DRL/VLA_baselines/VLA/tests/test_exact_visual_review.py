import json
from pathlib import Path
import tempfile
import unittest

from VLA.common.provenance import sha256_file
from VLA.publish_exact_visual_review import publish


class ExactVisualReviewTest(unittest.TestCase):
    def fixture(self, root):
        for name in ("combined_candidate", "native_smol512", "combined_source_audit", "preview"):
            (root/name).mkdir()
        (root/"combined_candidate/index.json").write_text("[]")
        for name in ("native_smol512", "combined_source_audit"):
            (root/name/"audit.json").write_text("{}")
        (root/"workflow_status.json").write_text(json.dumps({"status": "awaiting_actual_visual_review"}))
        images = []
        for name in ("native_smol512/native_inputs_contact_sheet.png", "combined_source_audit/camera_contact_sheet.png"):
            actual = root/"preview"/Path(name).name
            actual.write_bytes(b"unit-test-only-pixels")
            (root/name).write_bytes(actual.read_bytes())
            images.append({"path": str(actual), "sha256": sha256_file(actual), "final_relative_path": name})
        review = {"reviewer": "codex_actual_image_inspection", "acceptable_for_bounded_fit": True,
            "tasks_reviewed": ["touch", "pick", "move", "pick_place"], "byte_identical_republication_approved": True,
            "source_index_sha256": sha256_file(root/"combined_candidate/index.json"),
            "native_report_sha256": sha256_file(root/"native_smol512/audit.json"),
            "source_audit_sha256": sha256_file(root/"combined_source_audit/audit.json"), "images": images}
        path = root/"pre_review.json"
        path.write_text(json.dumps(review))
        return path

    def test_exact_pixels_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            self.assertTrue(publish(root, path).is_file())
            with self.assertRaises(FileExistsError):
                publish(root, path)

    def test_changed_final_pixels_reject_without_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            (root/"native_smol512/native_inputs_contact_sheet.png").write_bytes(b"different")
            with self.assertRaises(ValueError):
                publish(root, path)
            self.assertFalse((root/"visual_review_receipt.json").exists())

    def test_changed_report_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            (root/"native_smol512/audit.json").write_text('{"different": true}')
            with self.assertRaises(ValueError):
                publish(root, path)

    def test_cannot_publish_before_all_pre_review_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = self.fixture(root)
            (root/"workflow_status.json").write_text('{"status": "running"}')
            with self.assertRaisesRegex(ValueError, "barrier"):
                publish(root, path)
