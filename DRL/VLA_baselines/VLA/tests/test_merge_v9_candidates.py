import json
import os
from pathlib import Path
import tempfile
import unittest
from VLA.merge_v9_train_candidates import merge
from VLA.common.runtime_provenance import environment_runtime_fingerprint


class MergeV9CandidateTest(unittest.TestCase):
    def source(self, path, offset, split=None):
        path.mkdir()
        profile = {"gripper_controller": "body_contact_v9"}
        manifest = {"observation_profile": profile, "environment_runtime_fingerprint": environment_runtime_fingerprint(profile),
                    "robot": "ur5e", "required_success_steps": 1}
        if split:
            manifest["split"] = split
        (path/"dataset_manifest.json").write_text(json.dumps(manifest))
        rows = [{"file": f"{name}.npz", "success": True, "safety_violation": None, "steps": 1,
                 "task": {"name": name, "pick_name": "red apple", "pick_position": [offset+i, 0, 0],
                          "place_name": None, "scene_objects": []}}
                for i, name in enumerate(("touch", "pick", "move", "pick_place"))]
        for row in rows:
            (path/row["file"]).write_bytes(b"unit-test-only-payload")
        (path/"index.json").write_text(json.dumps(rows))
        return rows

    def test_every_row_and_payload_is_preserved_without_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.source(root/"a", 0, "train")
            self.source(root/"b", 10)
            self.source(root/"val", 20, "validation")
            result = merge([root/"a", root/"b"], root/"val", root/"merged")
            self.assertEqual(result["episodes"], 8)
            self.assertEqual(result["independent_layouts"], 8)
            manifest = json.loads((root/"merged/dataset_manifest.json").read_text())
            self.assertTrue(manifest["candidate_only"])
            self.assertFalse(manifest["training_certified"])
            self.assertEqual(os.stat(root/"a/pick.npz").st_ino, os.stat(root/"merged/component0_pick.npz").st_ino)

    def test_failed_component_cannot_be_silently_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.source(root/"a", 0)
            self.source(root/"val", 20, "validation")
            rows[0]["success"] = False
            (root/"a/index.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Failed/unsafe"):
                merge([root/"a"], root/"val", root/"merged")
            self.assertFalse((root/"merged").exists())

    def test_comparison_validation_is_excluded_by_physical_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.source(root/"a", 0)
            self.source(root/"val", 0, "validation")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                merge([root/"a"], root/"val", root/"merged")
