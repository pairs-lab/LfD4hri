import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.audit_training_readiness import physical_scene_key, source_reset_provenance
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import reset_source_episode


def task(pick="red apple", x=0):
    return {"name": "pick", "pick_name": pick, "pick_position": [x, -.5, .03],
            "place_name": None, "scene_objects": []}


class MixedSourceResetTest(unittest.TestCase):
    def test_raw_does_not_require_paired_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(source_reset_provenance(folder, {"task": task()}), {})

    def test_reset_fields_without_canonical_episode_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for extra in ({"recovery_start_actions_not_training_labels": [[0] * 7]},
                          {"canonical_source": folder},
                          {"paired_reset_strategy": "canonical_spawn_order_then_teacher_retarget"}):
                with self.assertRaises(ValueError):
                    source_reset_provenance(folder, {"task": task(), **extra})

    def test_mixed_parents_preserve_per_row_reset_and_pin_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for number in range(2):
                parent = root / f"parent{number}"
                parent.mkdir()
                index_path = parent / "index.json"
                index_path.write_text(json.dumps([{"file": "scene.npz", "task": task(x=number)}]))
                row = {"task": task(x=number), "paired_scene_source": "scene.npz", "variant": 0,
                       "canonical_source": str(parent), "recovery_start_actions_not_training_labels": [[0] * 7]}
                bound = source_reset_provenance(root, row)
                self.assertEqual(bound["canonical_source"], str(parent))
                self.assertEqual(bound["canonical_source_index_sha256"], sha256_file(index_path))
                index_path.write_text(index_path.read_text() + "\n")
                with self.assertRaises(ValueError):
                    source_reset_provenance(root, {**row, **bound})

    def test_changed_physical_scene_and_conflicting_parents_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "index.json").write_text(json.dumps([{"file": "scene.npz", "task": task()}]))
            row = {"task": task(x=1), "paired_scene_source": "scene.npz", "variant": 0,
                   "canonical_source": folder}
            with self.assertRaises(ValueError):
                source_reset_provenance(root, row)
            with self.assertRaises(ValueError):
                source_reset_provenance(root, {**row, "task": task(), "paired_source_dataset": str(root / "other")})

    def test_replay_uses_per_row_parent_and_excludes_startup_from_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "index.json").write_text(json.dumps([{"file": "scene.npz", "task": task()}]))
            startup = [[0., 0., 0., 0., 0., 0., 1.]]
            item = {"task": task(), "paired_scene_source": "scene.npz", "variant": 0,
                    "canonical_source": folder, "paired_reset_strategy": "canonical_spawn_order_then_teacher_retarget",
                    "recovery_start_actions_not_training_labels": startup}
            with patch("VLA.dataset_replay_context.TaskSpec") as spec, patch(
                    "VLA.audit_counterfactual_collection.CanonicalSceneResetEnvironment") as wrapper:
                spec.from_dict.side_effect = lambda x: x
                env = object()
                reset_source_episode(env, root / "split_without_audit_json", item)
                wrapper.assert_called_once_with(env, task(), startup)
                wrapper.return_value.reset.assert_called_once_with(task())

    def test_retarget_preserves_scene_key(self):
        scene = {**task(), "scene_objects": [{"name": "blue block", "position": [.2, -.5, .03], "fixed": False}]}
        other = {**scene, "pick_name": "blue block", "pick_position": [.2, -.5, .03],
                 "scene_objects": [{"name": "red apple", "position": [0, -.5, .03], "fixed": False}]}
        self.assertEqual(physical_scene_key(scene), physical_scene_key(other))


if __name__ == "__main__":
    unittest.main()
