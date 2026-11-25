import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from VLA.common.provenance import sha256_file
from VLA.prepare_repaired_train_fit import repaired_collection_ready


class RepairedTrainPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.candidate, self.original, self.outer = [Path(self.temp.name) / p for p in ("candidate", "original", "outer")]
        for p in (self.candidate, self.original, self.outer):
            p.mkdir()
        self.old_rows = [{"file": "touch.npz", "task": {"name": "touch"}, "scene_group": "a", "success": True},
                         {"file": "pick.npz", "task": {"name": "pick"}, "scene_group": "b", "success": True}]
        self.rows = copy.deepcopy(self.old_rows)
        # Seed rows intentionally have NO reset_pixels_and_feedback_exact flag.
        self.rows[0].update(expert_touch_terminal_semantics="original_strict_AND_surface_gap_le_0.2mm",
                            expert_terminal_surface_gap_m=.0001, expert_terminal_max_normal_force_n=0.)
        self.rows[1]["non_touch_payload_unchanged"] = True
        for p in (self.candidate, self.original):
            np.savez_compressed(p / "touch.npz", front_rgb=np.zeros((2, 2, 2, 3), np.uint8),
                wrist_rgb=np.zeros((2, 2, 2, 3), np.uint8), joint_positions=np.zeros((2, 6)), gripper_closure=np.zeros(2))
            (p / "pick.npz").write_bytes(b"unchanged test payload")
        self.write(self.original / "index.json", self.old_rows)
        self.write(self.original / "dataset_manifest.json", {"split": "train", "observation_profile_sha256": "p", "environment_runtime_fingerprint": {"v": 1}})
        self.write(self.outer / "index.json", [{"scene_group": "outer-never-training"}])
        self.write(self.candidate / "dataset_manifest.json", {"observation_profile_sha256": "p", "environment_runtime_fingerprint": {"v": 1}})
        self.plan = {"validation_labels_used": False, "final_test_used": False,
            "train_source": str(self.original), "outer_validation": str(self.outer),
            "train_index_sha256": sha256_file(self.original / "index.json"),
            "train_manifest_sha256": sha256_file(self.original / "dataset_manifest.json"),
            "outer_validation_index_sha256": sha256_file(self.outer / "index.json"), "planned_counts": {"touch": 1, "pick": 1}}
        self.write(self.candidate / "repair_plan.json", self.plan)
        self.state = {"status": "completed_candidate_not_qualified", "planned_episodes": 2, "completed_episodes": 2}
        self.commit()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def commit(self):
        self.write(self.candidate / "index.json", self.rows)
        self.state["index_sha256"] = sha256_file(self.candidate / "index.json")
        self.write(self.candidate / "collection_state.json", self.state)

    def ready(self):
        return repaired_collection_ready(self.candidate, self.original, self.outer)

    def test_complete_seed_without_metadata_flag_requires_actual_reset_parity(self):
        self.assertTrue(self.ready())

    def test_waits_without_reading_partial_payloads(self):
        self.state["status"] = "in_progress"
        self.commit()
        self.assertFalse(self.ready())

    def test_collection_failure_is_never_skipped(self):
        self.state["status"] = "stopped_preserved_failure"
        self.commit()
        with self.assertRaises(ValueError):
            self.ready()

    def test_non_touch_bytes_cannot_change(self):
        (self.candidate / "pick.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.ready()

    def test_seed_pixels_cannot_change_even_without_a_flag(self):
        np.savez_compressed(self.candidate / "touch.npz", front_rgb=np.ones((2, 2, 2, 3), np.uint8),
            wrist_rgb=np.zeros((2, 2, 2, 3), np.uint8), joint_positions=np.zeros((2, 6)), gripper_closure=np.zeros(2))
        with self.assertRaises(ValueError):
            self.ready()

    def test_dropping_case_fails(self):
        self.rows.pop()
        self.commit()
        with self.assertRaises(ValueError):
            self.ready()

    def test_failed_case_fails(self):
        self.rows[0]["success"] = False
        self.commit()
        with self.assertRaises(ValueError):
            self.ready()

    def test_changed_scene_fails(self):
        self.rows[0]["task"]["pick_name"] = "other"
        self.commit()
        with self.assertRaises(ValueError):
            self.ready()

    def test_validation_use_fails(self):
        self.plan["validation_labels_used"] = True
        self.write(self.candidate / "repair_plan.json", self.plan)
        with self.assertRaises(ValueError):
            self.ready()


if __name__ == "__main__":
    unittest.main()
