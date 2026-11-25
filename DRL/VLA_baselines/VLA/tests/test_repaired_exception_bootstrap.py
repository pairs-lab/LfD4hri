import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from VLA.collect_expanded_v9_candidate import bootstrap_repaired_exception, stratified_names
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks, randomize_rl_scene


class RepairedExceptionBootstrapTests(unittest.TestCase):
    def fixture(self, source):
        keys = ("contract_source", "contract_index_sha256", "contract_manifest_sha256", "validation",
            "validation_index_sha256", "validation_manifest_sha256", "observation_profile_sha256",
            "environment_runtime_fingerprint", "pybullet_binary_sha256")
        plan = {k: "frozen" for k in keys}
        plan.update(seed=2026101003, episodes_per_task=3, frozen_reward_distances={},
            expert_teacher="staged_open_obstacle_touch", closed_hover_recovery_every=2,
            planned_ordinary_episodes=12, planned_closed_hover_recovery_episodes=4, planned_episodes=16)
        rng, rows = np.random.default_rng(plan["seed"]), []
        task = select_tasks("all")[0]
        for n in range(2):
            pick, place = stratified_names(task.name, n)
            sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
            files = [f"touch_{n:05d}.npz"]+(["touch_00001_closedhover.npz"] if n == 1 else [])
            for filename in files:
                rows.append({"file": filename, "task": json.loads(json.dumps(sampled.to_dict())),
                    "success": True, "safety_violation": None})
                (source/filename).write_bytes(filename.encode())
        before = copy.deepcopy(rng.bit_generator.state)
        pick, place = stratified_names(task.name, 2)
        sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
        for name, data in (("collection_plan.json", plan), ("index.json", rows), ("dataset_manifest.json", {})):
            (source/name).write_text(json.dumps(data))
        state = {"status": "stopped_on_collection_exception", "file": "touch_00002.npz", "message": "old exception",
            "task": json.loads(json.dumps(sampled.to_dict())), "completed_episodes": len(rows), "rng_state": before,
            "rng_after_sampling": rng.bit_generator.state, "index_sha256": sha256_file(source/"index.json")}
        (source/"collection_state.json").write_text(json.dumps(state))
        (source/"collection_exception.json").write_text(json.dumps(state))
        reproduction = {"parent_source": str(source), "state_sha256": sha256_file(source/"collection_state.json"),
            "index_sha256": state["index_sha256"], "task": state["task"], "exception": {"message": state["message"]}}
        repro = source/"reproduction.json"
        repro.write_text(json.dumps(reproduction))
        teacher = Path(__file__).resolve().parents[1]/"yaw_clear_touch_teacher.py"
        repaired = {"source": str(source), "episode": state["file"], "source_state_sha256": reproduction["state_sha256"],
            "source_index_sha256": state["index_sha256"], "reproduction_sha256": sha256_file(repro), "task": state["task"],
            "passed": True, "reset_exact": True, "runtime_changed": False, "training_performed": False,
            "final_test_used": False, "teacher": "yaw_clear", "protected_hashes": {str(teacher): sha256_file(teacher)},
            "metadata": {"success": True, "safety_violation": None}}
        repair = source/"repair.json"
        repair.write_text(json.dumps(repaired))
        return {**plan, "expert_teacher": "staged_yaw_clear_touch"}, repro, repair, rows, before, repaired

    def test_recovery_rows_retained_rng_not_inferred_from_row_count(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            plan, repro, repair, original, before, _ = self.fixture(source)
            rows, rng, metadata = bootstrap_repaired_exception(source, plan, repro, repair)
            self.assertEqual(rows, original)
            self.assertEqual(rng, before)
            self.assertEqual(metadata["failed_file"], "touch_00002.npz")
            self.assertFalse(metadata["failed_scene_skipped"])
            self.assertTrue(metadata["original_exception_preserved"])

    def test_changed_contract_and_unsafe_or_inexact_repair_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            plan, repro, repair, _, _, evidence = self.fixture(source)
            with self.assertRaisesRegex(ValueError, "physical/data/seed/recovery"):
                bootstrap_repaired_exception(source, {**plan, "seed": 1}, repro, repair)
            for key in ("passed", "reset_exact"):
                repair.write_text(json.dumps({**evidence, key: False}))
                with self.assertRaisesRegex(ValueError, "safe same-reset"):
                    bootstrap_repaired_exception(source, plan, repro, repair)

    def test_new_section_teacher_requires_its_own_same_reset_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            plan, repro, repair, original, _, evidence = self.fixture(source)
            plan["expert_teacher"] = "staged_central_section_clearance"
            with self.assertRaisesRegex(ValueError, "safe same-reset"):
                bootstrap_repaired_exception(source, plan, repro, repair)
            teacher = Path(__file__).resolve().parents[1]/"central_section_clearance_teacher.py"
            evidence.update(teacher="central_section_clearance", protected_hashes={str(teacher): sha256_file(teacher)})
            repair.write_text(json.dumps(evidence))
            rows, _, metadata = bootstrap_repaired_exception(source, plan, repro, repair)
            self.assertEqual(rows, original)
            self.assertFalse(metadata["failed_scene_skipped"])


if __name__ == "__main__":
    unittest.main()
