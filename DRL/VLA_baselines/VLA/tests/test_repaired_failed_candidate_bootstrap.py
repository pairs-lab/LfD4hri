import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from VLA.collect_expanded_v9_candidate import bootstrap_repaired_prefix, stratified_names
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks, randomize_rl_scene


class RepairedFailedBootstrapTest(unittest.TestCase):
    def pose_fixture(self, source):
        plan, evidence, repair, before, rows = self.fixture(source, recovery_failure=True)
        plan["expert_teacher"] = "staged_pose_revalidated_acquisition"
        teacher = Path(__file__).resolve().parents[1]/"pose_revalidated_acquisition_teacher.py"
        repair.update(teacher="VLA.pose_revalidated_acquisition_teacher:PoseRevalidatedAcquisitionTeacher",
            reset_exact=False, reset_policy_inputs_exact=True,
            reset_errors={"front_rgb": 0., "wrist_rgb": 0., "joint_positions": 0., "gripper_closure": 8e-16},
            protected_hashes={str(teacher): sha256_file(teacher)})
        replay_path = source/"replay.json"
        summary = {"episodes": 1, "successes": 1, "initial_false_positives": 0}
        replay = {"passed": True, "mode": "both", "native_success_mode": "strict", "failures": [],
                  "source": str(source), "source_index_sha256": sha256_file(source/"index.json"),
                  "source_manifest_sha256": sha256_file(source/"dataset_manifest.json"),
                  "summaries": {"canonical": summary, "native": summary}}
        replay_path.write_text(json.dumps(replay))
        repair.update(canonical_native_replay_report=str(replay_path),
                      canonical_native_replay_sha256=sha256_file(replay_path))
        evidence.write_text(json.dumps(repair))
        return plan, evidence, repair, before, rows

    def test_pose_repair_requires_bound_dual_replay_and_preserves_float64_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, repair, before, old_rows = self.pose_fixture(source)
            rows, rng, lineage = bootstrap_repaired_prefix(source, plan, evidence)
            self.assertEqual(rows, old_rows[:-1])
            self.assertEqual(rng, before)
            self.assertFalse(repair["reset_exact"])
            self.assertIn("repair_replay_sha256", lineage)
            (source/"replay.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash-bound"):
                bootstrap_repaired_prefix(source, plan, evidence)

    def test_pose_repair_rejects_real_encoder_or_pixel_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, repair, *_ = self.pose_fixture(source)
            for field, value in (("front_rgb", 1), ("joint_positions", 1e-7), ("gripper_closure", 1e-8)):
                bad = copy.deepcopy(repair)
                bad["reset_errors"][field] = value
                evidence.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "same-reset"):
                    bootstrap_repaired_prefix(source, plan, evidence)

    def fixture(self, source, recovery_failure=False, first_failure=False):
        keys = ("contract_source", "contract_index_sha256", "contract_manifest_sha256", "validation",
                "validation_index_sha256", "validation_manifest_sha256", "observation_profile_sha256",
                "environment_runtime_fingerprint", "pybullet_binary_sha256")
        plan = {key: "frozen" for key in keys}
        plan.update(seed=2026101003, episodes_per_task=2, frozen_reward_distances={},
                    expert_teacher="staged", closed_hover_recovery_every=2,
                    planned_ordinary_episodes=8, planned_closed_hover_recovery_episodes=4, planned_episodes=12)
        rng, rows = np.random.default_rng(plan["seed"]), []
        task = select_tasks("all")[0]
        for number in range(1 if first_failure else 2):
            before = copy.deepcopy(rng.bit_generator.state)
            pick, place = stratified_names(task.name, number)
            sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
            files = [f"touch_{number:05d}.npz"]
            if number == 1 and recovery_failure:
                files.append("touch_00001_closedhover.npz")
            for name in files:
                rows.append({"file": name, "success": True, "safety_violation": None,
                             "task": json.loads(json.dumps(sampled.to_dict()))})
                (source/name).write_bytes(name.encode())
        rows[-1].update(success=False, safety_violation="preserved collision")
        for name, data in (("collection_plan.json", plan), ("index.json", rows), ("dataset_manifest.json", {})):
            (source/name).write_text(json.dumps(data))
        state = {"status": "stopped_on_failed_or_unsafe_episode", "completed_episodes": len(rows),
                 "last_file": rows[-1]["file"], "rng_state": rng.bit_generator.state,
                 "index_sha256": sha256_file(source/"index.json")}
        (source/"collection_state.json").write_text(json.dumps(state))
        teacher = Path(__file__).resolve().parents[1]/"obstacle_aware_touch_teacher.py"
        repair = {"source": str(source), "episode": rows[-1]["file"],
                  "source_index_sha256": state["index_sha256"],
                  "source_state_sha256": sha256_file(source/"collection_state.json"),
                  "source_payload_sha256": sha256_file(source/rows[-1]["file"]),
                  "task": rows[-1]["task"], "passed": True, "reset_exact": True, "runtime_changed": False,
                  "training_performed": False, "final_test_used": False, "teacher": "obstacle_aware",
                  "protected_hashes": {str(teacher): sha256_file(teacher)},
                  "metadata": {"success": True, "safety_violation": None}}
        evidence = source/"repair.json"
        evidence.write_text(json.dumps(repair))
        return {**plan, "expert_teacher": "staged_obstacle_touch"}, evidence, repair, before, rows

    def test_normal_failure_is_repeated_without_discarding_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, _, before, old_rows = self.fixture(source)
            rows, rng_state, lineage = bootstrap_repaired_prefix(source, plan, evidence)
            self.assertEqual(rows, old_rows[:-1])
            self.assertEqual(rng_state, before)
            self.assertFalse(lineage["failed_scene_skipped"])
            self.assertTrue(lineage["original_failure_preserved"])
            self.assertEqual(lineage["failed_file"], "touch_00001.npz")

    def test_recovery_failure_preserves_normal_and_replays_same_scene_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, _, before, old_rows = self.fixture(source, recovery_failure=True)
            rows, rng_state, lineage = bootstrap_repaired_prefix(source, plan, evidence)
            self.assertEqual(rows[-1]["file"], "touch_00001.npz")
            self.assertEqual(rng_state, before)
            self.assertEqual(lineage["failed_file"], "touch_00001_closedhover.npz")
            self.assertEqual(old_rows[-1]["task"], rows[-1]["task"])

    def test_first_failure_has_no_safe_prefix_but_keeps_initial_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, evidence, _, before, _ = self.fixture(Path(directory), first_failure=True)
            rows, rng_state, _ = bootstrap_repaired_prefix(directory, plan, evidence)
            self.assertEqual(rows, [])
            self.assertEqual(rng_state, before)

    def test_changed_camera_seed_recovery_and_unsafe_proof_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, repair, _, _ = self.fixture(source)
            for key, value in (("seed", 4), ("observation_profile_sha256", "new"), ("closed_hover_recovery_every", 0)):
                with self.assertRaisesRegex(ValueError, "physical/data/seed/recovery"):
                    bootstrap_repaired_prefix(source, {**plan, key: value}, evidence)
            for key, value in (("reset_exact", False), ("passed", False), ("runtime_changed", True)):
                evidence.write_text(json.dumps({**repair, key: value}))
                with self.assertRaisesRegex(ValueError, "exact safe same-reset"):
                    bootstrap_repaired_prefix(source, plan, evidence)

    def test_reroll_and_missing_contiguous_episode_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, repair, _, rows = self.fixture(source)
            rows[0]["task"]["pick_position"][0] += .01
            (source/"index.json").write_text(json.dumps(rows))
            state = json.loads((source/"collection_state.json").read_text())
            state["index_sha256"] = sha256_file(source/"index.json")
            (source/"collection_state.json").write_text(json.dumps(state))
            repair.update(source_index_sha256=state["index_sha256"],
                          source_state_sha256=sha256_file(source/"collection_state.json"))
            evidence.write_text(json.dumps(repair))
            with self.assertRaisesRegex(ValueError, "contiguous declared"):
                bootstrap_repaired_prefix(source, plan, evidence)


if __name__ == "__main__":
    unittest.main()
