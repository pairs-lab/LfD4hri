import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from VLA.collect_expanded_v9_candidate import bootstrap_prefix, stratified_names
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.common.provenance import sha256_file


class CandidateBootstrapTest(unittest.TestCase):
    def fixture(self, source):
        keys = ("contract_source", "contract_index_sha256", "contract_manifest_sha256", "validation",
                "validation_index_sha256", "validation_manifest_sha256", "observation_profile_sha256",
                "environment_runtime_fingerprint", "pybullet_binary_sha256")
        plan = {key: "frozen" for key in keys}
        plan.update(seed=2026100902, episodes_per_task=2, frozen_reward_distances={}, expert_teacher="old")
        rng = np.random.default_rng(plan["seed"])
        task = select_tasks("all")[0]
        pick, place = stratified_names(task.name, 0)
        first = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
        row = {"file": "touch_00000.npz", "success": True, "safety_violation": None,
               "task": json.loads(json.dumps(first.to_dict()))}
        (source/"collection_plan.json").write_text(json.dumps(plan))
        (source/"index.json").write_text(json.dumps([row]))
        (source/row["file"]).write_bytes(b"immutable-test-payload")
        (source/"dataset_manifest.json").write_text("{}")
        state = {"completed_episodes": 1, "index_sha256": sha256_file(source/"index.json"),
                 "rng_state": rng.bit_generator.state}
        (source/"collection_state.json").write_text(json.dumps(state))
        pick, place = stratified_names(task.name, 1)
        following = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
        reproduction = {"parent_source": str(source), "state_sha256": sha256_file(source/"collection_state.json"),
                        "index_sha256": state["index_sha256"], "exception": {"message": "preserved"},
                        "task": json.loads(json.dumps(following.to_dict()))}
        evidence = source/"reproduction.json"
        evidence.write_text(json.dumps(reproduction))
        return {**plan, "expert_teacher": "new"}, evidence, reproduction

    def test_exact_prefix_retains_rng_and_teacher_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, _ = self.fixture(source)
            rows, rng, proof = bootstrap_prefix(source, plan, evidence)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rng, json.loads((source/"collection_state.json").read_text())["rng_state"])
            self.assertEqual((proof["old_teacher"], proof["new_teacher"]), ("old", "new"))
            self.assertFalse(proof["failed_next_scene_skipped"])

    def test_changed_scene_or_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, reproduction = self.fixture(source)
            with self.assertRaisesRegex(ValueError, "physical/data/seed"):
                bootstrap_prefix(source, {**plan, "seed": 4}, evidence)
            reproduction["task"]["pick_position"][0] += .01
            evidence.write_text(json.dumps(reproduction))
            with self.assertRaisesRegex(ValueError, "next failed scene"):
                bootstrap_prefix(source, plan, evidence)

    def test_evidence_from_another_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            plan, evidence, reproduction = self.fixture(source)
            reproduction["parent_source"] = "/tmp/not-the-preserved-source"
            evidence.write_text(json.dumps(reproduction))
            with self.assertRaisesRegex(ValueError, "next failed scene"):
                bootstrap_prefix(source, plan, evidence)
