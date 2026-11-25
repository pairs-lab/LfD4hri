import copy
import json
from pathlib import Path
import unittest

from VLA.common.provenance import sha256_file
from VLA.run_evaluation_protocol import locked_split_seeds, locked_video_seed
from VLA.audit_evaluation_protocol import protocol_teacher


class LockedPilotEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.pilot = json.loads((self.root/"evaluation_protocol_v9_postfit_10_per_action.json").read_text())
        self.final = json.loads((self.root/"evaluation_protocol_v9_40_per_action.json").read_text())

    def test_all_four_tasks_ten_fresh_seeds_and_same_contract(self):
        seeds = locked_split_seeds(self.pilot, "development")
        self.assertEqual(len(seeds), 10)
        self.assertFalse(set(seeds)&set(self.final["development_seeds"]+self.final["final_test_seeds"]))
        for key in ("tasks", "robot", "gripper", "control_hz", "scene_profile",
                    "observation_profile_sha256", "grasp_candidate_scope"):
            self.assertEqual(self.pilot[key], self.final[key])
        self.assertEqual(self.pilot["success"]["semantic_thresholds"], self.final["success"]["semantic_thresholds"])
        self.assertEqual(self.pilot["reserved_final_protocol_sha256"],
                         sha256_file(self.root/"evaluation_protocol_v9_40_per_action.json"))
        self.assertEqual(self.pilot["action_chunk_selection"]["development_candidates"], [1])

    def test_video_one_predeclared_seed_legacy_dev_stays_without_video(self):
        seeds = locked_split_seeds(self.pilot, "development")
        video = locked_video_seed(self.pilot, "development", seeds)
        self.assertEqual(sum(seed == video for seed in seeds), 1)
        self.assertIsNone(locked_video_seed(self.final, "development", self.final["development_seeds"]))

    def test_empty_final_or_duplicate_leaked_seeds_rejected(self):
        with self.assertRaises(ValueError):
            locked_split_seeds(self.pilot, "final_test")
        for value in ([20266001, 20266001], [self.pilot["train_dataset_seed"]]):
            changed = copy.deepcopy(self.pilot)
            changed["development_seeds"] = value
            with self.assertRaises(ValueError):
                locked_split_seeds(changed, "development")

    def test_invalid_video_seed_rejected_before_rollout(self):
        changed = copy.deepcopy(self.pilot)
        changed["reporting"]["development_video_seed"] = 1
        with self.assertRaises(ValueError):
            locked_video_seed(changed, "development", changed["development_seeds"])

    def test_offline_teacher_explicit_and_never_a_learned_policy(self):
        from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
        from VLA.collision_planned_teacher import CollisionPlannedTeacher
        profile = {"gripper_controller": "body_contact_v9"}
        self.assertIs(protocol_teacher(self.pilot, profile), CentralSectionClearanceTeacher)
        self.assertIs(protocol_teacher(self.final, profile), CollisionPlannedTeacher)
        with self.assertRaises(ValueError):
            protocol_teacher(self.pilot, {})


if __name__ == "__main__":
    unittest.main()
