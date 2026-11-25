import unittest
from VLA.run_evaluation_protocol import validate_final_grasp_scope


class FinalGraspFairnessTest(unittest.TestCase):
    def test_legacy_unqualified_final_rejected(self):
        for profile in (None, {}, {"grasp_candidate_scope": "selected_target_legacy"}):
            with self.assertRaises(ValueError):
                validate_final_grasp_scope("final_test", profile)

    def test_new_scope_final_allowed(self):
        validate_final_grasp_scope("final_test", {"grasp_candidate_scope": "all_dynamic_scene_objects"})

    def test_historical_development_diagnostics_remain_allowed(self):
        validate_final_grasp_scope("development", None)


if __name__ == "__main__":
    unittest.main()
