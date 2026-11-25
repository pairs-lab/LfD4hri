from dataclasses import replace
import unittest

from VLA.audit_counterfactual_collection import CanonicalSceneResetEnvironment
from VLA.common.observation_profile import ProfiledManipulationEnv, baseline_profile, validate_profile
from VLA.common.tasks import SceneObjectSpec, TASKS


class GraspCandidateScopeTest(unittest.TestCase):
    def task(self):
        return replace(TASKS["pick"], scene_objects=(
            SceneObjectSpec("brown block", (.2, -.4, .03), False, "pick"),
            SceneObjectSpec("gray bowl", (-.2, -.4, 0), True, "place")))

    def test_new_scope_includes_wrong_movable_object_excludes_fixed(self):
        profile = {**baseline_profile(), "grasp_candidate_scope": "all_dynamic_scene_objects"}
        with ProfiledManipulationEnv(profile, image_size=16, render_observations=False) as env:
            task = self.task()
            CanonicalSceneResetEnvironment(env, task).reset(task)
            self.assertIn(env.object_ids[task.pick_name], env.gripper.object_ids)
            self.assertIn(env.object_ids["brown block"], env.gripper.object_ids)
            self.assertNotIn(env.object_ids["gray bowl"], env.gripper.object_ids)

    def test_historical_profile_keeps_historical_contract(self):
        with ProfiledManipulationEnv(baseline_profile(), image_size=16, render_observations=False) as env:
            env.reset(self.task())
            self.assertEqual(env.gripper.object_ids, [env.pick_id])

    def test_unknown_scope_rejected(self):
        with self.assertRaises(ValueError):
            validate_profile({**baseline_profile(), "grasp_candidate_scope": "magic_target"})


if __name__ == "__main__":
    unittest.main()
