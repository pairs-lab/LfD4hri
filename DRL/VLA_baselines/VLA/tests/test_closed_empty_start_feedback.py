"""Small real-physics TRAIN fixture: command CLOSE is not encoder closure."""
import json
from pathlib import Path
import unittest

import pybullet as p

from VLA.audit_counterfactual_collection import (
    CanonicalSceneResetEnvironment,
    closed_empty_workspace_start,
)
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.tasks import SceneObjectSpec, TaskSpec


class ClosedEmptyStartFeedbackTest(unittest.TestCase):
    def test_canonical_replay_is_physically_closed_empty_and_teacher_reopens(self):
        # Fixed TRAIN-only local fixture, not a benchmark/final-test seed.
        # Apple exercises the near-round planner branch that previously used
        # the current closed aperture as the gripper's maximum open capacity.
        task = TaskSpec(
            task_id=1, name="pick", prompt="Pick the red apple up",
            pick_name="red apple", pick_position=(-.10, -.53, 0.),
            scene_profile="rl_random_env",
            scene_objects=(SceneObjectSpec("green block", (.14, -.38, .03), False, "pick"),),
            config_pick_items=("red apple", "green block"),
        )
        root = Path(__file__).resolve().parents[2]
        profile = json.loads((root / "VLA/profiles/body_contact_v9_wrist65_f80_opposite_front_candidate.json").read_text())
        with BodyGraspEnvironment(profile, image_size=384, required_success_steps=1) as env:
            startup = closed_empty_workspace_start(env, task)
            closes = [action for action in startup if action[-1] > .2]
            self.assertGreaterEqual(len(closes), 5)
            collection = CanonicalSceneResetEnvironment(env, task, startup)
            observation = collection.reset(task)
            self.assertGreaterEqual(env.measured_gripper_closure(), .95)
            self.assertGreaterEqual(p.getJointState(env.gripper.body, 1)[0], .95 * env.gripper.CLOSED_ANGLE)
            self.assertIsNone(env.gripper.grasped_object)
            self.assertFalse(p.getContactPoints(bodyA=env.gripper.body))
            self.assertFalse(env.safety_latched)
            # Startup actions do not count as collected expert transitions.
            self.assertEqual(env.step_count, 0)
            self.assertEqual(env.success_streak, 0)
            self.assertFalse(env._last_success)
            teacher = CollisionPlannedTeacher(env)
            teacher.reset(task)
            action = teacher.predict(observation)
            self.assertEqual(teacher.phase, "approach")
            self.assertEqual(action[-1], -1.)
            self.assertGreater(teacher.plan["open_gap_m"], .088)


if __name__ == "__main__":
    unittest.main()
