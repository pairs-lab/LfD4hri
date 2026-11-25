"""Focused geometry and real-physics TRAIN-only near-miss regression."""
import json
from pathlib import Path
import unittest

import numpy as np
import pybullet as p

from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.tasks import SceneObjectSpec, TaskSpec
from VLA.near_object_recovery_start import (
    closed_empty_near_object_start, near_object_candidates,
    validated_near_object_reset_environment,
)


class NearObjectGeometryTest(unittest.TestCase):
    def test_candidates_clear_entire_open_gripper_and_are_near_body(self):
        bounds = [("green block", np.array([-.020, -.520, .004]),
                   np.array([.020, -.480, .044]))]
        offsets = (np.array([-.069, -.013, -.025]), np.array([.069, .013, .130]))
        candidates = near_object_candidates(bounds, offsets, .004)
        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            target = np.asarray(candidate["target_tcp"])
            self.assertLessEqual(candidate["planned_anchor_aabb_distance_m"], .11)
            self.assertGreaterEqual(target[2]+offsets[0][2], .014-1e-12)
            self.assertLessEqual(target[2], bounds[0][2][2]+.020)

    def test_crowded_scene_is_rejected_not_moved(self):
        bounds = [("huge obstruction", np.array([-.6, -.9, 0]),
                   np.array([.6, -.1, .5]))]
        self.assertEqual(near_object_candidates(bounds, (np.array([-.069, -.013, -.025]),
                                                       np.array([.069, .013, .130])), .004), [])


class NearObjectPhysicsTest(unittest.TestCase):
    def test_explicit_bounded_transit_keeps_physical_camera_gates(self):
        task = TaskSpec(task_id=2, name="pick", prompt="Pick the red apple up",
            pick_name="red apple", pick_position=(-.10, -.53, 0.), scene_profile="rl_random_env",
            scene_objects=(SceneObjectSpec("green block", (.14, -.38, .03), False, "pick"),),
            config_pick_items=("red apple", "green block"))
        root = Path(__file__).resolve().parents[2]
        profile = json.loads((root/"VLA/profiles/body_contact_v9_wrist65_f80_opposite_front_candidate.json").read_text())
        with BodyGraspEnvironment(profile, image_size=384, required_success_steps=1) as env:
            actions, evidence = closed_empty_near_object_start(env, task,
                bounded_transit=True, expanded_candidates=True)
            self.assertTrue(evidence["bounded_object_clear_transit"])
            self.assertTrue(evidence["expanded_offline_proposals"])
            self.assertAlmostEqual(evidence["transit_pad_tcp_height_m"], .24)
            self.assertGreaterEqual(evidence["actual_state"]["surface_clearance_m"], .003)
            self.assertTrue(all(v["passes"] for v in evidence["visibility"]["front"].values()))
            self.assertTrue(evidence["visibility"]["wrist"][evidence["anchor_name"]]["passes"])
            proxy = validated_near_object_reset_environment(env, task, actions, evidence)
            proxy.reset(task)
            self.assertIsNone(env.gripper.grasped_object)
            self.assertFalse(p.getContactPoints(bodyA=env.gripper.body))

    def test_actual_closed_empty_replay_reopens_before_any_verified_grasp(self):
        task = TaskSpec(task_id=1, name="pick", prompt="Pick the red apple up",
            pick_name="red apple", pick_position=(-.10, -.53, 0.), scene_profile="rl_random_env",
            scene_objects=(SceneObjectSpec("green block", (.14, -.38, .03), False, "pick"),),
            config_pick_items=("red apple", "green block"))
        root = Path(__file__).resolve().parents[2]
        profile = json.loads((root/"VLA/profiles/body_contact_v9_wrist65_f80_opposite_front_candidate.json").read_text())
        with BodyGraspEnvironment(profile, image_size=384, required_success_steps=1) as env:
            actions, evidence = closed_empty_near_object_start(env, task)
            self.assertGreaterEqual(evidence["close_control_count"], 5)
            self.assertFalse(evidence["startup_controls_are_training_labels"])
            self.assertFalse(evidence["source_object_poses_modified"])
            self.assertIn(evidence["anchor_name"], task.config_pick_items)
            proxy = validated_near_object_reset_environment(env, task, actions, evidence)
            observation = proxy.reset(task)
            actual = proxy.near_object_recovery_replay_evidence["actual_state"]
            self.assertGreaterEqual(actual["actual_closure"], .95)
            self.assertGreaterEqual(p.getJointState(env.gripper.body, 1)[0], .95*env.gripper.CLOSED_ANGLE)
            self.assertIsNone(env.gripper.grasped_object)
            self.assertFalse(p.getContactPoints(bodyA=env.gripper.body))
            self.assertEqual(env.step_count, 0)
            teacher = CollisionPlannedTeacher(env)
            teacher.reset(task)
            first = teacher.predict(observation)
            self.assertEqual(first[-1], -1)
            _, _, done, info = proxy.step(first)
            self.assertFalse(done)
            self.assertIsNone(info.get("safety_violation"))
            self.assertEqual(env.commanded_gripper_closure(), 0)
            self.assertIsNone(env.gripper.grasped_object)


if __name__ == "__main__":
    unittest.main()
