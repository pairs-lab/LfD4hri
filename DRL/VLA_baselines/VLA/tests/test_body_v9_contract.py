from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np

from VLA.common.body_gripper import BodyContactRobotiq
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.physics_safety import enforce_physics_safety, physics_diagnostics, PhysicsSafetyViolation
from VLA.collision_grasp_planner import wrap_axis_angle
import json

ROOT = Path(__file__).resolve().parents[2]


class BodyV9ContractTest(unittest.TestCase):
    def test_followers_track_actual_actuator_not_requested_angle(self):
        grip = object.__new__(BodyContactRobotiq)
        grip.body, grip.command, grip.grasp_constraint, grip._latched = 123, 1., None, None
        with patch("VLA.common.body_gripper.p.getJointState", return_value=(.2, 0, 0, 0)), patch("VLA.common.body_gripper.p.setJointMotorControl2") as motor:
            grip.apply_motor_control()
        calls = motor.call_args_list
        self.assertAlmostEqual(calls[0].kwargs["targetPosition"], .78)
        for call, sign in zip(calls[1:], grip.SIGNS[1:]):
            self.assertAlmostEqual(call.kwargs["targetPosition"], sign*.2)
        self.assertLessEqual(max(c.kwargs["maxVelocity"] for c in calls), 1.92)

    def test_derived_asset_changes_only_visual_pad_scale(self):
        original = ET.parse(ROOT / "asset/robotiq_2f_85/robotiq_2f_85.urdf").getroot()
        derived = ET.parse(ROOT / "VLA/assets/body_grasp_v9/robotiq_2f_85/robotiq_2f_85.urdf").getroot()
        for old, new in zip(original.findall("link"), derived.findall("link"), strict=True):
            for kind in ("collision", "inertial"):
                a, b = old.find(kind), new.find(kind)
                self.assertEqual(ET.tostring(a) if a is not None else None, ET.tostring(b) if b is not None else None)
        for old, new in zip(original.findall("joint"), derived.findall("joint"), strict=True):
            self.assertEqual(ET.tostring(old), ET.tostring(new))
        for side in ("left", "right"):
            pad = derived.find(f"link[@name='robotiq_2f_85_{side}_pad']")
            self.assertEqual(pad.find("visual/geometry/mesh").get("scale"), pad.find("collision/geometry/mesh").get("scale"))

    def test_unsafe_ground_contact_raises_before_more_control(self):
        contact = (0, 0, 0, 0, 0, 0, 0, 0, -.002)
        env = SimpleNamespace(ground_bodies=[0], gripper=SimpleNamespace(body=1))
        with patch("VLA.common.physics_safety.p.getClosestPoints", return_value=[contact]):
            with self.assertRaises(PhysicsSafetyViolation):
                enforce_physics_safety(env)

    def test_pre_rollback_diagnostics_preserve_offending_contact(self):
        env = SimpleNamespace(ground_bodies=[0], gripper=SimpleNamespace(
            body=1, JOINTS=BodyContactRobotiq.JOINTS, SIGNS=BodyContactRobotiq.SIGNS, command=-1.))
        states = [(q, .1, 0, 0) for q in (.1, -.1, .1, .1, .051, .1)]
        contact = (0, 1, 3, 9, -1, (1., 2., 3.), (0., 0., 0.), (0., 0., 1.), -.0002, 180.)
        with patch("VLA.common.physics_safety.p.getJointStates", return_value=states), \
             patch("VLA.common.physics_safety.p.getContactPoints", return_value=[contact]), \
             patch("VLA.common.physics_safety.p.getClosestPoints", return_value=[]):
            diagnostics = physics_diagnostics(env)
        self.assertAlmostEqual(diagnostics["follower_errors_rad"][3], .151)
        self.assertEqual(diagnostics["requested_gripper_command"], -1.)
        self.assertEqual(diagnostics["contacts"][0]["normal_force_n"], 180.)

    def test_explicit_dispatch_and_planar_axis_periodicity(self):
        from VLA.common.body_environment import BodyGraspEnvironment
        from VLA.common.observation_profile import ProfiledManipulationEnv
        self.assertIs(environment_class_for_profile({"gripper_controller": "body_contact_v9"}), BodyGraspEnvironment)
        self.assertIs(environment_class_for_profile({}), ProfiledManipulationEnv)
        with self.assertRaises(ValueError):
            environment_class_for_profile({"gripper_controller": "unknown"})
        self.assertAlmostEqual(wrap_axis_angle(.3), wrap_axis_angle(.3 + np.pi))

    def test_v9_rejects_task_identity_filtered_gripper(self):
        from VLA.common.body_environment import BodyGraspEnvironment
        profile = json.loads((ROOT/"VLA/profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())
        profile["grasp_candidate_scope"] = "selected_target_legacy"
        with self.assertRaisesRegex(ValueError, "task-independent"):
            BodyGraspEnvironment(profile, render_observations=False)

    def test_safety_abort_returns_standard_benchmark_fields_and_latches(self):
        from VLA.common.body_environment import BodyGraspEnvironment
        from VLA.common.environment import ManipulationEnv
        from VLA.common.observation_profile import ProfiledManipulationEnv
        from VLA.common.tasks import TASKS
        profile = json.loads((ROOT/"VLA/profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())
        with BodyGraspEnvironment(profile, render_observations=False, image_size=16) as env:
            env.reset(TASKS["touch"])
            with patch.object(ProfiledManipulationEnv, "step", side_effect=PhysicsSafetyViolation("test abort")):
                _, _, done, info = env.step(np.zeros(7, dtype=np.float32))
            self.assertTrue(done)
            self.assertTrue(info["rollback_restored"])
            for key in ("success", "raw_success", "strict_raw_success", "semantic_raw_success", "distance_only_success"):
                self.assertIs(info[key], False)
            for key in ("pick_lift_m", "goal_xy_distance_m", "goal_vertical_offset_m", "distance_threshold", "object_threshold"):
                self.assertIn(key, info)
            self.assertIn("joint_positions_rad", info["safety_diagnostics_before_rollback"])
            with self.assertRaisesRegex(PhysicsSafetyViolation, "Reset required"):
                env.step(np.zeros(7, dtype=np.float32))
