import copy
import unittest

import numpy as np
import pybullet as p

from VLA.audit_wrist_home import measure, projected_points
from VLA.common.environment import ManipulationEnv
from VLA.common.observation_profile import (ProfiledManipulationEnv, baseline_profile,
                                          profile_hash, validate_profile)
from VLA.common.tasks import TASKS


class TestObservationProfile(unittest.TestCase):
    def test_profile_hash_binds_home_and_camera(self):
        first = baseline_profile()
        for section in ("home", "camera"):
            second = copy.deepcopy(first)
            if section == "home":
                second["home_joints"][0] += .01
            else:
                second["cameras"]["wrist"]["fov_degrees"] += 1
            self.assertNotEqual(profile_hash(first), profile_hash(second))

    def test_motor_command_is_recorded_separately_from_feedback(self):
        with ProfiledManipulationEnv(baseline_profile(), image_size=16) as env:
            env.reset(TASKS["pick"])
            observed, _, _, _ = env.step(np.asarray([.02, -.015, -.03, 0, 0, 0, -1], dtype=np.float32))
            self.assertEqual(env.requested_arm_targets.shape, (6,))
            self.assertEqual(env.requested_arm_targets.dtype, np.float32)
            self.assertTrue(np.isfinite(env.requested_arm_targets).all())
            self.assertGreater(float(np.max(np.abs(env.requested_arm_targets-observed.joint_positions))), 1e-7)

    def test_invalid_profiles(self):
        for value in (float("nan"), 180):
            profile = baseline_profile()
            profile["cameras"]["wrist"]["fov_degrees"] = value
            with self.assertRaises(ValueError):
                validate_profile(profile)
        profile = baseline_profile()
        profile["wrist_frame"] = "object_tracking"
        with self.assertRaises(ValueError):
            validate_profile(profile)
        profile = baseline_profile()
        profile["front_camera_role"] = "policy_workspace"
        with self.assertRaises(ValueError):
            validate_profile(profile)

    def test_video_only_overview_does_not_change_policy_observation(self):
        profile = baseline_profile()
        profile["front_camera_role"] = "policy_workspace"
        profile["cameras"]["overview"] = copy.deepcopy(profile["cameras"]["front"])
        with ProfiledManipulationEnv(profile, image_size=224) as env:
            observation = env.reset(TASKS["pick"])
            env.render("overview")
            self.assertTrue(np.array_equal(observation.front_rgb, env.render("front")))
            self.assertTrue(np.array_equal(observation.wrist_rgb, env.render("wrist")))
            self.assertFalse(hasattr(observation, "overview_rgb"))

    def test_geometry_projection(self):
        view = p.computeViewMatrix((0, 0, 1), (0, 0, 0), (0, 1, 0))
        proj = p.computeProjectionMatrixFOV(90, 1, .01, 3)
        xy, inside = projected_points([(0, 0, 0), (.9, 0, 0), (1.2, 0, 0), (0, 0, 2)], view, proj, 224)
        self.assertTrue(np.allclose(xy[0], (112, 112)))
        self.assertEqual(inside, [True, True, False, False])

    def test_legacy_profile_and_occlusion_restore(self):
        # Existing UR5 tool inertial frame is identity. Changing to explicit
        # link-frame transforms must reproduce its current observations.
        with ManipulationEnv(image_size=224) as env:
            old = env.reset(TASKS["pick"])
        with ProfiledManipulationEnv(baseline_profile(), image_size=224) as env:
            new = env.reset(TASKS["pick"])
            self.assertTrue(np.array_equal(old.front_rgb, new.front_rgb))
            self.assertTrue(np.array_equal(old.wrist_rgb, new.wrist_rgb))
            self.assertTrue(np.array_equal(old.joint_positions, new.joint_positions))
            before = [p.getBasePositionAndOrientation(body) for body in (env.robot_id, env.gripper.body, env.pick_id)]
            measure(env, "wrist")
            after = [p.getBasePositionAndOrientation(body) for body in (env.robot_id, env.gripper.body, env.pick_id)]
            for old_state, new_state in zip(before, after):
                self.assertTrue(np.allclose(np.r_[old_state[0], old_state[1]],
                                            np.r_[new_state[0], new_state[1]], atol=1e-14, rtol=0))
            self.assertTrue(np.array_equal(env.render("wrist"), new.wrist_rgb))


if __name__ == "__main__":
    unittest.main()
