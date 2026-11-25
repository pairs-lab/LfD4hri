import unittest
from types import SimpleNamespace

import numpy as np

from VLA.audit_committed_sensor_alignment import sensor_errors


class CommittedSensorAlignmentTests(unittest.TestCase):
    def make_case(self):
        observation = SimpleNamespace(
            front_rgb=np.full((2, 2, 3), 255, dtype=np.uint8),
            wrist_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            joint_positions=np.arange(6, dtype=np.float32),
            ee_position=np.array([0., -.565, .695]),
            ee_rotation_6d=np.array([1., 0., 0., 0., 1., 0.]),
        )
        data = {name: np.stack([value.copy(), value.copy()])
                for name, value in vars(observation).items()}
        data["gripper_closure"] = np.array([.25, .25])
        env = SimpleNamespace(normalized_gripper_closure=lambda: .25)
        return observation, env, data

    def test_exact_pre_action_sensors_have_zero_error(self):
        observation, env, data = self.make_case()
        self.assertEqual(set(sensor_errors(observation, env, data, 0)),
                         set(data))
        self.assertTrue(all(value == 0. for value in
                            sensor_errors(observation, env, data, 0).values()))

    def test_rgb_comparison_does_not_wrap_unsigned_pixel_subtraction(self):
        observation, env, data = self.make_case()
        data["front_rgb"][1, 0, 0, 0] = 0
        data["wrist_rgb"][1, 0, 0, 0] = 255
        errors = sensor_errors(observation, env, data, 1)
        self.assertEqual(errors["front_rgb"], 255.)
        self.assertEqual(errors["wrist_rgb"], 255.)
        self.assertEqual(sensor_errors(observation, env, data, 0)["front_rgb"], 0.)

    def test_measured_joint_mount_rotation_and_aperture_are_independent(self):
        observation, env, data = self.make_case()
        for field in ("joint_positions", "ee_position", "ee_rotation_6d"):
            with self.subTest(field=field):
                changed = {name: value.copy() for name, value in data.items()}
                changed[field][1, 0] += .1
                errors = sensor_errors(observation, env, changed, 1)
                self.assertAlmostEqual(errors[field], .1, places=6)
                self.assertEqual(errors["gripper_closure"], 0.)
                self.assertEqual(errors["front_rgb"], 0.)
        data["gripper_closure"][1] = .75
        self.assertEqual(sensor_errors(observation, env, data, 1)["gripper_closure"], .5)


if __name__ == "__main__":
    unittest.main()
