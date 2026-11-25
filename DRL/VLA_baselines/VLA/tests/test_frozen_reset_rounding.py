import unittest
import numpy as np
from VLA.frozen_reset_rounding import compare_frozen_reset


class FrozenResetRoundingTest(unittest.TestCase):
    def data(self):
        return {"front_rgb": np.zeros((1, 2, 2, 3), np.uint8), "wrist_rgb": np.zeros((1, 2, 2, 3), np.uint8),
                "joint_positions": np.zeros((1, 6), np.float32), "gripper_closure": np.array([.8])}

    def test_one_machine_ulp_is_not_physical_reset_drift(self):
        old, new = self.data(), self.data()
        new["gripper_closure"][0] = np.nextafter(old["gripper_closure"][0], 1.)
        result = compare_frozen_reset(new, old)
        self.assertTrue(result["passed"])
        self.assertFalse(result["all_fields_bitwise_exact"])
        self.assertEqual(result["pixels_and_joints_tolerance"], 0)

    def test_aperture_beyond_machine_roundoff_is_rejected(self):
        old, new = self.data(), self.data()
        new["gripper_closure"][0] += 1e-14
        with self.assertRaises(ValueError):
            compare_frozen_reset(new, old)

    def test_single_pixel_change_is_rejected(self):
        old, new = self.data(), self.data()
        new["wrist_rgb"][0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            compare_frozen_reset(new, old)

    def test_single_joint_change_is_rejected(self):
        old, new = self.data(), self.data()
        new["joint_positions"][0, 0] = 1e-9
        with self.assertRaises(ValueError):
            compare_frozen_reset(new, old)


if __name__ == "__main__":
    unittest.main()
