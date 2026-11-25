import unittest

import numpy as np

from VLA.recollect_native_touch_failure import require_same_reset


class NativeTouchRecollectionTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"front_rgb": np.zeros((3, 3, 3), dtype=np.uint8),
                          "wrist_rgb": np.ones((3, 3, 3), dtype=np.uint8),
                          "joint_positions": np.arange(6, dtype=np.float32),
                          "gripper_closure": np.float32(.4)}
        self.arrays = {key: np.asarray([value]) for key, value in self.reference.items()}

    def test_exact_actual_reset_required(self):
        self.assertTrue(all(value == 0 for value in require_same_reset(self.arrays, self.reference).values()))

    def test_each_camera_and_proprio_field_is_checked(self):
        for key in self.arrays:
            with self.subTest(key=key):
                changed = {k: v.copy() for k, v in self.arrays.items()}
                changed[key].flat[0] += 1
                with self.assertRaises(ValueError):
                    require_same_reset(changed, self.reference)


if __name__ == "__main__":
    unittest.main()
