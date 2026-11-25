"""Action-statistics edge cases without loading neural weights."""
import unittest

import numpy as np

from VLA.openvla.finetune import PyBulletOpenVLADataset


class OpenVLAConstantChannelTest(unittest.TestCase):
    def dataset(self, gripper):
        dataset = PyBulletOpenVLADataset.__new__(PyBulletOpenVLADataset)
        dataset.low = np.array([0] * 6 + [gripper], dtype=np.float32)
        dataset.high = dataset.low.copy()
        dataset.mask = np.array([True] * 6 + [False])
        return dataset

    def test_all_open_touch_labels_remain_open(self):
        actual = self.dataset(1).normalize_action(np.array([0] * 6 + [1]))
        np.testing.assert_array_equal(actual, [0] * 6 + [1])

    def test_all_closed_labels_remain_closed(self):
        actual = self.dataset(0).normalize_action(np.zeros(7))
        np.testing.assert_array_equal(actual, np.zeros(7))

    def test_constant_continuous_dimensions_encode_zero(self):
        dataset = self.dataset(1)
        dataset.low[:6] = dataset.high[:6] = .3
        actual = dataset.normalize_action(np.array([.3] * 6 + [1]))
        np.testing.assert_array_equal(actual, [0] * 6 + [1])


if __name__ == "__main__":
    unittest.main()
