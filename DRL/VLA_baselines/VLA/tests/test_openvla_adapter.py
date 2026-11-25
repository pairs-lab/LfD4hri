from types import SimpleNamespace
import unittest

import numpy as np

from VLA.openvla.adapter import OpenVLAPolicy


class Inputs(dict):
    def to(self, *args, **kwargs):
        return self


class OpenVLAActionTest(unittest.TestCase):
    def policy(self, native):
        policy = OpenVLAPolicy()
        policy.processor = lambda *args: Inputs()
        policy.model = SimpleNamespace(predict_action=lambda **kwargs: native)
        return policy

    def observation(self):
        return SimpleNamespace(front_rgb=np.zeros((8, 8, 3), dtype=np.uint8), prompt="Pick the red block")

    def test_open_closed_conversion(self):
        for grip, expected in ((1, -1), (0, 1)):
            actual = self.policy([.01, .02, .03, 0, 0, 0, grip]).predict(self.observation())
            np.testing.assert_allclose(actual, [.01, .02, .03, 0, 0, 0, expected])

    def test_nonfinite_and_wrong_dimension_rejected(self):
        for native in ([0] * 6, [0] * 8, [0] * 6 + [float("nan")], [float("inf")] + [0] * 6):
            with self.subTest(native=native), self.assertRaises(ValueError):
                self.policy(native).predict(self.observation())


if __name__ == "__main__":
    unittest.main()
