import unittest
import numpy as np
from VLA.lateral_visibility_carry_teacher import side_waypoints


class LateralVisibilityCarryTest(unittest.TestCase):
    def test_moves_sideways_before_advancing_to_hidden_goal(self):
        first, second = side_waypoints([.026, -.696, .2], [.052, -.279, .03], .3)
        np.testing.assert_allclose(first, [-.064, -.696, .3])
        np.testing.assert_allclose(second, [-.064, -.279, .3])

    def test_left_hand_start_retrieves_view_toward_centre(self):
        first, second = side_waypoints([-.1, -.6, .2], [.2, -.3, .03], .3)
        self.assertGreater(first[0], -.1)
        self.assertEqual(first[1], -.6)
        self.assertEqual(first[2], second[2])

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            side_waypoints([float("nan"), 0, 0], [0, 0, 0], .3)
