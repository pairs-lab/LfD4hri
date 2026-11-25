import unittest
import numpy as np
from VLA.ray_refined_clearance_teacher import full_support_options


class RayRefinedProposalTest(unittest.TestCase):
    def points(self, width):
        return np.array([[x, y, z] for x in (-width, width) for y in (-width, width) for z in (0, .04)])

    def test_only_width_valid_axes_are_returned(self):
        points = self.points(.02)
        axes = full_support_options(points, [0, 0, .02], .0886)
        self.assertEqual(len(axes), 72)
        for axis in axes:
            self.assertLessEqual(2*np.abs(points[:, :2]@axis).max(), .0886-.004)

    def test_large_object_and_invalid_vertices_do_not_get_a_free_yaw(self):
        self.assertEqual(full_support_options(self.points(.06), [0, 0, .02], .0886), [])
        with self.assertRaises(ValueError):
            full_support_options([[np.nan, 0, 0]], [0, 0, 0], .0886)
