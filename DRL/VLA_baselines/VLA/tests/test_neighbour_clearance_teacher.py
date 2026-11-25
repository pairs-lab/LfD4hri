import unittest
import numpy as np
from unittest.mock import patch
from VLA.neighbour_clearance_teacher import select_clearance_axis, axis_obstacle_clearance, NoFullSupportYaw, ray_sampled_pad_sweep_clear


class NeighbourClearanceTeacherTest(unittest.TestCase):
    def points(self, halfwidth=.02):
        return np.array([[x, y, z] for x in (-halfwidth, halfwidth)
                         for y in (-halfwidth, halfwidth) for z in (0, .04)])

    def test_selects_yaw_away_from_neighbour(self):
        axis, clearance = select_clearance_axis(self.points(), [0, 0, .02],
            [([.05, -.05, 0], [.15, .05, .08])], .0886)
        self.assertGreater(abs(axis[1]), .9)
        self.assertGreater(clearance, 0)

    def test_no_fit_is_rejected(self):
        with self.assertRaises(NoFullSupportYaw):
            select_clearance_axis(self.points(.06), [0, 0, .02], [], .0886)

    def test_no_obstacle_clearance_is_rejected(self):
        with self.assertRaises(ValueError):
            select_clearance_axis(self.points(), [0, 0, .02],
                                  [([-.2, -.2, 0], [.2, .2, .1])], .0886)

    def test_original_cross_section_is_not_allowed_when_pad_centres_collide(self):
        self.assertLess(axis_obstacle_clearance([1, 0], [0, 0, .02],
            [([.04, -.02, 0], [.08, .02, .05])], .0886), 0)

    def test_actual_obstacle_ray_hit_rejects_central_section_proposal(self):
        with patch("VLA.neighbour_clearance_teacher.p.rayTestBatch", return_value=[(42,)]) as ray:
            self.assertFalse(ray_sampled_pad_sweep_clear([1, 0], [0, 0, .02], .0886, {42}, .2, .01))
            self.assertEqual(len(ray.call_args.args[0]), 34)
        with patch("VLA.neighbour_clearance_teacher.p.rayTestBatch", return_value=[(-1,)]):
            self.assertTrue(ray_sampled_pad_sweep_clear([1, 0], [0, 0, .02], .0886, {42}, .2, .01))
