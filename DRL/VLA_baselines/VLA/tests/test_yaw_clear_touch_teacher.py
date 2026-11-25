import unittest
import numpy as np

from VLA.yaw_clear_touch_teacher import rotated_aabb, yaw_touch_proposals


class YawClearTouchTests(unittest.TestCase):
    def test_rotated_box_keeps_all_corners_and_height(self):
        low, high = rotated_aabb([1, -.1, .2], [2, .1, .3], [0, 0, 0], np.pi/2)
        np.testing.assert_allclose(low, [-.1, 1, .2], atol=1e-12)
        np.testing.assert_allclose(high, [.1, 2, .3], atol=1e-12)
        with self.assertRaises(ValueError): rotated_aabb([1, 0, 0], [0, 0, 0], [0, 0, 0], 0)

    def geometry(self):
        fingers = [([- .06, -.51, 0], [-.04, -.49, .04]), ([.04, -.51, 0], [.06, -.49, .04])]
        vertices = [np.array([low, high]) for low, high in fingers]
        return fingers, vertices

    def test_zero_yaw_is_preserved_when_original_proposal_clears(self):
        fingers, vertices = self.geometry()
        rows, selected = yaw_touch_proposals(fingers, [], vertices, [0, -.5, .1], [0, -.5, .02], .2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(selected["yaw_world_rad"], 0)

    def test_neighbours_on_both_sides_require_rotation_not_scene_skip(self):
        fingers, vertices = self.geometry()
        obstacles = [([-.11, -.52, 0], [-.075, -.48, .06]), ([.075, -.52, 0], [.11, -.48, .06])]
        rows, selected = yaw_touch_proposals(fingers, obstacles, vertices, [0, -.5, .1], [0, -.5, .02], .2)
        self.assertTrue(all(r["sweep_aabb_clearance_m"] < .004 for r in rows[:2]))
        self.assertNotEqual(selected["yaw_world_rad"], 0)
        self.assertGreater(selected["sweep_aabb_clearance_m"], .004)
        self.assertIn(selected["pad_index"], (0, 1))

    def test_no_safe_grid_candidate_fails_closed(self):
        fingers, vertices = self.geometry()
        with self.assertRaisesRegex(ValueError, "declared yaw grid"):
            yaw_touch_proposals(fingers, [([-.5, -.8, 0], [.5, .1, 1])], vertices,
                                [0, -.5, .1], [0, -.5, .02], .2)


if __name__ == "__main__":
    unittest.main()
