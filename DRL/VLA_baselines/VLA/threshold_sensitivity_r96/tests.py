"""Tests for the analysis only; no model/plant/protocol mutation."""
import unittest
from .analyze import reduced_thresholds, predicate, forecast, geometry


OLD = {"touch_distance_m": .04, "pick_min_lift_m": .05,
    "move_xy_radius_m": .08, "move_vertical_below_m": .03, "move_vertical_above_m": .2,
    "pick_place_xy_radius_m": .08, "pick_place_vertical_below_m": .06, "pick_place_vertical_above_m": .1}


class Tests(unittest.TestCase):
    def test_literal_all_eight_distances(self):
        new = reduced_thresholds(OLD)
        self.assertEqual(new["pick_min_lift_m"], .04)
        self.assertEqual(new["move_vertical_below_m"], .02)
        self.assertEqual(new["pick_place_vertical_above_m"], .09)
        for key in OLD:
            self.assertAlmostEqual(OLD[key]-new[key], .01)

    def test_contact_branch_is_not_deleted(self):
        self.assertTrue(predicate("touch", {"distance": .2}, reduced_thresholds(OLD), True))
        self.assertFalse(predicate("touch", {"distance": .035}, reduced_thresholds(OLD), False))

    def test_pick_easier_but_grasp_still_required(self):
        v = {"lift": .045, "bilateral_correct_hold": True}
        self.assertFalse(predicate("pick", v, OLD))
        self.assertTrue(predicate("pick", v, reduced_thresholds(OLD)))
        v["bilateral_correct_hold"] = False
        self.assertFalse(predicate("pick", v, reduced_thresholds(OLD)))

    def test_vertical_lower_band_tightens_towards_zero(self):
        v = {"xy": .01, "vertical": -.025}
        self.assertTrue(predicate("move", v, OLD))
        self.assertFalse(predicate("move", v, reduced_thresholds(OLD)))

    def test_forecast_mean_exact_and_not_measured(self):
        f = forecast(120)
        self.assertAlmostEqual(f["expected_five_cohort_mean_percent"], 100*(120+800*120.5/201)/1000)
        self.assertFalse(f["five_measured_cohorts"])
        self.assertGreater(f["median_predictive_sample_std_percentage_points"], 0)
        self.assertLess(f["predictive_mean_95_percent"][0], 60)
        self.assertGreater(f["predictive_mean_95_percent"][1], 60)

    def test_logged_goal_is_not_place_reference(self):
        # Bottom=.04, place=.10, reward_target=.25, COM=.06, logged_goal=.27.
        # Correct semantic vertical error is -.06, NOT COM - goal = -.21.
        raw = [0.] * 64
        raw[26] = .04
        step = {"after": {"object_center_xyz_m": [0., 0., .06],
            "object_base_xyz_m": [0., 0., .06], "goal_xyz_m": [0., 0., .27],
            "pad_grasp_center_xyz_m": [0., 0., .10], "correct_object_held": False},
            "post_action_scene64": raw}
        values = geometry("move", step, .04)
        self.assertAlmostEqual(values["vertical"], -.06)
        self.assertAlmostEqual(values["lift"], 0.)
        self.assertAlmostEqual(values["distance"], .02)

    def test_stricter_region_is_a_subset(self):
        new = reduced_thresholds(OLD)
        for task in ("move", "pick_place"):
            for xy in (0., .069, .075, .081):
                for z in (-.07, -.055, -.025, 0., .095, .195, .205):
                    values = {"xy": xy, "vertical": z}
                    if predicate(task, values, new):
                        self.assertTrue(predicate(task, values, OLD))


if __name__ == "__main__":
    unittest.main()
