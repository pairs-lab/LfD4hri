import unittest
from VLA.settling_baseline import natural_settling_check


class SettlingBaselineTests(unittest.TestCase):
    def poses(self, banana_x):
        return {"banana": [[banana_x, -.4, .02], [0, 0, 0, 1]],
                "egg": [[0, -.6, .03], [0, 0, 0, 1]]}

    def test_natural_millimetre_motion_not_robot_specific(self):
        result = natural_settling_check(self.poses(.0014038), self.poses(.0014037))
        self.assertTrue(result["passed"])
        self.assertLess(result["object_differences"]["banana"]["translation_m"], 1e-5)

    def test_robot_caused_motion_or_missing_scene_body_fails_closed(self):
        self.assertFalse(natural_settling_check(self.poses(.003), self.poses(0))["passed"])
        with self.assertRaises(ValueError): natural_settling_check(self.poses(0), {})

    def test_rotation_and_nonfinite_pose_are_checked(self):
        a = self.poses(0); a["banana"][1] = [0, 0, .01, .99995]
        self.assertFalse(natural_settling_check(a, self.poses(0))["passed"])
        a["banana"][0][0] = float("nan")
        with self.assertRaises(ValueError): natural_settling_check(a, self.poses(0))


if __name__ == "__main__":
    unittest.main()
