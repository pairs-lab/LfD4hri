import unittest
from VLA.adaptive_clearance_teacher import adaptive_midbody_height


class AdaptiveClearanceTest(unittest.TestCase):
    def test_nominal_clearance_still_used_when_possible(self):
        self.assertAlmostEqual(adaptive_midbody_height(.024, .004, .004, .044), .0345)

    def test_preserved_low_profile_scene_remains_inside_body_band(self):
        low, high = .0009811338787834994, .04058972114054745
        height = adaptive_midbody_height(.0245, .004, low, high)
        self.assertLess(height, low+.8*(high-low))
        self.assertGreaterEqual(height, .004+.0275)

    def test_impossible_thin_scene_still_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_midbody_height(.015, .004, .004, .029)

    def test_nonfinite_geometry_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_midbody_height(float("nan"), .004, .004, .044)
