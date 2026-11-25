import unittest
from VLA.geometry_clearance_teacher import geometry_midbody_height


class GeometryClearanceTest(unittest.TestCase):
    def test_normal_block_retains_nominal_extra_clearance(self):
        self.assertAlmostEqual(geometry_midbody_height(.024, .004, .004, .044, .027), .0345)

    def test_low_box_is_not_rejected_by_an_arbitrary_extra_margin(self):
        lo, hi = .001868817391496927, .03874865579711728
        value = geometry_midbody_height(.0245, .004, lo, hi, .030)
        self.assertGreaterEqual(value, .030)
        self.assertLess(value, lo+.8*(hi-lo))

    def test_actual_floor_and_body_band_cannot_be_relaxed(self):
        with self.assertRaisesRegex(ValueError, "whole-gripper floor"):
            geometry_midbody_height(.024, .004, .004, .029, .028)
        with self.assertRaises(ValueError):
            geometry_midbody_height(float("nan"), .004, .004, .044, .027)
