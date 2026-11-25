import unittest
from VLA.clearance_correction_teacher import safe_midbody_height


class ClearanceCorrectionTeacherTest(unittest.TestCase):
    def test_low_block_moves_up_but_stays_midbody(self):
        height = safe_midbody_height(.024, .004, .004, .044)
        self.assertAlmostEqual(height, .0345)
        self.assertLessEqual((height-.004)/.040, .80)

    def test_normal_tall_object_point_unchanged(self):
        self.assertEqual(safe_midbody_height(.055, .004, .004, .10), .055)

    def test_unsafe_impossible_contact_band_fails_not_clipped_to_object_top(self):
        with self.assertRaises(ValueError):
            safe_midbody_height(.015, .004, .004, .029)
