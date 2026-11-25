import unittest
from VLA.raised_carry_visibility_teacher import raised_carry_height


class RaisedCarryTest(unittest.TestCase):
    def test_preserves_original_floor_clearance_and_adds_height(self):
        self.assertAlmostEqual(raised_carry_height(.1, .07, .15), .30)

    def test_already_clear_object_never_lowered_during_raise(self):
        self.assertAlmostEqual(raised_carry_height(.4, .37, .1), .48)

    def test_invalid_extra_clearance_is_not_accepted(self):
        for value in (-.1, float("nan")):
            with self.assertRaises(ValueError):
                raised_carry_height(.1, .07, .15, value)


if __name__ == "__main__":
    unittest.main()
