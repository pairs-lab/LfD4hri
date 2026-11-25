import unittest
from VLA.audit_touch_contact_quality import contact_classification


class TouchContactQualityTest(unittest.TestCase):
    def test_cached_positive_gap_is_not_force_verified(self):
        self.assertEqual(contact_classification(True, .0011, 0.),
                         "positive_gap_or_no_surface_without_positive_force")

    def test_failure_cannot_be_promoted_by_contact_or_proximity(self):
        self.assertEqual(contact_classification(False, -.001, 3.), "strict_failure")

    def test_collision_and_force_are_distinct(self):
        self.assertEqual(contact_classification(True, 0., 0.), "nonpositive_gap_without_positive_force")
        self.assertEqual(contact_classification(True, -.001, 0.), "nonpositive_gap_without_positive_force")
        self.assertEqual(contact_classification(True, .001, .01), "positive_force_at_first_hit")

    def test_no_surface_cannot_be_called_physical_contact(self):
        self.assertEqual(contact_classification(True, None, 0.),
                         "positive_gap_or_no_surface_without_positive_force")
