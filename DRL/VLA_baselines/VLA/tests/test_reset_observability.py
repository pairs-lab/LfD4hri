import unittest

from VLA.audit_finetune_data import reset_observability_failures, reset_view_quality


class ResetObservabilityTests(unittest.TestCase):
    def quality(self, **changes):
        return dict({"raw_pixels": 100, "pixels_224": 48, "minimum_dimension_224": 7, "contrast": 25.}, **changes)

    def sample(self, recovery=True, front=None, wrist=None):
        return {"file": "sample", "closed_empty_recovery": recovery,
                "views": {"front": front or self.quality(raw_pixels=0, pixels_224=0, minimum_dimension_224=0, contrast=0.),
                          "wrist": wrist or self.quality()}}

    def test_default_front_gate_remains(self):
        self.assertTrue(reset_observability_failures([self.sample()]))

    def test_dual_model_can_use_one_good_wrist_at_real_recovery_reset(self):
        self.assertFalse(reset_observability_failures([self.sample()], dual_recovery=True))

    def test_ordinary_reset_still_requires_front(self):
        self.assertTrue(reset_observability_failures([self.sample(recovery=False)], dual_recovery=True))

    def test_size_and_contrast_cannot_be_pooled_across_cameras(self):
        row = self.sample(front=self.quality(contrast=0.), wrist=self.quality(pixels_224=47))
        self.assertTrue(reset_observability_failures([row], dual_recovery=True))

    def test_all_original_cutoffs_unchanged(self):
        self.assertTrue(reset_view_quality(self.quality()))
        for change in ({"raw_pixels": 24}, {"pixels_224": 47}, {"minimum_dimension_224": 6}, {"contrast": 24.99}):
            self.assertFalse(reset_view_quality(self.quality(**change)))


if __name__ == "__main__":
    unittest.main()
