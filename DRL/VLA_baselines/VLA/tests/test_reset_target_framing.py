import unittest
import numpy as np

from VLA.audit_reset_target_framing import framing, summarize


class ResetTargetFramingTests(unittest.TestCase):
    def mask(self, edge=False):
        mask = np.zeros((20, 20), dtype=bool)
        first_row = 0 if edge else 5
        mask[first_row:first_row+5, 5:10] = True
        return mask

    def test_empty_is_not_a_clear_view(self):
        view = framing(np.zeros((20, 20), bool))
        self.assertFalse(view["visible_not_at_edge"])
        self.assertFalse(summarize({"front": view, "wrist": view})["any_visible_25px"])

    def test_border_risk_is_recorded_without_claiming_failure(self):
        view = framing(self.mask(True))
        self.assertTrue(view["edge_risk"])
        self.assertEqual(view["pixels"], 25)
        self.assertTrue(summarize({"wrist": view})["visible_edge_risk_without_interior_alternative"])

    def test_actual_interior_other_camera_complements_edge(self):
        flags = summarize({"front": framing(self.mask()), "wrist": framing(self.mask(True))})
        self.assertTrue(flags["any_visible_not_at_edge"])
        self.assertFalse(flags["visible_edge_risk_without_interior_alternative"])

    def test_invalid_mask_is_rejected(self):
        for mask in (np.zeros((20, 20), np.uint8), np.zeros(20, bool)):
            with self.assertRaises(ValueError):
                framing(mask)


if __name__ == "__main__":
    unittest.main()
