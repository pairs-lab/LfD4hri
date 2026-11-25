import unittest
import numpy as np
from VLA.audit_target_mask_coverage_r36 import summarize


class CoverageTests(unittest.TestCase):
    def test_full_current_views_include_empty_frames_without_hiding_them(self):
        front=np.zeros((2,384,384),bool);wrist=front.copy();wrist[0,10:20,10:20]=True
        result=summarize(front,wrist,2)
        self.assertEqual(result['chosen_wrist'],1);self.assertEqual(result['chosen_front'],1)
        self.assertEqual(result['both_current_masks_empty'],1)
        self.assertEqual(result['best_view_at_least48_raw_pixels'],1)
    def test_missing_or_wrong_timeline_is_not_accepted(self):
        with self.assertRaises(ValueError):summarize(np.zeros((1,384,384),bool),np.zeros((1,384,384),bool),2)


if __name__=='__main__':unittest.main()
