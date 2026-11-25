import unittest
import numpy as np

from VLA.near_object_candidate_expansion import supplementary_near_object_candidates
from VLA.near_object_recovery_start import near_object_candidates


class NearCandidateExpansionTest(unittest.TestCase):
    def test_default_cardinals_and_safe_extra_proposals(self):
        low, high = np.array([-.02, -.52, .004]), np.array([.02, -.48, .044])
        bounds = [('green block', low, high)]
        offsets = (np.array([-.069, -.013, -.025]), np.array([.069, .013, .13]))
        self.assertEqual(len(near_object_candidates(bounds, offsets, .004)), 4)
        extra = supplementary_near_object_candidates(bounds, offsets, .004)
        self.assertGreater(len(extra), 4)
        for proposal in extra:
            target = np.asarray(proposal['target_tcp'])
            self.assertGreaterEqual(target[2] + offsets[0][2], .014 - 1e-12)
            self.assertLessEqual(target[2], high[2] + .02)
            self.assertLessEqual(proposal['planned_anchor_aabb_distance_m'], .11)
            self.assertFalse(np.all(target + offsets[1] >= low - .008) and
                             np.all(target + offsets[0] <= high + .008))

    def test_obstruction_cannot_be_ignored(self):
        bounds = [('obstruction', np.array([-.6, -.9, 0]), np.array([.6, -.1, .5]))]
        offsets = (np.array([-.069, -.013, -.025]), np.array([.069, .013, .13]))
        self.assertEqual(supplementary_near_object_candidates(bounds, offsets, .004), [])

    def test_input_order_is_not_target_identity(self):
        bounds = [('red apple', np.array([-.02, -.52, .004]), np.array([.02, -.48, .044])),
                  ('blue block', np.array([.16, -.32, .004]), np.array([.20, -.28, .044]))]
        offsets = (np.array([-.069, -.013, -.025]), np.array([.069, .013, .13]))
        self.assertEqual(supplementary_near_object_candidates(bounds, offsets, .004),
                         supplementary_near_object_candidates(bounds[::-1], offsets, .004))


if __name__ == '__main__':
    unittest.main()
