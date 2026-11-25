import unittest
import numpy as np
from VLA.flower_r71.sampler import flags
from VLA.flower_r71.watch import needs_rehearsal, candidate_wins


class Transitions(unittest.TestCase):
    def test_actual_point2_trigger_and_contact(self):
        raw = np.zeros((3, 64), np.float32); command = np.zeros((3, 7), np.float32)
        command[:, 6] = [.21, -.21, .19]; raw[:, 28] = 1
        f = flags(raw, command)
        np.testing.assert_array_equal(f['pre_close'], [True, False, False])
        np.testing.assert_array_equal(f['contact_unattached'], [True, False, False])

    def test_wrong_grasp_not_accepted_as_correct(self):
        raw = np.zeros((3, 64), np.float32); command = np.zeros((3, 7), np.float32)
        raw[:, 31] = 1; raw[:, 30] = [1, 0, 0]
        f = flags(raw, command)
        np.testing.assert_array_equal(f['first_correct_hold'], [False, True, False])
        self.assertFalse(f['closing_unattached'].any())

    def result(self, pick, total, safety=0):
        rows = [{'success': i < total, 'safety_violation': 'abort' if i < safety else None} for i in range(40)]
        return {'completed_trials': 40, 'per_action': {
            t: {'n': 10, 'successes': pick if t == 'pick' else 0}
            for t in ('touch', 'pick', 'move', 'pick_place')}, 'rows': rows}

    def test_trigger_uses_complete_development_only(self):
        self.assertTrue(needs_rehearsal(self.result(4, 25)))
        self.assertTrue(needs_rehearsal(self.result(5, 19)))
        self.assertFalse(needs_rehearsal(self.result(5, 20)))
        partial = self.result(0, 0); partial['completed_trials'] = 39
        with self.assertRaises(ValueError): needs_rehearsal(partial)

    def test_candidate_must_improve_without_extra_safety_failures(self):
        baseline = self.result(2, 20, 2)
        self.assertTrue(candidate_wins(baseline, self.result(4, 23, 2)))
        self.assertFalse(candidate_wins(baseline, self.result(4, 23, 3)))
        self.assertFalse(candidate_wins(baseline, self.result(1, 23, 2)))
        self.assertFalse(candidate_wins(baseline, self.result(4, 20, 2)))


if __name__ == '__main__': unittest.main()
