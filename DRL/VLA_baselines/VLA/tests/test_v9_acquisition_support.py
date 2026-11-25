import unittest
import numpy as np

from VLA.audit_v9_acquisition_support import longest_run, phase_steps, transition_support, weighted_support


class AcquisitionSupportTest(unittest.TestCase):
    def arrays(self):
        action = np.zeros((6, 7), np.float32)
        action[:, 6] = [-1, -1, 1, 1, 1, 1]
        action[4:, 2] = .02
        return {"canonical_actions": action,
            "gripper_closure": np.array([.98, .6, .1, .3, .45, .45]),
            "next_gripper_closure": np.array([.6, .1, .3, .45, .45, .45]),
            "next_requested_gripper_closure": (action[:, 6] + 1) / 2,
            "next_grasped_object": np.array([False, False, False, True, True, True]),
            "next_verified_grasp": np.array([False, False, False, True, True, True])}

    def test_transition_alignment_does_not_call_acquiring_state_held(self):
        a = self.arrays()
        s = transition_support(a)
        self.assertEqual(np.flatnonzero(s['corrective_reopening']).tolist(), [0, 1])
        self.assertEqual(np.flatnonzero(s['closing_without_target']).tolist(), [2, 3])
        self.assertEqual(np.flatnonzero(s['target_acquired']).tolist(), [3])
        self.assertEqual(np.flatnonzero(s['lift_after_target_held']).tolist(), [4, 5])
        self.assertFalse(s['close_up_without_target_before_or_after'].any())
        steps = phase_steps(a, s)
        self.assertEqual(steps['pre_acquisition'], 3)
        self.assertEqual(steps['post_acquisition_observation'], 4)
        self.assertEqual(steps['last_pre_action_NOT_terminal'], 5)

    def test_unconfirmed_lift_and_loss_are_preserved_not_hidden(self):
        a = self.arrays()
        a['canonical_actions'][2, 2] = .02
        a['next_grasped_object'][5] = False
        s = transition_support(a)
        self.assertTrue(s['close_up_without_target_before_or_after'][2])
        self.assertTrue(s['target_lost'][5])

    def test_feedback_is_not_command_and_bad_temporal_alignment_fails(self):
        a = self.arrays()
        a['gripper_closure'][2] = 1
        with self.assertRaisesRegex(ValueError, 'transition-synchronized'):
            transition_support(a)
        a = self.arrays()
        a['next_requested_gripper_closure'][2] = 0
        with self.assertRaisesRegex(ValueError, 'native label'):
            transition_support(a)

    def test_nonfinite_out_of_range_and_lengths_fail(self):
        for key, value in (('gripper_closure', np.nan), ('next_gripper_closure', 2)):
            a = self.arrays(); a[key][0] = value
            with self.assertRaises(ValueError): transition_support(a)
        a = self.arrays(); a['next_grasped_object'] = np.zeros(5)
        with self.assertRaises(ValueError): transition_support(a)

    def test_sampling_mass_has_exact_frame_offsets(self):
        rows = [{'file': 'a', 'steps': 2}, {'file': 'b', 'steps': 3}]
        episodes = {'a': {'support': {'reopen': np.array([True, False])}},
                    'b': {'support': {'reopen': np.array([False, True, False])}}}
        self.assertAlmostEqual(weighted_support(rows, episodes, np.array([.1, .1, .2, .3, .3]))['reopen'], .4)
        with self.assertRaises(ValueError): weighted_support(rows, episodes, np.ones(4)/4)

    def test_empty_runs_and_single_action_terminal_not_invented(self):
        self.assertEqual(longest_run([False, True, True, False, True]), 2)
        a = {key: value[3:4].copy() for key, value in self.arrays().items()}
        steps = phase_steps(a, transition_support(a))
        self.assertNotIn('post_acquisition_observation', steps)


if __name__ == '__main__':
    unittest.main()
