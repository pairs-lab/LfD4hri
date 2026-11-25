from types import SimpleNamespace
import unittest
import numpy as np
from VLA.collect_v9_recovery_candidate import GentleRecoveryCollection, opening_retreat_action, recovery_checks


class RecoveryCandidateTest(unittest.TestCase):
    def test_near_object_opening_never_emits_horizontal_sweep(self):
        action=opening_retreat_action(np.array([.24,-.38,.03]),.20,.97)
        np.testing.assert_array_equal(action[[0,1,3,4,5]],np.zeros(5))
        self.assertGreater(action[2],0)
        self.assertEqual(action[-1],-1)
        self.assertIsNone(opening_retreat_action(np.array([.24,-.38,.199]),.20,.002))
        self.assertIsNotNone(opening_retreat_action(np.array([.24,-.38,.199]),.20,.5))
    def arrays(self):
        return {'canonical_actions': np.array([[0,0,0,0,0,0,-1],[0,0,.02,0,0,0,1]]),
                'gripper_closure': np.array([.97,.4]), 'next_requested_gripper_closure': np.array([0.,1.]),
                'next_grasped_object': np.array([False,True]), 'next_verified_grasp': np.array([False,True]),
                'next_bilateral_pad_contact': np.array([False,True])}

    def test_closed_measured_feedback_corrective_open_and_verified_carry(self):
        self.assertTrue(all(recovery_checks(self.arrays(), 'pick', {'safety_violation':None}).values()))
        for key in ('gripper_closure', 'next_requested_gripper_closure', 'next_verified_grasp'):
            a=self.arrays(); a[key][0]=.1 if key=='gripper_closure' else 1
            self.assertFalse(all(recovery_checks(a, 'pick', {'safety_violation':None}).values()))

    def test_touch_does_not_require_grasp_but_preserves_unheld_start_and_safety(self):
        a=self.arrays(); a['next_verified_grasp'][:]=False; a['next_bilateral_pad_contact'][:]=False
        self.assertTrue(all(recovery_checks(a, 'touch', {'safety_violation':None}).values()))
        self.assertFalse(all(recovery_checks(a, 'move', {'safety_violation':None}).values()))
        self.assertFalse(all(recovery_checks(a, 'touch', {'safety_violation':'mimic'}).values()))

    def test_gentle_touch_reset_replays_startup_instead_of_default_home(self):
        tasks=[]
        proxy=SimpleNamespace(env=SimpleNamespace(),reset=lambda task:tasks.append(task) or 'off_path_observation')
        wrapped=GentleRecoveryCollection(proxy)
        self.assertEqual(wrapped.reset('task'), 'off_path_observation')
        self.assertEqual(tasks, ['task'])


if __name__ == '__main__': unittest.main()
