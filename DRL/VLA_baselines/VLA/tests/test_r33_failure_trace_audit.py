import unittest
from VLA.audit_r33_failure_traces import inspect_episode


class TraceAuditTests(unittest.TestCase):
    def frame(self):
        state = [0.] * 64
        state[11:14] = [.2, 0., -.03]
        state[24:27] = [.04, .04, .04]
        return {'pre_action_scene64': state, 'action': [0, 0, -.02, 0, 0, 0, -1], 'after': {}}

    def test_misaligned_descent_is_only_a_symptom_not_a_changed_score(self):
        result = inspect_episode({'seed': 1, 'task': 'touch', 'success': False}, [self.frame()])
        self.assertFalse(result['success'])
        self.assertEqual(result['overlapping_diagnostic_counts_NOT_causal_findings']['descending_low_outside_target_xy_envelope'], 1)

    def test_approach_to_high_object_is_not_a_floor_height_false_positive(self):
        frame = self.frame()
        frame['pre_action_scene64'][13] = -.3
        result = inspect_episode({'seed': 1, 'task': 'pick', 'success': False}, [frame])
        self.assertNotIn('descending_low_outside_target_xy_envelope', result['overlapping_diagnostic_counts_NOT_causal_findings'])

    def test_actual_hold_after_command_is_retained(self):
        frame = self.frame()
        frame['after'] = {'correct_object_held': True, 'held_object': 'target'}
        result = inspect_episode({'seed': 1, 'task': 'pick', 'success': True}, [frame])
        self.assertTrue(result['ever_correct_hold'])
        self.assertFalse(result['ever_wrong_hold'])

    def test_opening_is_counted_from_before_command_even_if_body_drops(self):
        frame = self.frame()
        frame['pre_action_scene64'][17] = 1
        frame['pre_action_scene64'][14] = .2
        frame['after'] = {'correct_object_held': False, 'held_object': None}
        result = inspect_episode({'seed': 1, 'task': 'move', 'success': False}, [frame])
        self.assertEqual(result['overlapping_diagnostic_counts_NOT_causal_findings']['OPEN_while_correctly_held_outside_goal_xy8cm'], 1)

    def test_approaching_touch_timeout_is_not_declared_success(self):
        frame = self.frame()
        frame['pre_action_scene64'][11:14] = [.02, 0, -.07]
        result = inspect_episode({'seed': 1, 'task': 'touch', 'success': False}, [frame])
        self.assertFalse(result['success'])
        self.assertIn('touch_terminal_within12cm_still_descending_NOT_predicted_extra_success',
                      result['overlapping_diagnostic_counts_NOT_causal_findings'])


if __name__ == '__main__':
    unittest.main()
