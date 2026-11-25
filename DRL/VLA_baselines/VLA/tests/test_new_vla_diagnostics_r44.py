import unittest

import numpy as np

from VLA.watch_new_vla_diagnostics_r44 import diagnose_trace


class NativeDiagnosticsTest(unittest.TestCase):
    def step(self, held=False):
        raw = np.zeros(64)
        raw[11:14] = [.1, 0., 0.]
        raw[17] = raw[31] = int(held)
        state = {'correct_object_held': held, 'goal_xyz_m': [.1, 0., 0.],
                 'object_base_xyz_m': [0., 0., 0.]}
        return {'pre_action_scene64': raw.tolist(), 'action': [-.01, 0., 0., 0., 0., 0., 1.],
                'before': dict(state), 'after': dict(state), 'safety_violation': None}

    def test_held_transport_is_not_wrong_direction_reaching(self):
        result = diagnose_trace([self.step(True)], 'move')
        self.assertEqual(result['unheld_reach_away_steps'], 0)

    def test_empty_reach_away_is_a_symptom_not_a_root_cause(self):
        result = diagnose_trace([self.step(False)], 'pick')
        self.assertIn('unheld_reach_command_points_away_from_target', result['symptoms'])
        self.assertIn('never_attached_correct_target', result['symptoms'])

    def test_grasp_acquired_this_step_is_not_empty_lift(self):
        step = self.step(False)
        step['pre_action_scene64'][11] = .02
        step['action'][:3] = [0., 0., .02]
        step['after']['correct_object_held'] = True
        result = diagnose_trace([step], 'pick')
        self.assertEqual(result['near_empty_close_lift_over1cm_steps'], 0)


if __name__ == '__main__':
    unittest.main()
