import unittest
import numpy as np

from VLA.audit_wrongheld_native64 import feedback_checks


class WrongHeldNativeTests(unittest.TestCase):
    def sample(self):
        states = np.zeros((2, 64), np.float32)
        states[:, 18:24] = [1, 0, 0, 0, 1, 0]
        states[:, 24:27] = [.04, .04, .06]
        states[:, 27] = .04
        states[0, 30:32] = 1
        actions = np.zeros((2, 7), np.float32)
        actions[:, 6] = -1
        return states, actions, ['', '']

    def test_physical_wronghold_information_survives_all_views(self):
        proof = feedback_checks(*self.sample())
        self.assertEqual(proof['wrong_held_observations'], 1)
        self.assertTrue(proof['actual67_native32_and_full64_language_preserve_wrong_hold'])

    def test_empty_start_cannot_be_claimed_as_wrongheld(self):
        states, actions, held = self.sample()
        states[0, 30:32] = 0
        with self.assertRaisesRegex(ValueError, 'wrong-held'):
            feedback_checks(states, actions, held)

    def test_release_label_cannot_drag_or_keep_distractor(self):
        for change in ('translation', 'closure', 'held'):
            states, actions, held = self.sample()
            if change == 'translation': actions[0, 0] = .01
            if change == 'closure': actions[0, 6] = 1
            if change == 'held': held[0] = 'wrong object'
            with self.assertRaisesRegex(ValueError, 'release'):
                feedback_checks(states, actions, held)


if __name__ == '__main__': unittest.main()
