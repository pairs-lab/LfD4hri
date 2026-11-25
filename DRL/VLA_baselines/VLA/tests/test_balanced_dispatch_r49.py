import unittest
from pathlib import Path

from VLA.run_balanced_dispatch_r49 import fit_command


class BalancedDispatchTest(unittest.TestCase):
    def test_gate_and_long_invoke_same_actual_balanced_entrypoint(self):
        for mode, updates in (('gate', 2), ('long', 6000)):
            command = fit_command(mode, Path('/unused_test_output'), updates)
            self.assertEqual(command[2], 'VLA.stable_native_r44.fit_balanced')
            self.assertEqual(command[command.index('--mode')+1], mode)
            self.assertEqual(command[command.index('--updates')+1], str(updates))
            self.assertEqual(command[command.index('--total-updates')+1], '6000')

    def test_long_cannot_lose_explicit_own_resume_and_reload_arguments(self):
        command = fit_command('long', Path('/unused_test_output'), 6000,
            Path('/unused_test_gate'), Path('/unused_test_proof.json'))
        self.assertEqual(command[command.index('--resume')+1], '/unused_test_gate')
        self.assertEqual(command[command.index('--reload-proof')+1], '/unused_test_proof.json')


if __name__ == '__main__':
    unittest.main()
