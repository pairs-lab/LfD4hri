"""No real processes are signalled in these safety-policy unit tests."""
import signal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from VLA import cancel_fast_r66 as target


class CancelPolicyTests(unittest.TestCase):
    def test_JEPA_stages_never_cancelled(self):
        for phase in ('vla_jepa_real_native_finetune', 'vla_jepa_actual_locked800_no_video',
                      'vla_jepa_wait_all_R62_and_exclusive_GPU', 'finished_extension_review_errors_if_any'):
            self.assertFalse(target.is_fast_stage(phase))

    def test_FAST_phases_are_identified(self):
        self.assertTrue(target.is_fast_stage('pi0_fast_own_native_CPU_input_audit'))
        self.assertTrue(target.is_fast_stage('pi0_fast_real_native_finetune'))

    def test_only_FAST_children_allowed(self):
        def entry(name, *args):
            return {'argv': ['python', '-m', name, *args]}
        self.assertTrue(target.is_fast_child(entry('VLA.extension_r65.train_fast', 'fit')))
        self.assertTrue(target.is_fast_child(entry('VLA.extension_r65.server', '--model', 'pi0_fast')))
        self.assertFalse(target.is_fast_child(entry('VLA.extension_r65.train_jepa', 'fit')))
        self.assertFalse(target.is_fast_child(entry('VLA.extension_r65.server', '--model', 'vla_jepa')))
        self.assertFalse(target.is_fast_child(entry('VLA.cross_vla800_r62.evaluate', '--model', 'pi0')))

    def test_PID_reuse_cannot_be_signalled(self):
        expected = {'pid': 111, 'start_ticks': 'old', 'argv': ['python']}
        with patch.object(target, 'process', return_value={**expected, 'start_ticks': 'new'}), \
                patch.object(target.os, 'kill') as kill:
            target.send(expected, signal.SIGTERM)
            kill.assert_not_called()

    def test_JEPA_child_prevents_shared_owner_cancellation(self):
        owner = {'pid': 111, 'start_ticks': 'old', 'argv': ['python'], 'state': 'T'}
        jepa = {'pid': 222, 'argv': ['python', '-m', 'VLA.extension_r65.train_jepa']}
        with patch.object(target, 'same_process', return_value=owner), \
                patch.object(target, 'read', return_value={'stage': 'pi0_fast_real_native_finetune'}), \
                patch.object(target, 'children', return_value=[jepa]), \
                patch.object(target, 'send') as send:
            with self.assertRaisesRegex(RuntimeError, 'Non-FAST'):
                target.stop_fast_phase(owner, Path('/unused'), Path('/unused'))
            self.assertEqual([args.args[1] for args in send.call_args_list], [signal.SIGSTOP, signal.SIGCONT])

    def test_FAST_only_stop_finishes_without_signalling_JEPA(self):
        owner = {'pid': 111, 'start_ticks': 'owner', 'argv': ['python'], 'state': 'T'}
        fast = {'pid': 222, 'start_ticks': 'fast', 'argv': ['python', '-m', 'VLA.extension_r65.train_fast']}
        live = {111: owner, 222: fast}
        signals = []
        def send(info, sig):
            signals.append((info['pid'], sig))
            if sig == signal.SIGTERM:
                live.pop(info['pid'], None)
        state = {'stage': 'pi0_fast_actual_two_Adam_updates',
                 'models': {'vla_jepa': {'status': 'completed800_and_audit'}}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'REPORT.md').write_text('|pi0_fast|0/800|pending|pending|pending|pending|\n')
            with patch.object(target, 'same_process', side_effect=lambda p: live.get(p['pid'])), \
                    patch.object(target, 'read', return_value=state), \
                    patch.object(target, 'children', return_value=[fast]), \
                    patch.object(target, 'descendants', return_value=[fast]), \
                    patch.object(target, 'send', side_effect=send), \
                    patch.object(target.time, 'sleep'), \
                    patch('VLA.extension_r65.results.refresh'):
                self.assertTrue(target.stop_fast_phase(owner, root, root))
            self.assertEqual(signals, [(111, signal.SIGSTOP), (222, signal.SIGTERM),
                                       (111, signal.SIGTERM), (111, signal.SIGCONT)])
            self.assertTrue((root / 'CANCEL_COMPLETED.json').exists())
            self.assertEqual(state['models']['vla_jepa']['status'], 'completed800_and_audit')
            self.assertIn('user_cancelled', state['models']['pi0_fast']['status'])


if __name__ == '__main__':
    unittest.main()
