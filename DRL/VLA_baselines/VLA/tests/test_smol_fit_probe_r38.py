import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from VLA import audit_smol_fit_r38 as probe
from VLA import run_smol_priority_r38 as priority


class ProbeAdmissionTests(unittest.TestCase):
    def test_only_exact_inference_child_not_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            state={'status':'running','stage':'GTcrop_development120'}
            child={'pid':123,'stage':'GTcrop_development120'}
            process={'cwd':str(probe.ROOT),'tokens':[b'python',b'VLA.evaluate_target_crop_r36',str(root/'development120').encode()]}
            (root/'status.json').write_text(json.dumps(state));(root/'active_child.json').write_text(json.dumps(child))
            with patch.object(probe,'FIT',root),patch.object(probe,'identity',return_value=process):
                self.assertEqual(probe.inference_owner(),123)
                for stage in ('train','smoke','own_GTcrop_exact_reload8'):
                    (root/'status.json').write_text(json.dumps({**state,'stage':stage}))
                    self.assertIsNone(probe.inference_owner())

    def test_foreign_workspace_or_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'status.json').write_text(json.dumps({'status':'running','stage':'GTcrop_development120'}))
            (root/'active_child.json').write_text(json.dumps({'pid':123,'stage':'GTcrop_development120'}))
            cases=[{'cwd':'/tmp','tokens':[b'VLA.evaluate_target_crop_r36',str(root/'development120').encode()]},
                {'cwd':str(probe.ROOT),'tokens':[b'VLA.evaluate_target_crop_r36',b'another/output']}]
            for process in cases:
                with patch.object(probe,'FIT',root),patch.object(probe,'identity',return_value=process):
                    self.assertIsNone(probe.inference_owner())


class SupervisorRestoreTests(unittest.TestCase):
    def test_only_same_stopped_supervisor_gets_continue(self):
        lease={'supervisor_pid':22,'supervisor_start_ticks':'123'}
        process={'cwd':str(priority.ROOT),'start_ticks':'123','state':'T',
            'tokens':[b'VLA.run_all_privileged_r33',str(priority.OWNER).encode()]}
        with patch.object(priority,'identity',return_value=process),patch.object(priority.os,'kill') as send:
            self.assertTrue(priority.restore(lease));send.assert_called_once_with(22,priority.signal.SIGCONT)

    def test_recycled_or_not_stopped_process_never_signalled(self):
        lease={'supervisor_pid':22,'supervisor_start_ticks':'123'}
        for state,ticks in (('S','123'),('T','different')):
            process={'cwd':str(priority.ROOT),'start_ticks':ticks,'state':state,
                'tokens':[b'VLA.run_all_privileged_r33',str(priority.OWNER).encode()]}
            with patch.object(priority,'identity',return_value=process),patch.object(priority.os,'kill') as send:
                self.assertFalse(priority.restore(lease));send.assert_not_called()


if __name__=='__main__':unittest.main()
