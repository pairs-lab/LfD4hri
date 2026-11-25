"""Reporting regressions only; never actual benchmark outcomes."""
import tempfile,unittest
from pathlib import Path
from unittest import mock
from VLA.admission_r100.common import read
from . import workflow as fixed


class ReportingTests(unittest.TestCase):
    def test_null_is_unknown_not_zero_success_rate(self):
        self.assertEqual(fixed.percentage(None),'—')
        self.assertEqual(fixed.percentage(0.),'0.00%')
        self.assertEqual(fixed.percentage(50.),'50.00%')

    def test_zero_trial_progress_and_error_status_never_stop_evaluator(self):
        import VLA.admission_r100.workflow as base
        with tempfile.TemporaryDirectory(prefix='r104_SYNTHETIC_') as tmp:
            root=Path(tmp)
            with mock.patch.object(fixed.previous,'configure',return_value=base):cls=fixed.owner_class()
            owner=cls.__new__(cls);owner.family='falcon_esm';owner.stage='final800';owner.child=owner.server=None
            per={t:{'n':0,'successes':0,'SR_percent':None} for t in ('touch','pick','move','pick_place')}
            with mock.patch.object(fixed,'OUT',root),mock.patch.object(base,'OUT',root):
                owner.status(actual_progress={'completed_trials':0,'native_queries':0,'per_action':per})
                self.assertIn('|touch|0/0|—|',(root/'REPORT.md').read_text())
                self.assertIsNone(read(root/'WORKFLOW_STATUS.json')['actual_progress']['per_action']['touch']['SR_percent'])
                owner.status('technical_error_retained',error='synthetic fixture')
                self.assertEqual(read(root/'WORKFLOW_STATUS.json')['status'],'technical_error_retained')


if __name__=='__main__':unittest.main()
