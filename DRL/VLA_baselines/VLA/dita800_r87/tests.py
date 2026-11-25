import unittest
from pathlib import Path
import tempfile
from VLA.common.artifact_io import atomic_json
from .evaluate import authorized_frozen,compile_unchanged_physical_loop


class Contracts(unittest.TestCase):
    def setUp(self):
        self.method={'frozen_for_user_requested_benchmark':True,'frozen_after_confirmation':False,
            'benchmark_authorization':{'user_requested_800_despite_failed_pilot_gate':True}}
        self.cohort={'role':'fresh_secondary_final800','seeds':list(range(20284001,20284201))}
    def test_explicit_user_override_only(self):
        self.assertTrue(authorized_frozen(self.method,self.cohort))
        self.method['benchmark_authorization']={}
        self.assertFalse(authorized_frozen(self.method,self.cohort))
    def test_never_claim_confirmation_passed(self):
        self.method['frozen_after_confirmation']=True
        self.assertFalse(authorized_frozen(self.method,self.cohort))
    def test_missing_or_duplicate_seed_rejected(self):
        self.cohort['seeds'][-1]=self.cohort['seeds'][0]
        self.assertFalse(authorized_frozen(self.method,self.cohort))
    def test_compile_one_eligibility_change_in_existing_loop(self):
        from VLA.autonomy_r83 import evaluate as old
        self.assertTrue(callable(compile_unchanged_physical_loop(old)))
    def test_final_coverage_retains_false_confirmation_flag(self):
        from .audit import final_audit
        from VLA.autonomy_r83.common import metrics
        rows=[{'seed':s,'task':t,'success':False,'steps':1} for s in self.cohort['seeds']
            for t in ('touch','pick','move','pick_place')]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.method.update(code={},weights_sha256={})
            atomic_json(root/'PLAN.json',{'method':self.method,'cohort':self.cohort})
            progress={'rows':rows,'per_action':metrics(rows),'status':'completed_actual_final800',
                'completed_trials':800,'planned_trials':800,'motor_controls':800}
            atomic_json(root/'progress.json',progress)
            result=final_audit(root/'progress.json')
            self.assertEqual(result['actual_trials'],800)
            self.assertFalse(result['confirmation_gate_passed'])
            progress['rows'][-1]=progress['rows'][0]
            atomic_json(root/'progress.json',progress)
            with self.assertRaises(ValueError):final_audit(root/'progress.json')


if __name__=='__main__':unittest.main()
