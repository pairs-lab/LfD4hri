"""Synthetic fixtures only; they never count as physical benchmark outcomes."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from . import common, evaluate, workflow


class FinalContinuationTests(unittest.TestCase):
    def test_new_holdout_disjoint_and_exact_counts(self):
        self.assertTrue(common.verify_cohorts(range(20310001,20310201)))
        self.assertEqual((len(common.FINAL)*4,len(common.DEV)*4,len(common.CONF)*4),(800,40,40))
        with self.assertRaises(ValueError):common.verify_cohorts(common.FINAL)

    def test_one_final_fit_budget_and_lr_continuation(self):
        from VLA.admission_r100.recipe import RECIPES,learning_rate
        self.assertEqual(common.TARGET_UPDATES,2050+2048)
        self.assertEqual(RECIPES['falcon_esm']['maximum_updates'],4096)
        self.assertGreater(learning_rate(2048,RECIPES['falcon_esm']),learning_rate(4096,RECIPES['falcon_esm']))

    def test_new_final_requires_new_freeze_not_old800(self):
        method={'family':'falcon_esm','actual_checkpoint_optimizer_updates':4098,
            'R100_frozen_before_final':True,'R105_new_holdout_frozen_before_final':True,
            'frozen_for_user_requested_benchmark':True}
        cohort={'role':'fresh_secondary_final800','seeds':common.FINAL}
        self.assertTrue(common.approval(method,cohort))
        for key,value in [('family','Oracle'),('actual_checkpoint_optimizer_updates',2052),
                ('R105_new_holdout_frozen_before_final',False),('R100_frozen_before_final',False)]:
            self.assertFalse(common.approval({**method,key:value},cohort))
        self.assertFalse(common.approval(method,{**cohort,'seeds':list(range(20310001,20310201))}))
        self.assertTrue(common.approval({**method,'actual_checkpoint_optimizer_updates':2050},cohort))

    def test_physical_callbacks_restore_after_error(self):
        import VLA.admission_r100.evaluate as base
        saved=base.approved
        with self.assertRaises(RuntimeError):
            with evaluate.callbacks():
                self.assertIs(base.approved,common.approval)
                with base.callbacks():
                    import VLA.autorefinement_r97.audit as auditor
                    self.assertIs(auditor.approved_final,common.approval)
                    raise RuntimeError('SYNTHETIC callback fixture')
        self.assertIs(base.approved,saved)

    def test_final_report_does_not_claim_zero_completed(self):
        per={t:{'n':200,'successes':50,'SR_percent':25.} for t in common.TASKS}
        txt=workflow.summary({'status':'completed','stage':'completed_actual800_certified',
            'certified_result':{'actual_trials':800,'per_action':per}})
        self.assertIn('800/800',txt);self.assertNotIn('0/800',txt.replace('800/800',''))
        self.assertIn('NOT strict1cm',txt)

    def test_zero_trial_sr_report_is_unknown_not_success_or_error(self):
        per={t:{'n':0,'successes':0,'SR_percent':None} for t in common.TASKS}
        txt=workflow.summary({'status':'running','stage':'final800',
            'actual_progress':{'completed_trials':0,'planned_trials':800,'per_action':per}})
        self.assertIn('|touch|0/0|—|',txt)
        self.assertNotIn('0.00%',txt)

    def test_never_signal_legacy_user_worker_or_reused_pid(self):
        bad={'pid':123,'command':['python','-m','VLA.run_smol_priority_r38']}
        with mock.patch.object(workflow,'same_owned_process',return_value=True),mock.patch('os.kill') as kill:
            with self.assertRaises(ValueError):workflow.stop_owned([bad])
            kill.assert_not_called()
        with mock.patch.object(workflow,'same_owned_process',return_value=False),mock.patch('os.kill') as kill:
            workflow.stop_owned([bad]);kill.assert_not_called()

    def test_approved_owned_evaluator_termination_is_narrow(self):
        expected={'pid':123,'command':['python','-m','VLA.falcon_final_r105.evaluate']}
        with mock.patch.object(workflow,'same_owned_process',side_effect=[True,False,False]),mock.patch('os.kill') as kill:
            workflow.stop_owned([expected])
            self.assertEqual(kill.call_count,1)

    def test_no_synthetic_sr_in_live_training_report(self):
        with tempfile.TemporaryDirectory(prefix='R105_SYNTHETIC_NOT_ROBOT_') as tmp:
            root=Path(tmp);folder=root/'falcon_esm/fit';folder.mkdir(parents=True)
            common.atomic_json(root/'WORKFLOW_STATUS.json',{'status':'running','stage':'fit'})
            common.atomic_json(folder/'status.json',{'pid':123,'updates':2051,'target_updates':4098})
            ident={'pid':123,'command':['python','-m','VLA.admission_r100.train']}
            with mock.patch.object(workflow,'OUT',root),mock.patch.object(workflow,'alive',return_value=True),mock.patch.object(
                workflow,'identity',return_value=ident):
                result=workflow.live_snapshot()
            self.assertEqual(result['actual_trainer_status']['updates'],2051)
            self.assertNotIn('per_action',result)
            self.assertIn('2051/4098',(root/'REPORT.md').read_text())

    def test_pending_query_probe_excludes_committed_outcomes(self):
        progress={'rows':[{'seed':1,'task':'touch','steps':80}]}
        records=[{'seed':1,'task':'touch','index':i} for i in range(80)]
        records += [{'seed':1,'task':'pick','index':i} for i in range(6)]
        pending=workflow.pending_records(progress,records)
        self.assertEqual(len(pending),6)
        self.assertTrue(all(r['task']=='pick' for r in pending))

    def test_half_written_journal_line_is_not_fabricated_into_a_query(self):
        with tempfile.TemporaryDirectory(prefix='R105_SYNTHETIC_') as tmp:
            p=Path(tmp)/'journal.jsonl'
            p.write_text('{"index":0}\n{"index":')
            self.assertEqual(workflow.journal_records(p),[{'index':0}])


if __name__=='__main__':unittest.main()
