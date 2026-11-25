"""Synthetic fixtures never appear in experiment ledgers."""
import copy
import json
import math
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock
import numpy as np

from .common import DEV,CONF,FINAL,PARENT,approved,callbacks,learning_rate
from .data import regimes
from .owner import Owner,checked_signal


class Contracts(unittest.TestCase):
    def test_disjoint_locked_cohorts(self):
        self.assertEqual((len(DEV),len(CONF),len(FINAL)),(10,10,200))
        self.assertFalse(set(DEV)&set(CONF));self.assertFalse(set(DEV)&set(FINAL));self.assertFalse(set(CONF)&set(FINAL))
        self.assertFalse(set(FINAL)&set(range(20294001,20294201)))

    def test_no_final_authorization_without_frozen_flag(self):
        method={'family':'vlanext_s','R99_locked_before_final':True,
            'frozen_for_user_requested_benchmark':True,
            'benchmark_authorization':{'user_requested_800_after_refinement':True}}
        cohort={'role':'fresh_secondary_final800','seeds':FINAL}
        self.assertTrue(approved(method,cohort))
        self.assertFalse(approved({**method,'family':'dita'},cohort))
        self.assertFalse(approved({**method,'R99_locked_before_final':False},cohort))
        self.assertFalse(approved(method,{**cohort,'seeds':FINAL[:-1]}))

    def test_native_callbacks_restore_on_error(self):
        import VLA.autorefinement_r97.evaluate as evaluation
        import VLA.autorefinement_r97.audit as audit
        previous=evaluation.approved_final,audit.approved_final
        with self.assertRaises(RuntimeError):
            with callbacks():
                self.assertIs(evaluation.approved_final,approved)
                self.assertIs(audit.approved_final,approved)
                raise RuntimeError('synthetic')
        self.assertEqual((evaluation.approved_final,audit.approved_final),previous)

    def test_rehearsal_masks_are_disjoint_from_unsafe_descent(self):
        raw=np.zeros((4,64));raw[1,17]=1.;raw[2,28]=1.;raw[3,10]=1.
        actions=np.zeros((4,7));actions[0,:3]=[.02,0,-.02]
        actions[1,:3]=[.01,0,.01];actions[1,6]=1.
        result=regimes(raw,actions)
        self.assertFalse(result['safe_lateral_approach'][0])
        self.assertTrue(result['held_transport'][1])
        self.assertTrue(result['contact_or_grip_transition'][2])
        self.assertTrue(result['contact_or_grip_transition'][3])
        self.assertTrue(result['full_support'].all())

    def test_noncanonical_mask_input_fails(self):
        with self.assertRaises(ValueError):regimes(np.zeros((2,63)),np.zeros((2,7)))

    def test_learning_rate_bounded_resumable_and_finite(self):
        values=[learning_rate(i,512) for i in range(512)]
        self.assertTrue(all(math.isfinite(v) and 1e-6<=v<=2e-5 for v in values))
        self.assertLess(values[-1],values[32])
        self.assertEqual(values[128],learning_rate(128,512))
        with self.assertRaises(ValueError):learning_rate(512,512)

    def test_pid_reuse_and_foreign_module_never_signaled(self):
        ident={'pid':123,'command':['python','-m','foreign.worker']}
        with mock.patch('VLA.refinement_r99.owner.alive',return_value=True),mock.patch(
            'VLA.refinement_r99.owner.identity',return_value=ident),mock.patch('os.kill') as kill:
            with self.assertRaises(ValueError):checked_signal(ident,signal.SIGTERM)
            kill.assert_not_called()
        owned={**ident,'command':['python','-m','VLA.refinement_r99.train']}
        with mock.patch('VLA.refinement_r99.owner.alive',return_value=True),mock.patch(
            'VLA.refinement_r99.owner.identity',return_value=None),mock.patch('os.kill') as kill:
            with self.assertRaises(ValueError):checked_signal(owned,signal.SIGTERM)
            kill.assert_not_called()

    def test_evaluation_reuses_exact_tested_bytecode_new_paths(self):
        from VLA.postfit800_r98.owner import Owner as Previous
        from .common import OUT,audit_result
        self.assertIs(Owner.evaluate.__code__,Previous.evaluate.__code__)
        self.assertIs(Owner.wait.__code__,Previous.wait.__code__)
        self.assertEqual(Owner.evaluate.__globals__['OUT'],OUT)
        self.assertIs(Owner.evaluate.__globals__['audit_result'],audit_result)

    def test_no_historical_fit_entrypoint(self):
        instance=Owner.__new__(Owner);instance.bound={}
        with mock.patch('VLA.refinement_r99.owner.verify'):
            with self.assertRaises(ValueError):instance.spawn(Path('python'),
                'VLA.recovery_quality_r90.train',['--family','vlanext_s','fit'],'synthetic')

    def test_forecast_is_conditional_not_five_observations(self):
        from VLA.threshold_sensitivity_r96.analyze import forecast
        result=forecast(36,n=200,draws=100000,rng_seed=99)
        self.assertFalse(result['five_measured_cohorts'])
        self.assertTrue(result['same_checkpoint_not_training_seed_variance'])
        self.assertAlmostEqual(result['expected_five_cohort_mean_percent'],18.127363184079602)
        self.assertGreater(result['median_predictive_sample_std_percentage_points'],0)
        self.assertEqual(result['observed_n'],200)

    def test_native_pose_gripper_unchanged(self):
        from VLA.continuous_r86.vlanext_server import native_gripper
        actions=np.zeros((8,7),np.float32);actions[:,:6]=.013
        actions[:,6]=np.linspace(-.1,.1,8)
        actual=native_gripper(actions)
        np.testing.assert_array_equal(actual[:,:6],actions[:,:6])
        self.assertEqual(set(actual[:,6]),{-1.,1.})


class Orchestration(unittest.TestCase):
    """Exercise both complete pipeline branches in temporary synthetic storage."""
    def progress(self,root,name,count):
        from VLA.autonomy_r83.common import metrics
        from .common import TASKS
        rows=[{'seed':s,'task':task,'success':i<count} for i,(s,task) in enumerate(
            (s,t) for s in DEV for t in TASKS)]
        path=root/(name+'_synthetic_progress.json')
        path.write_text(json.dumps({'completed_trials':40,'rows':rows,'per_action':metrics(rows),
            'status':'completed_actual_development','cohort_role':'development_selection'}))
        return path

    def run_branch(self,improves):
        with tempfile.TemporaryDirectory(prefix='R99_SYNTHETIC_NOT_EXPERIMENT_') as tmp:
            root=Path(tmp);owner=Owner.__new__(Owner);owner.bound={};owner.child=owner.server=None
            owner.children=[];owner.stage='synthetic'
            paths={name:self.progress(root,name,count) for name,count in (
                ('baseline_dev40',12),('rehearsal512_dev40',20 if improves else 10),
                ('rehearsal1024_dev40',19),('confirmation40',18))}
            paths['final800']=root/'synthetic_final_placeholder'
            gate=root/'gate';first=root/'first';second=root/'second'
            fits={'gradient_gate16':gate,'rehearsal512':first,'rehearsal1024':second}
            fit_calls=[];eval_calls=[]
            def fit(parent,label,**kwargs):fit_calls.append(label);return fits[label]
            def method(cp,label,final=False):
                path=root/(label+'_METHOD.json')
                path.write_text(json.dumps({'code':{},'weights_sha256':{},'synthetic':True,'checkpoint':str(cp)}))
                return path
            def evaluate(method,label,role,seeds):eval_calls.append(label);return paths[label]
            result={'per_action':{task:{'n':200,'successes':0,'SR_percent':0,'safety_aborts':0}
                for task in ('touch','pick','move','pick_place')}}
            with mock.patch('VLA.refinement_r99.owner.OUT',root),mock.patch(
                'VLA.refinement_r99.owner.metric_guard'),mock.patch.object(owner,'fit',side_effect=fit),mock.patch.object(
                owner,'native_reload'),mock.patch.object(owner,'method',side_effect=method),mock.patch.object(
                owner,'evaluate',side_effect=evaluate),mock.patch('VLA.refinement_r99.owner.audit_result',return_value=result),mock.patch.object(
                owner,'status'):
                owner.run()
                expected=['gradient_gate16','rehearsal512']+(['rehearsal1024'] if improves else [])
                self.assertEqual(fit_calls,expected)
                self.assertEqual(eval_calls[-2:],['confirmation40','final800'])
                selected=json.loads((root/'SELECTED_BEFORE_CONF.json').read_text())
                self.assertEqual(selected['checkpoint'],str(first if improves else PARENT))
                self.assertTrue((root/'CERTIFIED_RESULT.json').exists())
                fit_calls.clear();eval_calls.clear();owner.run()
                self.assertFalse(fit_calls) # A restarted final never refits or selects again.
                self.assertEqual(eval_calls,['final800'])

    def test_improved_DEV_all_stages_then_freeze_then800(self):self.run_branch(True)
    def test_no_DEV_improvement_baseline_kept_no_second_fit(self):self.run_branch(False)


if __name__=='__main__':unittest.main()
