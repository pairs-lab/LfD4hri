import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from .common import OLD_THRESHOLDS,DEV,CONF,FINAL,TASKS,score,approved_final,metric_guard
from .train import learning_rate
from .evaluate import ledger


def progress(seeds=(11,),done_tasks=None):
    from VLA.autonomy_r83.common import metrics
    tasks=TASKS if done_tasks is None else done_tasks
    rows=[{'seed':s,'task':t,'success':True,'steps':1,'trajectory':'placeholder.json',
        'safety_violation':None} for s in seeds for t in tasks]
    return {'rows':rows,'per_action':metrics(rows),'completed_trials':len(rows),
        'planned_trials':len(seeds)*4,'status':'completed_actual_development',
        'cohort_role':'development_selection'}


class Contracts(unittest.TestCase):
    def test_original_thresholds_not_minus1cm(self):
        self.assertEqual(metric_guard()['semantic_thresholds'],OLD_THRESHOLDS)
        self.assertEqual(OLD_THRESHOLDS['pick_min_lift_m'],.05)
        self.assertEqual(OLD_THRESHOLDS['touch_distance_m'],.04)

    def test_wrong_metric_fails_closed(self):
        with mock.patch('VLA.autorefinement_r97.common.read',return_value={'success':{
            'semantic_thresholds':{**OLD_THRESHOLDS,'pick_min_lift_m':.04},
            'primary_mode':'semantic_target_region','required_consecutive_steps':1,
            'reward_xml_threshold_scale':1.25}}):
            with self.assertRaises(ValueError):metric_guard()

    def test_blocks_disjoint_and_full800(self):
        self.assertFalse(set(DEV)&set(CONF) or set(DEV)&set(FINAL) or set(CONF)&set(FINAL))
        self.assertEqual(len(FINAL)*len(TASKS),800)
        self.assertFalse(set(FINAL)&set(range(20284001,20284201)))

    def test_continuation_lr_is_small_finite_bounded(self):
        values=[learning_rate(i,512) for i in range(513)]
        self.assertGreaterEqual(min(values),1e-6)
        self.assertLessEqual(max(values),1e-5)
        self.assertAlmostEqual(values[-1],1e-6)

    def test_resumed_lr_uses_original_phase_offset(self):
        self.assertEqual(learning_rate(10144-10016,512),learning_rate(128,512))
        self.assertNotEqual(learning_rate(128,512),learning_rate(0,512))

    def test_only_complete40_may_be_scored(self):
        p=progress(range(11,21))
        self.assertEqual(score(p),(40,10,0))
        p['rows'].pop()
        with self.assertRaises(ValueError):score(p)

    def test_score_rejects_final_role(self):
        p=progress(range(11,21));p['cohort_role']='fresh_secondary_final800'
        with self.assertRaises(ValueError):score(p)

    def test_fixed_prefix_ledger(self):
        p=progress(done_tasks=TASKS[:2])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.json';path.write_text(json.dumps(p))
            data,rows,keys=ledger(path,{'seeds':[11]})
            self.assertEqual(len(rows),2)
            self.assertIn((11,'pick'),keys)

    def test_duplicate_or_out_of_order_rejected(self):
        p=progress()
        for rows in ([p['rows'][0]]*4,list(reversed(p['rows']))):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'p.json';path.write_text(json.dumps({**p,'rows':rows}))
                with self.assertRaises(ValueError):ledger(path,{'seeds':[11]})

    def test_partial_is_not_a_completed_failure(self):
        p=progress();p['completed_trials']=0
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.json';path.write_text(json.dumps(p))
            with self.assertRaises(ValueError):ledger(path,{'seeds':[11]})

    def test_explicit_final_authorization_no_fake_conf_pass(self):
        cohort={'role':'fresh_secondary_final800','seeds':FINAL}
        method={'family':'dita','frozen_for_user_requested_benchmark':True,
            'frozen_after_confirmation':False,'R97_locked_before_final':True,
            'benchmark_authorization':{'user_requested_800_after_refinement':True}}
        self.assertTrue(approved_final(method,cohort))
        self.assertFalse(method['frozen_after_confirmation'])
        self.assertFalse(approved_final(method,{**cohort,'seeds':FINAL[:-1]}))

    def test_resume_transform_still_has_four_exact_bookkeeping_edits(self):
        from VLA.autonomy_r83 import evaluate as old
        from VLA.receipt_recovery_r88.evaluate import compile_resume
        from types import SimpleNamespace
        fn=compile_resume(old,lambda:([],0,0),lambda c:[],lambda *a:False,
            SimpleNamespace(Session=lambda:None),lambda *a:None)
        self.assertIn('step',fn.__code__.co_names)
        self.assertIn('model_input',fn.__code__.co_names)

    def test_signals_cannot_target_unrelated_or_optimizer_modules(self):
        from .owner import exact_signal
        ident={'pid':123,'command':['python','-m','unrelated.training']}
        with mock.patch('VLA.autorefinement_r97.owner.alive',return_value=True), \
            mock.patch('VLA.autorefinement_r97.owner.identity',return_value=ident), \
            mock.patch('VLA.autorefinement_r97.owner.os.kill') as kill:
            with self.assertRaises(ValueError):exact_signal(ident,signal.SIGTERM)
            with self.assertRaises(ValueError):exact_signal(ident,signal.SIGSTOP,idle_queue=True)
            kill.assert_not_called()

    def test_pid_reuse_never_signaled(self):
        from .owner import exact_signal
        ident={'pid':123,'command':['python','-m','VLA.autorefinement_r97.train']}
        with mock.patch('VLA.autorefinement_r97.owner.alive',return_value=True), \
            mock.patch('VLA.autorefinement_r97.owner.identity',return_value={**ident,'start_ticks':'changed'}), \
            mock.patch('VLA.autorefinement_r97.owner.os.kill') as kill:
            with self.assertRaises(ValueError):exact_signal(ident,signal.SIGTERM)
            kill.assert_not_called()


if __name__=='__main__':
    unittest.main()
