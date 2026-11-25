"""Offline fault-injection checks before adopting a running study."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from VLA.common.artifact_io import atomic_json
from .audit import trace_metrics, cohort_audit


def step(*, held=False, contact=False, success=False, safety=None):
    raw = [0.]*64
    raw[11:14] = [.1, .2, -.3]
    post = raw.copy(); post[17] = float(held); post[28] = float(contact)
    return {'pre_action_scene64': raw, 'post_action_scene64': post,
            'action': [.01, .02, -.03, 0., 0., 0., -1.],
            'executed_canonical_action': [.01, .02, -.03, 0., 0., 0., -1.],
            'success': success, 'safety_violation': safety}


class Contracts(unittest.TestCase):
    def test_target_direction_alignment(self):
        self.assertAlmostEqual(trace_metrics([step()])['action_target_cosine'], 1.)

    def test_gripper_matches_actual_official_denormalizer(self):
        import ast
        from .vlanext_server import native_gripper
        from VLA.replacement_r84.data import denormalize_actions
        # Extract the exact pure function from the cloned official evaluator;
        # do not import its unrelated MuJoCo/LIBERO runtime into PyBullet.
        source=Path('VLA/third_party/vlanext/src/evaluation/libero_bench/parallel_eval_utils.py')
        tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_denormalize_action')
        namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        values=np.random.default_rng(86).normal(size=(9,7)).astype(np.float32)
        values[:,6]=[-1,-.2,-.001,0,.001,.15,.2,.9,1]
        stats={'actions':{'q01':[-.045]*3+[-.03,-.03,-.08,-1],
                          'q99':[.045]*3+[.03,.03,.08,1]}}
        decoded=denormalize_actions(values,stats)
        ours=native_gripper(decoded)
        reference=np.stack([namespace['_denormalize_action'](v,np.asarray(stats['actions']['q01'][:6],np.float32),
            np.asarray(stats['actions']['q99'][:6],np.float32)) for v in values])
        np.testing.assert_allclose(ours,reference,atol=1e-8)
        np.testing.assert_array_equal(ours[:,:6],decoded[:,:6])
        np.testing.assert_array_equal(ours[:,6],[-1,-1,-1,-1,1,1,1,1,1])
        # The helper never mutates source neural output in place.
        np.testing.assert_array_equal(decoded[:,6],np.clip(values[:,6],-1,1))

    def test_drop_not_invented_without_prior_hold(self):
        self.assertEqual(trace_metrics([step(), step()])['hold_then_loss_transitions'], 0)
        self.assertEqual(trace_metrics([step(held=True), step()])['hold_then_loss_transitions'], 1)

    def test_safety_not_relabelled_success(self):
        r = trace_metrics([step(safety='surface collision')])
        self.assertEqual(r['signature_NOT_causal_diagnosis'], 'safety_abort')

    def test_nonfinite_is_integration_error(self):
        r = step(); r['action'][0] = float('nan')
        with self.assertRaises(ValueError): trace_metrics([r])

    def test_empty_is_integration_error(self):
        with self.assertRaises(ValueError): trace_metrics([])

    def test_final_must_be_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root/'PLAN.json', {'method': {'frozen_after_confirmation': False}})
            atomic_json(root/'progress.json', {'cohort_role': 'fresh_secondary_final800'})
            with self.assertRaises(ValueError): cohort_audit(root/'progress.json')

    def test_partial_cannot_promote(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root/'PLAN.json', {'method': {}, 'cohort': {'seeds': [1]}})
            atomic_json(root/'progress.json', {'cohort_role': 'development_selection', 'rows': [],
                'status': 'running_actual_neural', 'completed_trials': 0, 'planned_trials': 4})
            with self.assertRaises(ValueError): cohort_audit(root/'progress.json', require_complete=True)

    def test_duplicate_episode_cannot_promote(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root/'PLAN.json', {'method': {}, 'cohort': {'seeds': [1]}})
            row = {'seed': 1, 'task': 'touch'}
            atomic_json(root/'progress.json', {'cohort_role': 'development_selection', 'rows': [row, row]})
            with self.assertRaises(ValueError): cohort_audit(root/'progress.json')

    def test_pid_reuse_never_signalled(self):
        from . import owner
        expected={'pid':123,'start_ticks':'1','command':['VLA.recovery_r85.owner']}
        actual={**expected,'start_ticks':'2'}
        with mock.patch.object(owner,'identity',return_value=actual), mock.patch.object(owner.os,'kill') as kill:
            with self.assertRaises(ValueError):owner.checked_signal(expected,19)
            kill.assert_not_called()

    def test_unrelated_process_never_signalled(self):
        from . import owner
        expected={'pid':123,'start_ticks':'1','command':['unrelated.py']}
        with mock.patch.object(owner,'identity',return_value=expected), mock.patch.object(owner.os,'kill') as kill:
            with self.assertRaises(ValueError):owner.checked_signal(expected,19)
            kill.assert_not_called()

    def test_complete_metric_rule_unchanged(self):
        from .owner import ready
        tasks=('touch','pick','move','pick_place')
        p={'status':'completed_actual_development','completed_trials':40,
           'rows':[{'seed':s,'task':t,'success':s<6} for s in range(10) for t in tasks],
           'per_action':{t:{'n':10,'successes':6} for t in tasks}}
        self.assertTrue(ready(p))
        p['status']='integration_error_NOT_zero_model_SR'
        with self.assertRaises(ValueError):ready(p)

    def test_boundary_handoff_waits_for_child_and_only_signals_owned_processes(self):
        from . import owner
        expected={'pid':123,'ppid':1,'start_ticks':'1','command':['VLA.recovery_r85.owner']}
        server={'pid':43,'ppid':123,'start_ticks':'2','command':['VLA.replacement_r84.server']}
        living={123:True,42:False,43:True}
        signals=[]
        def signal_proc(ident,sig):
            signals.append((ident['pid'],sig))
            if sig==owner.signal.SIGTERM:living[ident['pid']]=False
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'old';out=root/'new'
            state={'status':'running','stage':'dev40_fit2048','child_pid':42,'server_pid':43}
            atomic_json(old/'owner_status.json',state)
            atomic_json(old/'dita/owner_status.json',{'status':'queued','stage':'waiting'})
            atomic_json(old/'vlanext_s/dev40_fit2048/progress.json',{'status':'completed_actual_development'})
            out.mkdir()
            with mock.patch.object(owner,'R85',old),mock.patch.object(owner,'OUT',out),\
                 mock.patch.object(owner,'alive',side_effect=lambda pid:living.get(pid,False)),\
                 mock.patch.object(owner,'identity',side_effect=lambda pid:expected if pid==123 else server),\
                 mock.patch.object(owner,'checked_signal',side_effect=signal_proc),\
                 mock.patch.object(owner,'cohort_audit',return_value={'complete':True}),\
                 mock.patch.object(owner.time,'sleep'):
                owner.handoff(expected)
            self.assertEqual(signals,[(123,owner.signal.SIGSTOP),(43,owner.signal.SIGTERM),
                (123,owner.signal.SIGTERM),(123,owner.signal.SIGCONT)])
            self.assertTrue((out/'HANDOFF_RECEIPT.json').exists())
            self.assertEqual(json.loads((old/'owner_status.json').read_text())['status'],
                'superseded_at_completed_cohort_boundary')

    def test_recorder_is_read_only_and_disabled_for_final(self):
        from .evaluate import observed_class
        class Plant:
            def __init__(self,*,seed):self.calls=0;self.index=0
            def observe(self):
                self.obs=SimpleNamespace(step=self.index,
                    front_rgb=np.full((16,16,3),self.index,np.uint8),
                    wrist_rgb=np.full((16,16,3),self.index+10,np.uint8))
                return self.obs
            def reset(self,task):self.calls+=1;return self.observe()
            def step(self,action):
                self.calls+=1;self.index+=1;self.last_action=action
                return self.observe(),1.,self.index==3,{'success':False}
        raw=np.zeros(64,np.float32);raw[18]=raw[22]=1;raw[24:27]=.04
        with tempfile.TemporaryDirectory() as folder:
            cls=observed_class(Plant,folder,{7},lambda env,obs:raw.copy())
            plant=cls(seed=7);obs=plant.reset(SimpleNamespace(name='pick'))
            self.assertIs(obs,plant.obs)
            actions=[np.array([0.,0.,-.01,0.,0.,0.,v]) for v in (-1.,1.,1.)]
            for action in actions:
                before=action.copy();out=plant.step(action)
                self.assertIs(out[0],plant.obs);self.assertIs(plant.last_action,action)
                np.testing.assert_array_equal(action,before)
            self.assertEqual(plant.calls,4)
            receipt=Path(folder)/'seed7/pick_INFERENCE_TIMELINE.json'
            self.assertTrue(receipt.exists())
            rows=json.loads(receipt.read_text())['actual_observations']
            self.assertEqual([r['step'] for r in rows],[0,2,3])
            cls=observed_class(Plant,folder,set(),lambda *args:(_ for _ in ()).throw(AssertionError('final must not capture')))
            plant=cls(seed=8);plant.reset(SimpleNamespace(name='pick'))
            for action in actions:plant.step(action)
            self.assertFalse((Path(folder)/'seed8').exists())


if __name__ == '__main__':
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful(): raise SystemExit(1)
