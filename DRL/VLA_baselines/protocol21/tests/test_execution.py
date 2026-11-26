import unittest
import numpy as np
from protocol21.execution import Executor,prediction
from protocol21.inputs import rotation
from protocol21.specs import SPECS
from protocol21.tests.test_inputs import frame,stats


class FakeRuntime:
    """Scheduler fixture, never a replacement model used by the native CLI."""
    def __init__(self,family):self.family=family;self.calls=[]
    def predict(self,s,seed=0):
        self.calls.append(s)
        n=SPECS[self.family].horizon
        if self.family=='xvla':return np.tile(s['state'][:20],(n,1))
        x=np.zeros((n,7),np.float32);x[:,6]=-1
        return x


class ExecutionTests(unittest.TestCase):
    def test_HK_scheduling_all_models(self):
        for family,spec in SPECS.items():
            with self.subTest(family=family):
                m=FakeRuntime(family);e=Executor(family,m,stats())
                for t in range(spec.execute+1):
                    a=e.step(frame(t*.1));self.assertEqual(a.shape,(7,));self.assertTrue(np.isfinite(a).all())
                    e.record_execution({'executed_canonical_action':a})
                self.assertEqual(len(m.calls),2)
                e.reset();self.assertEqual(e.queries,0);self.assertFalse(e.queue)

    def test_xvla_absolute_decode_uses_current_pose_each_step(self):
        m=FakeRuntime('xvla');e=Executor('xvla',m,stats())
        f=frame();e.step(f)
        moved=frame(.1,ee_position=f.ee_position+np.array([.01,0,0]))
        action=e.step(moved)
        self.assertAlmostEqual(action[0],-.01,places=6)
        self.assertEqual(len(m.calls),1)

    def test_padding_history_at_first_query(self):
        for family in ('dita','seer','vlanext_s'):
            m=FakeRuntime(family);e=Executor(family,m,stats());e.step(frame())
            self.assertEqual(m.calls[0]['history_times'],[0.]*SPECS[family].history)

    def test_actions_cannot_enter_observation(self):
        m=FakeRuntime('flower');e=Executor('flower',m,stats());e.step(frame())
        self.assertNotIn('action',m.calls[0]);self.assertNotIn('future_front_labels',m.calls[0])

    def test_offline_labels_stripped_before_native_predict(self):
        from protocol21.inputs import Inputs
        inp=Inputs('flower',stats());inp.push(frame());s=inp.sample()
        s['action']=np.ones((10,7));s['future_front_labels']=np.zeros((7,32,32,3))
        m=FakeRuntime('flower');prediction(m,'flower',s,stats())
        self.assertNotIn('action',m.calls[0]);self.assertNotIn('future_front_labels',m.calls[0])

    def test_native_binary_gripper_not_an_added_neutral_band(self):
        class Runtime(FakeRuntime):
            def predict(self,s,seed=0):
                x=super().predict(s,seed);x[:,9]=.51;return x
        e=Executor('xvla',Runtime('xvla'),stats())
        self.assertEqual(e.step(frame())[6],1)

    def test_seer_requires_actual_safety_clamped_feedback(self):
        e=Executor('seer',FakeRuntime('seer'),stats());f=frame();e.step(f)
        with self.assertRaises(ValueError):e.step(frame(.1))
        e.record_execution({'executed_canonical_action':np.array([.01,0,0,0,0,0,-1])})
        self.assertAlmostEqual(e.previous_command[0,3],f.ee_position[0]+.01,places=6)
        e.step(frame(.1))
