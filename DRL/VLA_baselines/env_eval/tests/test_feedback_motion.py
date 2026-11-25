import unittest
import numpy as np
from env_eval.feedback_motion import JointReference,CorrectionEnsemble,transport_chunk


class FeedbackMotionTests(unittest.TestCase):
    def test_replanning_does_not_reset_velocity_or_break_jerk_bound(self):
        dt=1/240;executor=JointReference(np.zeros(2),dt,max_jerk=35.)
        states=[]
        for i in range(700):
            # Includes an abrupt target change during motion, like an early
            # contact transition. The executor must keep its current state.
            target=np.array([.7,-.4]) if i<250 else np.array([.1,.25])
            states.append(executor.step(target))
        q,v,a=map(np.array,zip(*states))
        self.assertLessEqual(abs(np.diff(a,axis=0)/dt).max(),35.+1e-8)
        self.assertLessEqual(abs(a).max(),2.5+1e-8)
        self.assertLessEqual(abs(v).max(),1.15+1e-8)
        self.assertLess(np.linalg.norm(q[-1]-target),.01)

    def test_ensemble_averages_predictions_for_the_same_future_time(self):
        ensemble=CorrectionEnsemble()
        velocity=np.array([.02,-.01,0.])
        for t in (0.,.04,.08):ensemble.add(t,velocity*t,velocity)
        np.testing.assert_allclose(ensemble.sample(.12),velocity*.12,atol=1e-12)

    def test_transport_has_no_internal_stop_and_smooth_endpoints(self):
        curve=transport_chunk([-.12,-.5,.025],[.15,-.5,.07],[0,1,0,0],[0,1,0,0],.15,4.)
        times=np.linspace(0,4,1001);points=np.array([curve.sample(t)[0] for t in times])
        speed=np.linalg.norm(np.diff(points,axis=0),axis=1)/(times[1]-times[0])
        self.assertGreater(speed[200:800].min(),.025)
        self.assertLess(speed[0],1e-4);self.assertLess(speed[-1],1e-4)


if __name__=='__main__':unittest.main()
