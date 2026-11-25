import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from VLA.common.mimic_lag_guard import limited_target,install

SIGNS=np.array([1,-1,1,1,-1,1])


class Tests(unittest.TestCase):
    def test_all_qualified_source_error_envelope_is_inactive(self):
        q=SIGNS*.4;q[1]+=.05265534
        target,active,error=limited_target(.78,q,SIGNS)
        self.assertEqual(target,.78);self.assertFalse(active);self.assertLess(error,.08)
    def test_main_brakes_to_lagging_followers(self):
        q=SIGNS*.4;q[0]=.55
        target,active,error=limited_target(.78,q,SIGNS)
        self.assertTrue(active);self.assertLess(target,q[0]);self.assertLessEqual(target,.42+1e-12)
    def test_opening_does_not_run_away_from_blocked_follower(self):
        q=SIGNS*.6;q[0]=.44
        target,active,error=limited_target(0.,q,SIGNS)
        self.assertTrue(active);self.assertGreater(target,q[0])
    def test_readonly_target_property_and_requested_intent_are_preserved(self):
        class Grip:
            OPEN_ANGLE=0.;CLOSED_ANGLE=.78;JOINTS=range(6);SIGNS=SIGNS;body=1
            grasp_constraint=None;_latched=None;command=1.
            @property
            def target_angle(self):return (self.command+1)*self.CLOSED_ANGLE/2
            def apply_motor_control(self):self.applied_target=self.target_angle
        g=Grip();install(g)
        positions=SIGNS*.4;positions[0]=.55
        with patch("VLA.common.mimic_lag_guard.p.getJointStates",return_value=[(q,0,0,0) for q in positions]):g.apply_motor_control()
        self.assertEqual(g.command,1.);self.assertLess(g.applied_target,.55);self.assertEqual(g.mimic_governor_ticks,1)
    def test_latched_request_is_restored(self):
        g=SimpleNamespace(OPEN_ANGLE=0.,CLOSED_ANGLE=.78,JOINTS=range(6),SIGNS=SIGNS,body=1,
            grasp_constraint=3,_latched=np.array([.6]*6),command=1.)
        applied=[];g.apply_motor_control=lambda:applied.append(float(g._latched[0]));install(g)
        positions=SIGNS*.4;positions[0]=.55
        with patch("VLA.common.mimic_lag_guard.p.getJointStates",return_value=[(q,0,0,0) for q in positions]):g.apply_motor_control()
        self.assertLess(applied[0],.55);np.testing.assert_array_equal(g._latched,[.6]*6)


if __name__ == "__main__":unittest.main()
