import unittest
import numpy as np
from VLA.audit_native_policy_fit_r37 import metrics,select_frames,to_canonical


class NativePolicyFitTests(unittest.TestCase):
    def test_selection_is_substantive_before_action_and_preserves_close(self):
        states=np.zeros((6,64));actions=np.zeros((6,7));actions[:,6]=-1
        actions[1,0]=.03;actions[3,6]=1;states[4:,17]=1;actions[4:,2]=.02
        self.assertEqual(select_frames(actions,states),[('xy_alignment',1),('near_unheld_close',3),('verified_hold_transport',4)])

    def test_zero_motion_is_not_a_fake_accurate_probe(self):
        actions=np.zeros((5,7));actions[:,6]=-1
        with self.assertRaisesRegex(ValueError,'substantive'):select_frames(actions,np.zeros((5,64)))

    def test_open_and_closure_conversion_exactly_match_runtime_client(self):
        opened=np.array([.01,.02,.03,0,0,.05,.75],np.float32)
        converted=to_canonical(opened,'native_delta_ee_and_open_gripper')
        np.testing.assert_array_equal(converted[:6],opened[:6]);self.assertEqual(converted[6],-.5)
        self.assertEqual(opened[6],.75)
        result=metrics(opened,'native_delta_ee_and_open_gripper',converted,np.zeros(7))
        self.assertEqual(result['XYZ_error_mm'],0);self.assertTrue(result['gripper_sign_matches'])

    def test_rdt_joint_wrap_error_does_not_compare_radians_to_mm(self):
        gold=np.array([0,0,0,0,0,0,1],np.float32)
        joints=np.array([-np.pi+.01,0,0,0,0,0,1],np.float32)
        predicted=joints.copy();predicted[0]+=2*np.pi
        result=metrics(predicted,'absolute_joint_targets_and_closure',gold,joints)
        self.assertLess(result['native_joint_max_error_rad'],1e-6)
        self.assertNotIn('XYZ_error_mm',result);self.assertTrue(result['gripper_sign_matches'])

    def test_invalid_neural_prediction_cannot_count_as_zero_error(self):
        with self.assertRaises(ValueError):metrics([float('nan')]*7,'canonical_delta_ee_7d',[0]*7,[0]*7)


if __name__=='__main__':unittest.main()
