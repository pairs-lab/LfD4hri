import unittest
from VLA.audit_neural_safety_replay_r37 import classify


class SafetyReplayTests(unittest.TestCase):
    def test_nonpad_contact_is_separate_from_valid_pad_contact(self):
        d={'follower_errors_rad':[.01,-.16], 'contacts':[
            {'other_body':5,'gripper_link':4,'normal_force_n':2.},
            {'other_body':5,'gripper_link':3,'normal_force_n':5.},
            {'other_body':0,'gripper_link':2,'normal_force_n':0.}]}
        r=classify(d,{5:'object:apple',0:'ground'},{4,9})
        self.assertTrue(r['nonpad_object_contact']);self.assertFalse(r['ground_contact'])
        self.assertEqual(r['pad_contact_object_names'],['object:apple'])
        self.assertEqual(r['strongest_contact_force_N'],5.)

    def test_zero_force_manifolds_are_not_collisions_with_force(self):
        d={'follower_errors_rad':[0.], 'contacts':[{'other_body':1,'gripper_link':9,'normal_force_n':0.}]}
        r=classify(d,{1:'robot_arm'},{4,9})
        self.assertFalse(r['robot_arm_contact']);self.assertEqual(r['active_contacts'],[])


if __name__=='__main__':unittest.main()
