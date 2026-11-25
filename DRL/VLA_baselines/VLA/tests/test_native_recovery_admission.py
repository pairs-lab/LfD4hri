import unittest
from VLA.qualify_native_recovery_view import require_all_attempts


class NativeRecoveryAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.parent=[{'source_file':str(i),'file':str(i)+'.npz' if i<38 else None} for i in range(40)]
        self.parent_plan={'planned_attempts':40,'selected_files':[str(i) for i in range(40)],
            'selection_uses_neural_outcomes':False,'validation_dev_final_used_for_training':False,
            'prefix_actions_are_not_teacher_labels':True}
        self.plan={'pilot_only':False,'real_previous_camera_at_student_handoff':True,
            'selected_files':[str(i) for i in range(38)]}
        self.rows=[{'source_file':str(i),'file':'native'+str(i)+'.npz','success':True,'error':None} for i in range(38)]
    def check(self):return require_all_attempts(self.parent_plan,self.parent,self.plan,self.rows)
    def test_complete38_with_two_unlabeled_preserved(self):self.assertEqual(len(self.check()),38)
    def test_pilot_cannot_qualify(self):
        self.plan['pilot_only']=True
        with self.assertRaises(ValueError):self.check()
    def test_failed_attempt_cannot_be_filtered(self):
        self.rows[7]['success']=False
        with self.assertRaises(ValueError):self.check()
        del self.rows[7]
        with self.assertRaises(ValueError):self.check()
    def test_duplicate_attempt_cannot_qualify(self):
        self.rows[4]['source_file']='3'
        with self.assertRaises(ValueError):self.check()
    def test_no_validation_or_test_data(self):
        self.parent_plan['validation_dev_final_used_for_training']=True
        with self.assertRaises(ValueError):self.check()
    def test_previous_camera_must_be_real(self):
        self.plan['real_previous_camera_at_student_handoff']=False
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
