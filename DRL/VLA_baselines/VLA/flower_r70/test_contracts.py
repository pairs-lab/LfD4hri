import unittest
from VLA.flower_r70.watch import select_prefix
from VLA.flower_r70.report import classify


def pilot(nsuccess,safety=0,n=8):
    return {'completed_trials':n,'rows':[{'success':i<nsuccess,'safety_violation':i<safety} for i in range(n)]}


class Tests(unittest.TestCase):
    def test_prefix_prefers_real_success_then_safety_then_closed_loop(self):
        self.assertEqual(select_prefix({1:pilot(3),5:pilot(4),10:pilot(2)}),5)
        self.assertEqual(select_prefix({1:pilot(4,0),5:pilot(4,1),10:pilot(4,0)}),1)

    def test_incomplete_pilot_cannot_win(self):
        with self.assertRaises(ValueError):select_prefix({1:pilot(8),5:pilot(3,n=4)})

    def test_failure_classifier_does_not_invent_memory_loss(self):
        row={'success':False,'safety_violation':'ground_collision'}
        self.assertEqual(classify(row,[]),'safety_abort')
        self.assertEqual(classify({'success':True},[]),'success')

    def test_all_failure_trials_remain_in_pilot(self):
        self.assertEqual(select_prefix({1:pilot(0),5:pilot(0),10:pilot(0)}),1)


if __name__=='__main__':unittest.main()
