import unittest
import numpy as np
from VLA.audit_contact_stop_candidate_r37 import project


class ContactStopTests(unittest.TestCase):
    def contact(self,link=4,normal=1.,force=1.):
        c=[None]*10;c[3]=link;c[7]=(0.,0.,normal);c[9]=force;return c
    def test_open_down_contact_stops_only_downward_component(self):
        action=np.array([.01,-.02,-.03,0,0,.04,-.9],np.float32)
        applied,stop=project(action,[self.contact()],{4,9})
        self.assertTrue(stop);self.assertEqual(applied[2],0)
        np.testing.assert_array_equal(applied[[0,1,3,4,5,6]],action[[0,1,3,4,5,6]])
        self.assertLess(action[2],0)
    def test_closed_sidewall_and_upward_actions_unchanged(self):
        for action,contact in [([0,0,-.02,0,0,0,1],self.contact()),
            ([0,0,.02,0,0,0,-1],self.contact()),([0,0,-.02,0,0,0,-1],self.contact(normal=.6)),
            ([0,0,-.02,0,0,0,-1],self.contact(force=0)),([0,0,-.02,0,0,0,-1],self.contact(link=3))]:
            applied,stop=project(action,[contact],{4,9});self.assertFalse(stop)
            np.testing.assert_array_equal(applied,np.asarray(action,np.float32))


if __name__=='__main__':unittest.main()
