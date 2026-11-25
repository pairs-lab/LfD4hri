import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.family_transfer_r54.data import Data,action_stats
from VLA.family_transfer_r54.pi0 import pi_sample
from VLA.family_transfer_r54.run import choose_prefix,score


class TransferTests(unittest.TestCase):
    def test_frame_addresses_no_cross_episode(self):
        d=Data.__new__(Data);d.ends=[2,5]
        self.assertEqual([d.address(i) for i in range(5)],[(0,0),(0,1),(1,0),(1,1),(1,2)])
        for index in (-1,5):
            with self.assertRaises(IndexError):d.address(index)

    def sample(self):
        raw=np.zeros(64,np.float32);raw[:6]=[4,-4,1,2,3,0];raw[6]=.25
        action=np.arange(35,dtype=np.float32).reshape(5,7)/100;action[:,6]=[-1,1,-1,1,-1]
        return {'raw64':raw,'action':action,'front':np.zeros((2,2,3),np.uint8),
                'wrist':np.ones((2,2,3),np.uint8),'map':np.full((2,2,3),2,np.uint8),'prompt':'full64'}

    def test_pi0_actions_are_EEF_NOT_joint_delta(self):
        x=self.sample();before=x['action'].copy();p=pi_sample(x)
        np.testing.assert_array_equal(p['actions'][:,:6],before[:,:6])
        np.testing.assert_array_equal(p['actions'][:,6],[0,1,0,1,0])
        np.testing.assert_array_equal(x['action'],before)

    def test_all_camera_roles_preserved(self):
        x=self.sample();p=pi_sample(x)
        for source,target in [('front','image'),('wrist','wrist_image'),('map','map_image')]:
            np.testing.assert_array_equal(x[source],p[target])

    def test_action_stats_do_not_collapse_rotation(self):
        a=np.zeros((10,7),np.float32);a[:5,6]=-1;a[5:,6]=1
        data=SimpleNamespace(rows=[None],numeric=lambda _: {'canonical7':a})
        s=action_stats(data)
        self.assertGreaterEqual(s['std'][3],.0049999)
        self.assertGreaterEqual(s['q99'][3]-s['q01'][3],.0599999)
        self.assertEqual(s['mean'][6],0.)
        c=action_stats(data,closure01=True);self.assertEqual(c['mean'][6],.5)

    def result(self,prefix,successes,n=8,checkpoint='same'):
        return {'prefix':prefix,'successes':successes,'completed_trials':n,'planned_trials':8,
                'checkpoint':checkpoint,'cohort':'same_dev','rows':[{'success':i<successes} for i in range(n)]}

    def test_global_prefix_not_per_task_splicing(self):
        results={Path('a'):self.result(2,3),Path('b'):self.result(4,4),Path('c'):self.result(5,4)}
        with patch('VLA.family_transfer_r54.run.read',side_effect=results.__getitem__):
            self.assertEqual(choose_prefix(list(results)),5)

    def test_reject_partial_cohort_selection(self):
        with patch('VLA.family_transfer_r54.run.read',return_value=self.result(2,3,n=7)):
            with self.assertRaises(ValueError):choose_prefix([Path('a')])

    def test_reject_mixed_weights(self):
        results={Path('a'):self.result(2,3),Path('b'):self.result(5,4,checkpoint='different')}
        with patch('VLA.family_transfer_r54.run.read',side_effect=results.__getitem__):
            with self.assertRaises(ValueError):choose_prefix(list(results))


if __name__=='__main__':unittest.main()
