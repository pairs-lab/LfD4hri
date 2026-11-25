import unittest
from collections import defaultdict
import numpy as np
from VLA.flower_rotation_r76.sampler import Data,phase_flags
from VLA.target_balance_r74.sampler import Data as Parent


def synthetic():
    d=Data.__new__(Data);d.groups={};d.rows=[];cache={}
    d.transitions=defaultdict(lambda:defaultdict(dict));d.rotations=defaultdict(dict)
    for task in ('touch','pick','move','pick_place'):
        d.groups[task]={'complete_original':{},'fresh_corrective':{},'pose_corrective':{}}
        for target in ('a','b'):
            ei=len(d.rows);d.rows.append({'task':task,'pick_name':target})
            d.groups[task]['complete_original'][target]=[ei]
            a=np.zeros((8,7),np.float32);a[:,6]=-1;a[1,0]=.02;a[2,5]=.08
            a[3,2]=.003  # Outside the old lateral/vertical/quiet/close flags.
            cache[ei]={'canonical7':a,'scene64':np.zeros((8,64),np.float32)}
            d.rotations[task][target]=[(ei,np.array([2]))]
    d.numeric=lambda ei:cache[ei]
    return d,cache


class SamplingTests(unittest.TestCase):
    def test_rotation_only_excluded_by_old_flags(self):
        d,c=synthetic();n=c[0];flags=phase_flags(n['scene64'],n['canonical7'])
        self.assertFalse(any(f[2] for f in flags))
        self.assertFalse(any(f[3] for f in flags))

    def test_full_support_draws_all_original_frames(self):
        d,c=synthetic();rng=np.random.default_rng(76);seen=set();before={i:x['canonical7'].copy() for i,x in c.items()}
        for _ in range(12000):
            ei,fi=d.draw(rng);seen.add((ei,fi));self.assertTrue(0<=fi<8)
        self.assertEqual(len(seen),len(c)*8)
        for ei,n in c.items():np.testing.assert_array_equal(before[ei],n['canonical7'])

    def test_rotation_start_is_actually_sampled(self):
        d,c=synthetic();rng=np.random.default_rng(76);count=0
        for _ in range(1000):
            ei,fi=d.draw(rng);count+=d.last_sampling_mode=='rotation_start'
            if d.last_sampling_mode=='rotation_start':self.assertEqual(fi,2)
        self.assertGreater(count,100)

    def test_no_changed_sample_or_observation_encoder(self):
        self.assertIs(Data.at,Parent.at)


if __name__=='__main__':unittest.main()
