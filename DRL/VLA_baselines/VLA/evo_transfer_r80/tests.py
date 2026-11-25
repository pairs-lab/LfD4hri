"""No motor actions: normalization, temporal masking and zero-tail extension."""
import unittest
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from .data import normalize, denormalize, targets, canonical, state
from .native import expand_state


class Contracts(unittest.TestCase):
    def setUp(self):
        self.stats = {'action': {'min': [-.05]*3+[-.25]*3+[0.],
                                 'max': [.05]*3+[.25]*3+[1.]},
                      'native8': {'min': [-1.]*8, 'max': [1.]*8},
                      'scene67': {'mean': [0.]*67, 'std': [1.]*67}}
        self.sample = {'action': np.zeros((50, 7), np.float32),
            'action_is_pad': np.arange(50)>=3, 'native8': np.zeros(8,np.float32),
            'numeric67': np.arange(67,dtype=np.float32)/100}
        self.sample['action'][:, 6] = -1

    def test_bounds_roundtrip(self):
        x = np.random.default_rng(80).uniform(-1,1,(200,7)).astype(np.float32)
        np.testing.assert_allclose(normalize(denormalize(x,self.stats['action']),self.stats['action']),x,atol=2e-7)

    def test_real_future_and_channels(self):
        a,m=targets(self.sample,self.stats)
        self.assertEqual(m.sum(),21)
        self.assertFalse(a[3:].any());self.assertFalse(m[:,7:].any())

    def test_gripper_once(self):
        for closure in (-1.,1.):
            self.sample['action'][:,6]=closure
            a,_=targets(self.sample,self.stats)
            decoded=canonical(a,self.stats)
            self.assertAlmostEqual(float(decoded[0,6]),closure)

    def test_complete_numeric(self):
        x=state(self.sample,self.stats)
        self.assertEqual(x.shape,(91,))
        np.testing.assert_array_equal(x[24:],self.sample['numeric67'])

    def test_native_state_preserved(self):
        old=nn.Linear(24,1024);weight=old.weight.detach().clone();bias=old.bias.detach().clone()
        head=SimpleNamespace(state_encoder=SimpleNamespace(fc1=SimpleNamespace(linear=old)))
        expand_state(head);new=head.state_encoder.fc1.linear
        self.assertTrue(torch.equal(new.weight[:,:24],weight));self.assertTrue(torch.equal(new.bias,bias))
        self.assertEqual(int(new.weight[:,24:].count_nonzero()),0)


if __name__=='__main__':
    unittest.main()
