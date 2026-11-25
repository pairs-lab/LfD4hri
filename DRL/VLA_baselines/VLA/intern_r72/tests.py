import unittest
import numpy as np
import torch
from torch import nn
from VLA.intern_r72.native import LoRA
from VLA.intern_r72.data import targets,decode,measured_feedback


class Contracts(unittest.TestCase):
    def test_native_normalization_roundtrip(self):
        a=np.random.default_rng(7).normal(size=(50,7)).astype(np.float32)
        stats={'mean':[.01]*6+[0.],'std':[.03]*6+[1.]}
        pad=np.zeros(50,bool);out=targets(a,pad,stats)
        np.testing.assert_allclose(decode(out,stats),a,atol=3e-7)
        self.assertFalse(out[:,7:].any())

    def test_nonexistent_motion_is_not_a_label(self):
        a=np.ones((50,7),np.float32);b=a.copy();b[4:]=1000
        pad=np.arange(50)>=4;stats={'mean':[0.]*7,'std':[1.]*7}
        np.testing.assert_array_equal(targets(a,pad,stats),targets(b,pad,stats))
        self.assertFalse(targets(a,pad,stats)[pad].any())

    def test_lora_is_pretrained_identity_and_has_gradient(self):
        base=nn.Linear(4,3);layer=LoRA(base);x=torch.randn(2,4)
        torch.testing.assert_close(layer(x),base(x),rtol=0,atol=0)
        layer(x).sum().backward();self.assertGreater(float(layer.B.grad.abs().sum()),0.)
        self.assertFalse(base.weight.requires_grad)

    def test_feedback_describes_sensors_not_action_advice(self):
        raw=np.zeros(64,np.float32);raw[30]=raw[31]=1
        text=measured_feedback(raw)
        self.assertIn('wrong object attached',text)
        for word in ('should','waypoint','phase=','success='):self.assertNotIn(word,text)


if __name__=='__main__':unittest.main()
