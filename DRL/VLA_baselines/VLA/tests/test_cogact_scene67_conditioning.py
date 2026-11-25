import unittest
import torch
from VLA.cogact.scene67_conditioning import conditioned_call
from VLA.cogact.scene67_learning import StateAdapter


class Tests(unittest.TestCase):
    def test_adapter_initially_preserves_native_cognition(self):
        m=StateAdapter({'mean':[0.]*67,'std':[1.]*67})
        self.assertTrue(torch.all(m(torch.randn(3,67))==0))
        self.assertEqual(tuple(m(torch.randn(3,67)).shape),(3,1,4096))
    def test_native_cfg_positional_does_not_change_unconditional_branch(self):
        z=torch.randn(2,1,4096);original=z.clone();delta=torch.ones(1,1,4096)
        actual=conditioned_call(lambda x,t,z:z,(None,None,z),{},delta)
        torch.testing.assert_close(actual[0],z[0]+1)
        torch.testing.assert_close(actual[1],z[1],rtol=0,atol=0)
        torch.testing.assert_close(z,original,rtol=0,atol=0)
    def test_native_keyword_no_cfg(self):
        z=torch.zeros(1,1,4096)
        actual=conditioned_call(lambda x,t,z:z,(None,None),{'z':z},torch.ones_like(z))
        self.assertTrue(torch.all(actual==1))


if __name__=='__main__':unittest.main()
