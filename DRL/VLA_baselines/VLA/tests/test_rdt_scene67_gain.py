import unittest
import torch
from VLA.rdt.scene67_native import gated_state,install_gains,NEW_SLOTS


class Dummy(torch.nn.Module):
    def compute_loss(self,state_tokens):return state_tokens.sum()
    def predict_action(self,state_tokens):return state_tokens


class Tests(unittest.TestCase):
    def test_new_input_gains_zero_legacy_untouched(self):
        m=install_gains(Dummy());state=torch.randn(2,1,128);before=state.clone()
        value=m.predict_action(state_tokens=state)
        torch.testing.assert_close(state,before)
        self.assertTrue(torch.all(value[...,list(NEW_SLOTS)]==0))
        keep=[i for i in range(128) if i not in NEW_SLOTS]
        torch.testing.assert_close(value[...,keep],state[...,keep],rtol=0,atol=0)
    def test_actual_gain_gradients(self):
        m=install_gains(Dummy());loss=m.compute_loss(torch.ones(2,1,128));loss.backward()
        torch.testing.assert_close(m.scene67_gain.grad,torch.full((49,),2.))
    def test_gain_is_a_saved_neural_parameter(self):
        m=install_gains(Dummy());self.assertIn('scene67_gain',m.state_dict())
        self.assertEqual(m.scene67_gain.dtype,torch.float32)
        with self.assertRaises(ValueError):install_gains(m)


if __name__=='__main__':unittest.main()
