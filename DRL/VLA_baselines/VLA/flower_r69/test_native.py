import unittest
import numpy as np
import torch
from torch import nn
from VLA.flower_r69.native import LoRALinear, action_targets, canonical_actions, image_tensor, mapped_state, MEAN, STD
from VLA.flower_r69.native import FlowerVLA
from VLA.flower_r69.loss import loss as padded_loss


class ContractTests(unittest.TestCase):
    def test_native_key_mapping(self):
        a = torch.tensor([3.])
        self.assertIs(mapped_state({'agent.dit.0.mlp.c_fc1.weight': a})['dit.0.mlp.fc1.weight'], a)

    def test_no_mapping_collision(self):
        with self.assertRaises(ValueError):
            mapped_state({'agent.dit.mlp.c_fc1.weight': torch.ones(1),
                          'agent.dit.mlp.fc1.weight': torch.ones(1)})

    def test_no_unknown_prefix(self):
        with self.assertRaises(ValueError):
            mapped_state({'mystery.weight': torch.ones(1)})

    def test_lora_starts_as_exact_native_and_gets_gradient(self):
        layer = nn.Linear(5, 4)
        x = torch.randn(2, 5)
        expected = layer(x).detach()
        wrapped = LoRALinear(layer, 2)
        torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)
        wrapped(x).sum().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertGreater(float(wrapped.B.grad.abs().sum()), 0)

    def test_image_order_scale_and_normalization(self):
        image = np.zeros((224, 224, 3), np.uint8)
        image[..., 0] = 255
        t = image_tensor(image)
        expected = (np.array([1., 0., 0.])-np.array(MEAN))/np.array(STD)
        np.testing.assert_allclose(t[:, 100, 100].numpy(), expected, atol=1e-6)

    def test_pre_normalized_or_bgr_dtype_not_accepted(self):
        with self.assertRaises(ValueError):
            image_tensor(np.zeros((224, 224, 3), np.float32))

    def test_native_open_gripper_and_roundtrip(self):
        stats = {'adapter': {'action': {'q01': [-.045]*3+[-.03]*3+[0],
                                      'q99': [.045]*3+[.03]*3+[1]}}}
        a = np.array([[.01, -.02, .03, .001, -.002, .003, -1.],
                      [-.01, .02, -.03, -.001, .002, -.003, 1.]], np.float32)
        target = action_targets(a, stats)
        self.assertEqual(target[0, 6], 1.)
        self.assertEqual(target[1, 6], 0.)
        np.testing.assert_allclose(canonical_actions(target, stats), a, atol=1e-8)

    def test_invalid_action_dimension_rejected(self):
        with self.assertRaises(ValueError):
            action_targets(np.zeros((10, 8)), {})

    def test_flow_all_valid_matches_authors_actual_objective(self):
        class Fake(nn.Module):
            sampling_type='uniform'
            accelerator=None
            def __init__(self):
                super().__init__()
                self.weight=nn.Parameter(torch.tensor(.23))
            def dit_forward(self,z,t,cond):
                return self.weight*z+t[:,None,None]
        model=Fake()
        model.action_space_index=type('Registry',(),{'action_spaces':{'eef_delta':1},
            'get_action_dim':lambda this,index:7})()
        cond={'action_type':torch.ones(2,dtype=torch.long)}
        actions=torch.randn(2,10,7)
        torch.manual_seed(19)
        original,_=FlowerVLA.rf_loss(model,cond,actions)
        torch.manual_seed(19)
        masked=padded_loss(model,cond,actions,torch.zeros(2,10,dtype=torch.bool))
        torch.testing.assert_close(original,masked,rtol=0,atol=0)

    def test_tail_padding_cannot_change_loss_or_gradient(self):
        class Causal(nn.Module):
            sampling_type='uniform'
            def __init__(self):
                super().__init__();self.weight=nn.Parameter(torch.tensor(.23))
            def dit_forward(self,z,t,cond):
                return z*self.weight
        model=Causal();cond={'action_type':torch.ones(1,dtype=torch.long)}
        target=torch.randn(1,10,7);mask=torch.zeros(1,10,dtype=torch.bool);mask[:,5:]=True
        values=[];grads=[]
        for offset in (0.,100.):
            changed=target.clone();changed[:,5:]+=offset
            torch.manual_seed(5);model.zero_grad()
            value=padded_loss(model,cond,changed,mask);value.backward()
            values.append(value.detach());grads.append(model.weight.grad.clone())
        torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
        torch.testing.assert_close(grads[0],grads[1],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
