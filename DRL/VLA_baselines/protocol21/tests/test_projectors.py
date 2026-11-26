import unittest
from types import SimpleNamespace
import torch
from torch import nn
from VLA.new_vlas.privileged_inputs import expand_adapter_proprio,expand_xvla_proprio


class DomainLinear(nn.Module):
    """Native domain-linear tensor layout, small CPU fixture only."""
    def __init__(self,input_size,output_size,num_domains):
        super().__init__();self.input_size=input_size;self.output_size=output_size
        self.fc=nn.Embedding(num_domains,input_size*output_size);self.bias=nn.Embedding(num_domains,output_size)


class ProjectorTests(unittest.TestCase):
    def test_adapter_native_columns_preserved_and_new_zeros(self):
        obj=SimpleNamespace(fc1=nn.Linear(8,16))
        old=obj.fc1.weight.detach().clone();bias=obj.fc1.bias.detach().clone()
        expand_adapter_proprio(obj)
        torch.testing.assert_close(obj.fc1.weight[:,:8],old,rtol=0,atol=0)
        torch.testing.assert_close(obj.fc1.bias,bias,rtol=0,atol=0)
        self.assertEqual(obj.fc1.in_features,29)
        self.assertEqual(torch.count_nonzero(obj.fc1.weight[:,8:]),0)
        self.assertEqual(obj.fc1(torch.randn(2,29)).shape,(2,16))

    def test_xvla_pretrained_time_columns_shifted_not_overwritten(self):
        obj=SimpleNamespace(action_encoder=DomainLinear(20+20+4,12,2),dim_action=20,dim_time=4)
        old=obj.action_encoder.fc.weight.detach().reshape(2,44,12).clone()
        expand_xvla_proprio(obj,20)
        new=obj.action_encoder.fc.weight.detach().reshape(2,65,12)
        torch.testing.assert_close(new[:,:40],old[:,:40],rtol=0,atol=0)
        torch.testing.assert_close(new[:,61:],old[:,40:],rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(new[:,40:61]),0)

    def test_old_extension_width_rejected(self):
        with self.assertRaises(ValueError):expand_adapter_proprio(SimpleNamespace(fc1=nn.Linear(8,8)),extra_dim=67)
