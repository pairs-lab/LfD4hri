"""Actual official DiT-S/MLP forward/backward on CPU, NOT CUDA qualification."""
from pathlib import Path
import sys
import unittest
import torch
from VLA.cogact.scene67_learning import StateAdapter,OFFICIAL


class TestNativeHead(unittest.TestCase):
    def test_original_action_head_loss_and_new_observation_gradient(self):
        root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'VLA/third_party/cogact'))
        from action_model.action_model import ActionModel
        torch.set_num_threads(2);torch.manual_seed(20261015)
        head=ActionModel(token_size=4096,model_type='DiT-S',in_channels=7,future_action_window_size=15,past_action_window_size=0)
        original=torch.load(OFFICIAL,map_location='cpu',mmap=True,weights_only=True)['model']['action_model']
        head.load_state_dict(original,strict=True)
        adapter=StateAdapter({'mean':[0.]*67,'std':[1.]*67})
        states=torch.randn(2,67);features=torch.randn(2,1,4096);actions=torch.rand(2,16,7)
        loss=head.loss(actions,features+adapter(states));self.assertTrue(torch.isfinite(loss))
        loss.backward();gradient=adapter.net[-1].weight.grad
        self.assertIsNotNone(gradient);self.assertTrue(torch.any(gradient!=0));self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(all(p.dtype==torch.float32 for p in head.parameters()))


if __name__=='__main__':unittest.main()
