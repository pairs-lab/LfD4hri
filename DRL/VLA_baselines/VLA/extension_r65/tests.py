"""No GPU, weights or actual model SR: contract and native tiny-head regressions."""
import unittest
import numpy as np


class Tests(unittest.TestCase):
    def test_jepa_config_explicit_embodiment_and_gripper(self):
        from .jepa import config
        c=config()
        self.assertEqual((c.state_dim,c.action_dim,c.chunk_size),(67,7,7))
        self.assertTrue(c.freeze_qwen)
        self.assertFalse(c.enable_world_model)
        self.assertFalse(c.binarize_gripper_action)
        self.assertFalse(c.pre_snap_gripper_action)
        self.assertFalse(c.use_relative_actions)
        self.assertEqual(len(c.image_features),3)

    def test_normalization_no_joint_subtraction_or_gripper_flip(self):
        from .jepa import batch
        stat={'state':{'mean':[0.]*67,'std':[1.]*67},
              'action':{'mean':[0.]*7,'std':[.03]*6+[1.]}}
        x={'numeric67':np.arange(67,dtype=np.float32)/10,'instruction':'Pick object',
           'action':np.array([[.01,.02,-.01,0,0,0,1.]]*7,np.float32),'action_is_pad':np.zeros(7,bool),
           **{k:np.zeros((384,384,3),np.uint8) for k in ('front','wrist','map')}}
        b=batch(x,stat,'cpu',True);a=b['action'].numpy()[0]*np.asarray(stat['action']['std'])
        np.testing.assert_allclose(a,x['action'],atol=1e-7)
        self.assertEqual(float(b['action'][0,0,6]),1.)
        self.assertLessEqual(float(b['observation.state'].abs().max()),5.)
        self.assertFalse(bool(b['action_is_pad'].any()))

    def test_native_tiny_DiT_gradient_and_padding(self):
        import torch
        from .jepa import config
        from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead
        c=config();c.action_model_type='DiT-test';c.action_num_heads=2;c.action_attention_head_dim=8
        c.action_hidden_size=32;c.action_num_layers=2;c.repeated_diffusion_steps=1;c.action_dropout=0.
        model=VLAJEPAActionHead(c,cross_attention_dim=16)
        conditions=torch.randn(1,32,16);actions=torch.randn(1,7,7);state=torch.randn(1,1,67)
        mask=torch.tensor([[False,False,False,True,True,True,True]])
        torch.manual_seed(17);one=model(conditions,actions,state,mask)
        # No hard oracle/action machinery: native loss is differentiable.
        self.assertTrue(bool(torch.isfinite(one)));one.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters()))
        torch.manual_seed(81);p=model.predict_action(conditions,state)
        torch.manual_seed(81);q=model.predict_action(conditions,state)
        torch.testing.assert_close(p,q,atol=0,rtol=0)
        self.assertEqual(tuple(p.shape),(1,7,7))

    def test_native_flow_bfloat16_Qwen_to_FP32_DiT_boundary(self):
        import torch
        from .jepa import config,configure_fp32_action_loss
        from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel
        c=config();c.action_model_type='DiT-test';c.action_num_heads=2;c.action_attention_head_dim=8
        c.action_hidden_size=32;c.action_num_layers=2;c.repeated_diffusion_steps=1;c.action_dropout=0.
        native=VLAJEPAModel.__new__(VLAJEPAModel);torch.nn.Module.__init__(native)
        native.config=c;native.action_model=VLAJEPAActionHead(c,cross_attention_dim=16)
        configure_fp32_action_loss(native)
        seen=[]
        native.action_model.state_encoder.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
        state=torch.randn(1,1,67);hidden=torch.randn(1,32,16,dtype=torch.bfloat16)
        loss=native._action_loss(hidden,torch.randn(1,7,7),state,torch.zeros(1,7,dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(loss)));loss.backward()
        torch.testing.assert_close(seen[0],state,atol=0,rtol=0)
        self.assertTrue(all(p.dtype==torch.float32 for p in native.action_model.parameters()))

    def test_privileged_feature_coverage_same_XVLA_source(self):
        from .common import privileged_coverage
        from VLA.scene64_schema import FEATURE_NAMES
        for model in ('vla_jepa','pi0_fast'):
            c=privileged_coverage(model)
            self.assertEqual(c['source_fields64'],list(FEATURE_NAMES))
            self.assertEqual(len(c['source_fields64']),64)
            self.assertEqual(len(c['views']),3)
            self.assertTrue(c['no_future_values_teacher_actions_waypoints_or_success_flags'])
        self.assertEqual(len(privileged_coverage('vla_jepa')['numeric_names67']),67)

    def test_prefix_selection_complete_development_only(self):
        from .run import choose_prefix
        rows=[{'success':False,'safety_violation':None} for _ in range(8)]
        p={'completed_trials':8,'rows':rows}
        self.assertEqual(choose_prefix([(3,p),(7,p)]),7)
        with self.assertRaises(ValueError):choose_prefix([(3,{'completed_trials':7,'rows':rows[:-1]})])

    def test_no_future_seed_forecasts_from_partial(self):
        from VLA.cross_vla800_r62.report import forecast
        with self.assertRaises(ValueError):forecast([])


if __name__=='__main__':unittest.main()
