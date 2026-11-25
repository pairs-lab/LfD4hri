from types import SimpleNamespace
import unittest
import torch

from VLA.rdt.signal_precision import PROFILE, install_fp32_signal_islands, signal_dtype


class SignalIslandsTest(unittest.TestCase):
    def runner(self):
        first,last=torch.nn.Linear(2,2),torch.nn.Linear(2,2)
        with torch.no_grad():
            for layer in (first,last):
                layer.weight.copy_(torch.eye(2));layer.bias.zero_()
        return SimpleNamespace(state_adaptor=first,model=SimpleNamespace(final_layer=last))

    def test_default_profile_unchanged_and_unknown_profile_rejected(self):
        self.assertEqual(signal_dtype('bf16'),torch.bfloat16)
        self.assertEqual(signal_dtype(PROFILE),torch.float32)
        with self.assertRaises(ValueError):signal_dtype('floatish')

    def test_autocast_does_not_round_away_fine_absolute_joint_signal(self):
        runner=self.runner()
        values=torch.tensor([[2.,-1.5],[2.002,-1.498]])
        with torch.autocast('cpu',dtype=torch.bfloat16):
            legacy=runner.model.final_layer(runner.state_adaptor(values))
        torch.testing.assert_close(legacy[0],legacy[1],rtol=0,atol=0)
        install_fp32_signal_islands(runner)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            repaired=runner.model.final_layer(runner.state_adaptor(values))
        self.assertEqual(repaired.dtype,torch.float32)
        torch.testing.assert_close(repaired,values,rtol=0,atol=0)

    def test_no_parameter_names_or_values_change_and_gradients_still_flow(self):
        runner=self.runner()
        before={k:v.clone() for k,v in runner.state_adaptor.state_dict().items()}
        install_fp32_signal_islands(runner);install_fp32_signal_islands(runner)
        self.assertEqual(set(before),set(runner.state_adaptor.state_dict()))
        for k,v in before.items():torch.testing.assert_close(v,runner.state_adaptor.state_dict()[k],rtol=0,atol=0)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=runner.model.final_layer(runner.state_adaptor(torch.tensor([[2.002,1.001]])))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(runner.state_adaptor.weight.grad).all())
        self.assertGreater(float(runner.state_adaptor.weight.grad.abs().max()),0)

    def test_bf16_loaded_weights_cannot_be_promoted_silently(self):
        runner=self.runner();runner.state_adaptor.bfloat16()
        with self.assertRaises(ValueError):install_fp32_signal_islands(runner)

    def test_actual_official_sampler_keeps_fp32_latent_and_joint_output(self):
        from VLA.rdt import server  # Adds the pinned official source path.
        from models.rdt_runner import RDTRunner
        config={'rdt':{'hidden_size':64,'depth':1,'num_heads':4},
                'lang_adaptor':'linear','img_adaptor':'linear','state_adaptor':'linear',
                'noise_scheduler':{'num_train_timesteps':20,'num_inference_timesteps':2,
                    'beta_schedule':'squaredcos_cap_v2','prediction_type':'sample','clip_sample':False}}
        model=RDTRunner(action_dim=128,pred_horizon=5,config=config,lang_token_dim=16,
                       img_token_dim=16,state_token_dim=128,max_lang_cond_len=4,img_cond_len=6,
                       dtype=torch.float32).eval()
        with torch.no_grad():
            model.model.final_layer.ffn_final.fc2.bias[0]=2.002
        saved={key:value.clone() for key,value in model.state_dict().items()}
        inputs={'lang_tokens':torch.zeros(1,4,16),'lang_attn_mask':torch.ones(1,4,dtype=torch.bool),
                'img_tokens':torch.zeros(1,6,16),'state_tokens':torch.zeros(1,1,128),
                'action_mask':torch.ones(1,1,128,dtype=torch.bfloat16),'ctrl_freqs':torch.tensor([10.])}
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            original=model.predict_action(**inputs)
        self.assertEqual(original.dtype,torch.bfloat16)
        install_fp32_signal_islands(model)
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            repaired=model.predict_action(**inputs)
        self.assertEqual(repaired.dtype,torch.float32)
        torch.testing.assert_close(repaired[...,0],torch.full((1,5),2.002),rtol=0,atol=1e-6)
        self.assertGreater(float((repaired[...,0]-original[...,0].float()).abs().max()),.001)
        for key,value in saved.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
