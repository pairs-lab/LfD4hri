"""CPU tests of the actual native input layers; no surrogate policy or SR."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from VLA.new_vlas.privileged_inputs import append_numeric67, expand_adapter_proprio, expand_xvla_proprio, inject_native_lora
from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer


class PrivilegedInputTests(unittest.TestCase):
    def test_all_linear_lora_preserves_native_instance_and_initial_prediction(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))
        value = torch.randn(2, 8)
        expected = model(value).detach()
        actual_model = inject_native_lora(model, 4)
        self.assertIs(actual_model, model)
        torch.testing.assert_close(actual_model(value), expected, rtol=0, atol=0)
        actual_model(value).square().mean().backward()
        gradients = [p.grad for n, p in actual_model.named_parameters() if 'lora_B' in n]
        self.assertTrue(gradients)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients))

    def test_xvla_preserves_action_proprio_and_time_columns(self):
        torch.manual_seed(123)
        model = SoftPromptedTransformer(hidden_size=32, multi_modal_input_size=16,
            depth=1, num_heads=4, mlp_ratio=2, num_domains=3, dim_action=20,
            dim_propio=20, dim_time=8, len_soft_prompts=2)
        old = model.action_encoder
        domains = torch.tensor([0, 2])
        x = torch.randn(2, 4, 48)
        expected = old(x, domains)
        proof = expand_xvla_proprio(model, 20)
        expanded = torch.cat((x[..., :40], torch.randn(2, 4, 67), x[..., 40:]), -1)
        actual = model.action_encoder(expanded, domains)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(proof['new_proprio_dim'], 87)
        actual.square().mean().backward()
        gradient = model.action_encoder.fc.weight.grad.reshape(3, 115, 32)[:, 40:107]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_adapter_real_projector_preserved_and_receives_feedback_gradients(self):
        # Import only the official module, not its unrelated dataset/TensorFlow
        # package initialization or an incompatible global prismatic package.
        import importlib.util
        path = Path(__file__).resolve().parents[1]/'third_party/vla_adapter/prismatic/models/projectors.py'
        spec = importlib.util.spec_from_file_location('native_vla_adapter_projector', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ProprioProjector(llm_dim=32, proprio_dim=8)
        prefix = torch.randn(2, 8)
        expected = model(prefix).detach()
        proof = expand_adapter_proprio(model)
        actual = model(torch.cat((prefix, torch.randn(2, 67)), -1))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(proof['new_proprio_dim'], 75)
        actual.square().mean().backward()
        self.assertGreater(torch.count_nonzero(model.fc1.weight.grad[:, 8:]).item(), 0)

    def test_all_67_values_survive_including_last_neighbor(self):
        from VLA.scene64_conditioning import encode_numeric
        raw = np.zeros(64, np.float32)
        raw[18:24] = [1, 0, 0, 0, 1, 0]
        raw[24:27] = .03
        raw[27] = .08
        for index in range(6):
            raw[40+4*index:44+4*index] = [.1, -.1, .02, .03]
        stats = {'mean': np.zeros(67).tolist(), 'std': np.ones(67).tolist()}
        result = append_numeric67(np.arange(20), raw, stats)
        np.testing.assert_array_equal(result[:20], np.arange(20))
        np.testing.assert_array_equal(result[20:], encode_numeric(raw))
        self.assertEqual(len(result), 87)
        self.assertEqual(result[-1], raw[-1])
        with self.assertRaises(ValueError):
            append_numeric67(np.arange(20), raw[:-1], stats)


if __name__ == '__main__':
    unittest.main()
