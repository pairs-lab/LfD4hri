import unittest

import torch

from VLA.smolvla.precision import promote_trainable_masters


class SmolVLAPrecisionTest(unittest.TestCase):
    def test_small_adam_update_is_lost_with_bf16_but_retained_with_fp32(self):
        bf16 = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        bf16.weight.data.fill_(.02)
        fp32 = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        fp32.load_state_dict(bf16.state_dict())
        identity = id(fp32.weight)
        report = promote_trainable_masters(fp32)
        self.assertTrue(report["passed"])
        self.assertEqual(id(fp32.weight), identity)
        for model in (bf16, fp32):
            before = model.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            if model is bf16:
                self.assertTrue(torch.equal(before, model.weight))
            else:
                self.assertFalse(torch.equal(before, model.weight))
                self.assertEqual(optimizer.state[model.weight]["exp_avg"].dtype, torch.float32)

    def test_frozen_weights_stay_bf16_and_reloaded_trainables_keep_full_precision(self):
        model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        model.bias.requires_grad_(False)
        promote_trainable_masters(model)
        self.assertEqual(model.bias.dtype, torch.bfloat16)
        value = torch.tensor([[.020001]], dtype=torch.float32)
        model.load_state_dict({"weight": value, "bias": model.bias.detach()})
        self.assertTrue(torch.equal(model.weight, value))
        self.assertFalse(torch.equal(model.weight, value.bfloat16().float()))


if __name__ == "__main__":
    unittest.main()
