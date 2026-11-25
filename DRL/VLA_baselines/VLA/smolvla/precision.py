"""Keep trainable SmolVLA masters FP32, independently of AMP compute dtype."""

from functools import lru_cache


def promote_trainable_masters(policy) -> dict:
    import torch

    before = {}
    promoted = 0
    names = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        dtype = str(parameter.dtype)
        before[dtype] = before.get(dtype, 0) + parameter.numel()
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
            if parameter.grad is not None:
                parameter.grad = parameter.grad.to(torch.float32)
            promoted += parameter.numel()
        names.append(name)
    if not names or any(p.dtype != torch.float32 for p in policy.parameters() if p.requires_grad):
        raise RuntimeError("Trainable master parameters must all be FP32")
    return {"passed": True, "trainable_dtype_counts_before": before,
            "promoted_parameter_values": promoted, "trainable_parameter_names": names,
            "master_dtype": "torch.float32", "amp_compute_dtype": "torch.bfloat16"}


@lru_cache(maxsize=2)
def fp32_policy_class(train_visual_connector=False):
    from lerobot.policies.smolvla import SmolVLAPolicy

    class FP32MasterSmolVLAPolicy(SmolVLAPolicy):
        # Promoting before from_pretrained loads the checkpoint also prevents
        # newly learned FP32 weights being silently quantized on reload.
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            if train_visual_connector:
                for parameter in self.model.vlm_with_expert.get_vlm_model().connector.parameters():
                    parameter.requires_grad_(True)
            self._fp32_master_audit = promote_trainable_masters(self)
            self._fp32_master_audit["train_visual_connector"] = bool(train_visual_connector)

    return FP32MasterSmolVLAPolicy
