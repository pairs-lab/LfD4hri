"""Opt-in FP32 signal islands; official RDT architecture/objective unchanged.

The original state adaptor returns BF16 under autocast; the official sampler
then casts its continuous action latent to that dtype at EVERY denoise step.
Keeping the state adaptor and final action readout in FP32 makes the official
sampler keep its latent in FP32 without editing the upstream implementation.
Image/language encoders and transformer attention retain BF16 compute.
"""
from types import MethodType
import torch


PROFILE = "fp32_state_action_readout_latent_v1"


def install_fp32_signal_islands(runner):
    if getattr(runner, "_vla_signal_precision", None) == PROFILE:
        return
    if getattr(runner, "_vla_signal_precision", None) is not None:
        raise ValueError("Conflicting signal precision profile")
    for module in (runner.state_adaptor, runner.model.final_layer):
        if any(parameter.dtype != torch.float32 for parameter in module.parameters()):
            raise ValueError("FP32 signal islands require losslessly loaded FP32 masters")
        original = module.forward
        def forward(self, values, _original=original):
            # These two official modules take a single tensor. Preserve all
            # parameter names/state_dict keys and backpropagation.
            with torch.autocast(device_type=values.device.type, enabled=False):
                return _original(values.float())
        module.forward = MethodType(forward, module)
    runner._vla_signal_precision = PROFILE


def signal_dtype(profile):
    if profile == "bf16":
        return torch.bfloat16
    if profile == PROFILE:
        return torch.float32
    raise ValueError("Unrecognized RDT signal precision: " + str(profile))
