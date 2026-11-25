"""Lossless input-width extensions, NOT a controller or action correction.

The native proprio prefix keeps its original meaning. All causal67 simulator
features are appended, never substituted for an incompatible native prefix.
New columns start at zero while every pretrained column remains exact.
"""
import torch
from torch import nn

from VLA.scene64_conditioning import encode_numeric


def inject_native_lora(model, rank):
    """Official all-linear LoRA on an nn.Module with a dataclass config.

    The host supplies PEFT's dictionary metadata without replacing the native
    policy config or forward/generate methods. Real linear layers are adapted
    in place; callers continue to execute the original VLA forward path.
    """
    from peft import LoraConfig, get_peft_model
    class Host(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            self.config = {'model_type': 'native_vla_module'}

        def forward(self, *args, **kwargs):
            return self.policy(*args, **kwargs)
    host = get_peft_model(Host(model), LoraConfig(r=rank, lora_alpha=2*rank,
        lora_dropout=0., target_modules='all-linear', init_lora_weights='gaussian'))
    adapted = host.base_model.model.policy
    if adapted is not model:
        raise ValueError('LoRA must preserve the actual original policy instance')
    for name, parameter in adapted.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return adapted


def append_numeric67(native_proprio, raw64, stats):
    import numpy as np
    prefix = np.asarray(native_proprio, np.float32)
    mean, std = (np.asarray(stats[k], np.float32) for k in ('mean', 'std'))
    if (prefix.ndim != 1 or not np.isfinite(prefix).all()
            or mean.shape != (67,) or std.shape != (67,)
            or not np.isfinite(np.r_[mean, std]).all() or np.any(std < 0)):
        raise ValueError('Finite native prefix and fitted TRAIN67 normalization required')
    tail = (encode_numeric(raw64)-mean)/(std+np.float32(1e-8))
    result = np.r_[prefix, tail].astype(np.float32)
    if result.shape != (len(prefix)+67,) or not np.isfinite(result).all():
        raise ValueError('Do not truncate privileged feedback')
    return result


def expand_xvla_proprio(transformer, original_proprio_dim, extra_dim=67):
    """Preserve native [action, proprio, time] weights around inserted columns."""
    old = transformer.action_encoder
    prefix = transformer.dim_action+original_proprio_dim
    if extra_dim != 67 or old.input_size != prefix+transformer.dim_time:
        raise ValueError('Unexpected native X-VLA action/proprio/time layout')
    new = type(old)(old.input_size+extra_dim, old.output_size,
                    num_domains=old.fc.num_embeddings).to(old.fc.weight)
    original = old.fc.weight.detach().reshape(old.fc.num_embeddings, old.input_size, old.output_size)
    with torch.no_grad():
        expanded = new.fc.weight.detach().reshape(new.fc.num_embeddings, new.input_size, new.output_size)
        expanded.zero_()
        expanded[:, :prefix].copy_(original[:, :prefix])
        expanded[:, prefix+extra_dim:].copy_(original[:, prefix:])
        new.bias.weight.copy_(old.bias.weight)
    if (not torch.equal(expanded[:, :prefix], original[:, :prefix])
            or not torch.equal(expanded[:, prefix+extra_dim:], original[:, prefix:])
            or torch.count_nonzero(expanded[:, prefix:prefix+extra_dim])):
        raise ValueError('Pretrained X-VLA values changed during input extension')
    transformer.action_encoder = new
    return {'original_proprio_dim': original_proprio_dim, 'appended_numeric_dim': extra_dim,
        'new_proprio_dim': original_proprio_dim+extra_dim,
        'all_pretrained_columns_and_bias_preserved_exactly': True,
        'new_columns_zero_initialized': True, 'schema': 'native_proprio_plus_causal67_v1'}


def expand_adapter_proprio(projector, extra_dim=67):
    """Extend the official ProprioProjector without replacing its learned prefix."""
    old = projector.fc1
    if not isinstance(old, nn.Linear) or extra_dim != 67:
        raise ValueError('Expected official linear proprio input projector')
    new = nn.Linear(old.in_features+extra_dim, old.out_features,
                    bias=old.bias is not None).to(old.weight)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_features].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    if not torch.equal(new.weight[:, :old.in_features], old.weight):
        raise ValueError('Pretrained VLA-Adapter proprio weights changed')
    projector.fc1 = new
    projector.proprio_dim = new.in_features
    return {'original_proprio_dim': old.in_features, 'appended_numeric_dim': extra_dim,
        'new_proprio_dim': new.in_features, 'all_pretrained_columns_and_bias_preserved_exactly': True,
        'new_columns_zero_initialized': True, 'schema': 'native_proprio_plus_causal67_v1'}
