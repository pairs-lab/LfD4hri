"""Exact trained-parameter checkpoints; authentic frozen bases stay referenced."""
from pathlib import Path
from .common import read,atomic_json


def save_adaptation(model,destination):
    from safetensors.torch import save_file
    destination=Path(destination)
    state={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
    if not state:raise ValueError('No actual learned parameters to checkpoint')
    save_file(state,str(destination/'weights.safetensors'))
    atomic_json(destination/'ADAPTATION_SCHEMA.json',{'adapter_parameters_only':True,
        'keys':sorted(state),'elements':sum(p.numel() for p in state.values()),
        'frozen_authentic_foundation_and_fixed_native_buffers_reloaded_separately':True,
        'not_a_replacement_action_model':True})


def load_adaptation(model,checkpoint):
    from safetensors.torch import load_file,load_model
    import torch
    checkpoint=Path(checkpoint);schema=checkpoint/'ADAPTATION_SCHEMA.json'
    if not schema.exists():
        # Only the documented earlier R100 full-model gate checkpoint format.
        load_model(model,str(checkpoint/'weights.safetensors'),strict=True);return
    proof=read(schema)
    if not proof['adapter_parameters_only']:raise ValueError('Unknown learned checkpoint schema')
    state=load_file(str(checkpoint/'weights.safetensors'),device='cpu')
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(state)!=set(params) or sorted(state)!=proof['keys']:
        raise ValueError('Missing/unexpected learned native/privileged/LoRA parameter')
    with torch.no_grad():
        for name,value in state.items():
            if value.shape!=params[name].shape:raise ValueError('Learned native parameter shape mismatch: '+name)
            params[name].copy_(value.to(device=params[name].device,dtype=params[name].dtype))
