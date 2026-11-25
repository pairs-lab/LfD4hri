"""Full expert first; explicit pretrained expert-LoRA fallback only on OOM."""
from torch import nn
import torch
from VLA.intern_r72.native import adapt as full_adapt, LoRA


def adapt(model, mode):
    if mode == 'expert_full': return {'mode':mode,**full_adapt(model)}
    if mode != 'expert_lora': raise ValueError('Unknown memory-qualified adaptation')
    model.requires_grad_(False)
    for name in ('action_in_proj','action_out_proj','action_time_mlp_in','action_time_mlp_out','state_proj'):
        getattr(model,name).requires_grad_(True)
    counts={}
    for name, branch, rank in (
            ('VLM',model.qwen3_5_with_expert.qwen3_5,8),
            ('action_expert',model.qwen3_5_with_expert.action_expert,16)):
        selected=[(n,m) for n,m in branch.named_modules() if isinstance(m,nn.Linear)]
        for key,module in selected:
            parent,_,leaf=key.rpartition('.')
            setattr(branch.get_submodule(parent) if parent else branch,leaf,LoRA(module,rank))
        counts[name+'_LoRA_layers']=len(selected)
    model._setup_wan_grad();model.to('cuda')
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    model.gradient_checkpointing_enable()
    return {'mode':mode,**counts,'VLM_rank':8,'expert_rank':16,
        'native_action_expert_full_finetune':False,'pretrained_expert_NOT_random_head_replacement':True,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'reason':'Only selected after full expert actual optimizer gate OOM; must requalify exact variant'}
