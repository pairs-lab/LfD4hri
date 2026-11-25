"""Official EEF rectified-flow objective with an explicit terminal time mask.

The author's heterogeneous action-dimension mask remains EEF7 here. Fixed
10-step chunk length is retained; no loss is applied to nonexistent tail steps.
"""
import torch


def loss(model, cond, actions, is_pad):
    if model.sampling_type != 'uniform' or not torch.all(cond['action_type'] == 1):
        raise ValueError('This tested adapter only implements native uniform EEF7 flow loss')
    dtype=next(model.parameters()).dtype
    actions=actions.to(dtype)
    b,h,d=actions.shape
    if d!=7 or is_pad.shape!=(b,h) or is_pad.dtype!=torch.bool or is_pad.all():
        raise ValueError('Finite EEF7 labels and a nonempty real temporal mask required')
    t=(torch.rand(1,device=actions.device)+torch.arange(b,device=actions.device)/b)%(1.-1e-5)
    t=t.to(dtype)
    z1=torch.randn((b,h,7),dtype=dtype,device=actions.device)
    zt=(1-t[:,None,None])*actions+t[:,None,None]*z1
    predicted=model.dit_forward(zt,t,cond)
    diff=(z1-actions)-predicted
    valid=(~is_pad)[...,None].expand_as(diff)
    return diff[valid].square().mean()
