"""Inject learned observed-state features into native DiT, preserving CFG."""
import torch


def conditioned_call(original,args,kwargs,delta):
    if 'z' in kwargs:z=kwargs['z']
    elif len(args)>=3:z=args[2]
    else:raise ValueError('Actual native cognition argument missing')
    if z.shape[0]==2*delta.shape[0]:delta=torch.cat((delta,torch.zeros_like(delta)),dim=0)
    elif z.shape[0]!=delta.shape[0]:raise ValueError('Unexpected native CFG batch')
    if 'z' in kwargs:return original(*args,**{**kwargs,'z':z+delta})
    return original(*args[:2],z+delta,*args[3:],**kwargs)
