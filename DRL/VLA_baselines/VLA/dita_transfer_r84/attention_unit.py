"""Check the official fallback repair against independent SDPA, including gradients."""
import sys
from .common import REPO,OUT
from VLA.common.artifact_io import atomic_json

def main():
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(REPO/'scripts'))
    from vision_tokenizers.qformer import CrossAttention
    torch.set_num_threads(2);torch.manual_seed(84)
    module=CrossAttention(48,num_heads=3,kv_dim=48,attn_drop=0.,proj_drop=0.).double().eval()
    x=torch.randn(2,32,48,dtype=torch.float64,requires_grad=True)
    context=torch.randn(2,17,48,dtype=torch.float64,requires_grad=True)
    native,_=module(x,context)
    q=module.q(x).reshape(2,32,3,16).transpose(1,2)
    k,v=module.kv(context).reshape(2,17,2,3,16).permute(2,0,3,1,4)
    ref=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=module.scale)
    ref=module.proj(ref.transpose(1,2).reshape(2,32,48))
    forward_error=float((native-ref).detach().abs().max())
    native_grads=torch.autograd.grad(native.square().sum(),(x,context,*module.parameters()),retain_graph=True)
    ref_grads=torch.autograd.grad(ref.square().sum(),(x,context,*module.parameters()))
    grad_error=max(float((a-b).abs().max()) for a,b in zip(native_grads,ref_grads,strict=True))
    if forward_error>1e-9 or grad_error>1e-8:raise ValueError('Fallback is not equivalent to real attention')
    atomic_json(OUT/'ATTENTION_FALLBACK_UNIT.json',{'passed':True,'dtype':'float64 CPU',
        'queries':32,'heads':3,'keys':17,'independent_reference':'torch scaled_dot_product_attention',
        'max_forward_error':forward_error,'max_gradient_error':grad_error,
        'native_weights_or_attention_architecture_changed':False})

if __name__=='__main__':main()
