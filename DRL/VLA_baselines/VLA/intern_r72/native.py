"""Native pretrained MoT/flow expert; declared continuous-state67 + LoRA."""
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from accelerate import init_empty_weights
from VLA.intern_r72.common import SOURCE, BASE, read
sys.path.insert(0, str(SOURCE / 'src'))
from transformers import AutoConfig
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Tokenizer
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.action_tokens import ensure_qwen35_action_tokens
from VLA.intern_r72.data import targets


def foundation():
    tokenizer = Qwen3_5Tokenizer.from_pretrained(BASE / 'qwen_assets', local_files_only=True)
    ensure_qwen35_action_tokens(tokenizer)
    with safe_open(BASE / 'model.safetensors', framework='pt', device='cpu') as f:
        saved = {k.removeprefix('model.'): f.get_tensor(k) for k in f.keys()}
    ignored = {k: saved.pop(k) for k in list(saved) if k in
        ('learnable_to_wan_proj.weight','learnable_to_wan_proj.bias','_wan_grid_sizes')}
    if set(ignored) != {'learnable_to_wan_proj.weight','learnable_to_wan_proj.bias','_wan_grid_sizes'}:
        raise ValueError('Unexpected WAN checkpoint variant')
    qcfg = AutoConfig.from_pretrained(BASE / 'qwen_assets', local_files_only=True)
    qcfg.text_config.vocab_size = saved['qwen3_5_with_expert.qwen3_5.lm_head.weight'].shape[0]
    qcfg.vocab_size = qcfg.text_config.vocab_size
    cfg = InternVLAA15Config(vlm_model_name_or_path=str(BASE / 'qwen_assets'), action_loss_only=True,
        tokenize_state=False, max_state_dim=67, max_action_dim=32, device='cpu', use_sdpa=True,
        gradient_checkpointing=True, freeze_learnable_tokens=True)
    with init_empty_weights(include_buffers=False), \
            patch.object(Qwen3_5ForConditionalGeneration,'from_pretrained',
                side_effect=lambda *a,**kw: Qwen3_5ForConditionalGeneration(qcfg)), \
            patch.object(Qwen3_5Tokenizer,'from_pretrained',return_value=tokenizer):
        model = InternVLAA15(cfg)
    current = model.state_dict()
    # Official safetensors omit tied input embeddings, retaining lm_head.
    alias = 'qwen3_5_with_expert.qwen3_5.model.language_model.embed_tokens.weight'
    if alias in current and alias not in saved:
        saved[alias] = saved['qwen3_5_with_expert.qwen3_5.lm_head.weight']
    fresh = {'state_proj.weight','state_proj.bias'}
    missing = set(current)-set(saved)-fresh; extra = set(saved)-set(current)
    shapes = {k:[list(current[k].shape),list(saved[k].shape)] for k in set(saved)&set(current)
              if current[k].shape != saved[k].shape}
    if missing or extra or shapes: raise ValueError(str({'missing':sorted(missing),'extra':sorted(extra),'shapes':shapes}))
    result = model.load_state_dict(saved, strict=False, assign=True)
    if set(result.missing_keys) != fresh or result.unexpected_keys: raise ValueError('Unexpected non-native tensors')
    model.state_proj = nn.Linear(67,1024)
    nn.init.zeros_(model.state_proj.weight); nn.init.zeros_(model.state_proj.bias)
    model.qwen3_5_with_expert.qwen3_5.tie_weights()
    if any(p.is_meta for p in model.parameters()): raise ValueError('Unmaterialized meta parameter')
    return model, {'active_pretrained_tensors_loaded':len(saved),
        'discarded_training_only_WAN_tensors':sorted(ignored), 'new_zero_initialized_tensors':sorted(fresh),
        'tied_input_embedding_from_exact_native_lm_head':True, 'action_loss_only_no_WAN_aux':True,
        'state67_native_continuous_prefix_addition_declared':True, 'native_flow_horizon':50,
        'native_action_layout':'world_deltaXYZ_m,local_Euler_rad,signed_absolute_close; first7_of32',
        'no_extra_Qwen_foundation_weights_loaded':True}


class LoRA(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__(); self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank,base.in_features,device=base.weight.device,dtype=torch.float32)*.02)
        self.B = nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self, x):
        return self.base(x)+2*nn.functional.linear(nn.functional.linear(x,self.A),self.B)


def adapt(model, device='cuda'):
    model.requires_grad_(True)
    vlm = model.qwen3_5_with_expert.qwen3_5
    vlm.requires_grad_(False)
    candidates = [(n,m) for n,m in vlm.named_modules() if isinstance(m,nn.Linear)]
    for name, module in candidates:
        parent, _, leaf = name.rpartition('.')
        setattr(vlm.get_submodule(parent) if parent else vlm,leaf,LoRA(module))
    model._setup_wan_grad(); model.to(device)
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    model.gradient_checkpointing_enable()
    return {'VLM_LoRA_rank':8,'VLM_LoRA_layers':len(candidates),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'native_action_expert_full_finetune':True,'continuous67_prefix':True,'full_paper_recipe_NOT_reproduced':True}


def loss(model, batch, sample, stats, *, noise=None, time=None):
    a = targets(sample['action'],sample['action_is_pad'],stats['intern_action'])
    device = batch['state'].device
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
        errors, ce, _, _, _ = model.forward(**batch, actions=torch.from_numpy(a)[None].to(device),noise=noise,time=time)
    valid = torch.from_numpy(~sample['action_is_pad'])[None].to(device)
    action_loss = (errors[:,:,:7]*valid[:,:,None]).sum()/(valid.sum()*7)
    total = 10*action_loss + ce.mean()
    return total, {'flow_valid7_MSE':float(action_loss.detach()),'FAST_CE':float(ce.mean().detach()),
        'valid_target_steps':int(valid.sum()),'padded_target_steps':int((~valid).sum())}


def prediction(model, batch, seed=202672):
    model.eval(); args = {k:v for k,v in batch.items() if k != 'labels'}
    with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        torch.manual_seed(seed)
        out = model.sample_actions(**args)
    if out.shape != (1,50,32) or not torch.isfinite(out).all(): raise ValueError('Invalid native flow output')
    return out[0].float().cpu().numpy()
