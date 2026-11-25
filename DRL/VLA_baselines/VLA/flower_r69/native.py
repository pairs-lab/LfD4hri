"""Official FLOWER flow core with explicit causal scene67 / third-view additions.

No planner, geometric action residual, teacher phase, or success flag is used.
Architecture is checkpoint-derived (12 blocks), not paper/default-derived (18).
"""
import functools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TVF
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'VLA/third_party/flower_vla_pret'
BASE = ROOT / 'VLA/checkpoints/candidates_r68/flower'
sys.path.insert(0, str(SOURCE))
from flower_vla.agents.flower import FlowerVLA

MEAN = (.48145466, .4578275, .40821073)
STD = (.26862954, .26130258, .27577711)


def mapped_state(state):
    result = {}
    for key, value in state.items():
        if not key.startswith('agent.'):
            raise ValueError('Unexpected foundation tensor prefix: ' + key)
        key = key.removeprefix('agent.')
        for old, new in (('vlm.language_encoder.', 'vlm.language_model.model.encoder.'),
                         ('.mlp.c_fc1.', '.mlp.fc1.'), ('.mlp.c_fc2.', '.mlp.fc2.'),
                         ('.mlp.c_proj.', '.mlp.proj.')):
            key = key.replace(old, new)
        if key in result:
            raise ValueError('Foundation key mapping collision: ' + key)
        result[key] = value
    return result


class Native(FlowerVLA):
    def _setup_vlm(self, vlm_path, freeze_vision_tower, freeze_florence, freeze_embeddings_only):
        # Called under torch.device(meta). Do not load another Florence weight
        # copy: the complete VLM already exists in the FLOWER foundation file.
        cfg = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True, local_files_only=True)
        cfg.text_config.vocab_size = 51290  # official checkpoint includes <Flow>
        cfg.vocab_size = 51290
        self.vlm = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                  attn_implementation='sdpa')
        del self.vlm.language_model.model.decoder, self.vlm.language_model.lm_head
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)
        self.train_vlm = True
        self.prompt_embeds = torch.empty(1, 1, cfg.text_config.d_model)

    def encode_observations(self, batch):
        # Preserve native pretraining order: Flow token, RGB views, text. Use
        # actual language padding mask, NOT all-ones or truncating scene text.
        images = [self.vlm._encode_image(batch['images'][:, i]) for i in range(3)]
        visual = torch.cat(images, dim=1)
        text = self.vlm.get_input_embeddings()(batch['input_ids'])
        flow = self.prompt_embeds.to(visual).expand(visual.shape[0], -1, -1)
        merged = torch.cat((flow, visual, text), dim=1)
        # The official pretraining path masks the frozen Flow token as a key.
        mask = torch.cat((torch.zeros((len(merged), 1), device=merged.device, dtype=torch.bool),
            torch.ones(visual.shape[:2], device=merged.device, dtype=torch.bool),
            batch['attention_mask'].bool()), dim=1)
        features = self.vlm.get_encoder()(inputs_embeds=merged, attention_mask=mask).last_hidden_state
        self.last_input_shapes = {'images': list(batch['images'].shape),
            'visual_tokens_per_view': [x.shape[1] for x in images],
            'language_tokens': list(batch['input_ids'].shape), 'context_tokens': merged.shape[1],
            'numeric': list(batch['numeric'].shape), 'frequency_Hz': 10, 'action_space': 'eef_delta7'}
        return {'features': self.vlm_token_dropout(features), 'attention_mask': mask,
            'frequency_embeds': self.frequency_embedder(torch.full((len(merged), 1), 10., device=self.device)),
            'action_type': torch.ones(len(merged), device=self.device, dtype=torch.long),
            'action_space_embeds': None, 'proprio': batch['numeric']}


class LoRALinear(nn.Module):
    """Standard W x + (alpha/r) B A x; frozen native W is never replaced."""
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device,
                                           dtype=torch.float32) * .02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                          dtype=torch.float32))
        self.scale = 2.

    def forward(self, x):
        return self.base(x) + self.scale * nn.functional.linear(nn.functional.linear(x, self.A), self.B)


def load_foundation():
    import yaml
    from accelerate import init_empty_weights
    spec = yaml.safe_load((BASE / 'config.yaml').read_text())['trainer']['agent']['agent']
    if spec['n_layers'] != 12 or spec['rope_theta'] != 1000.:
        raise ValueError('Foundation architecture configuration changed')
    # Keep small constructor constants on CPU (DaViT calls linspace.item()).
    # Only learned parameters are meta; no second full VLM allocation occurs.
    with init_empty_weights(include_buffers=False):
        model = Native(device=torch.device('meta'), process_id=0, accelerator=None,
            target_modality='actions', obs_modalities='observation', goal_modalities='task',
            img_modalities=['image_primary', 'image_wrist', 'scene_map'], lang_modalities=['language_instruction'],
            vlm_path=str(BASE / 'florence_assets_r69'), lowdim_obs_dim=67, action_dim=7,
            act_window_size=10, multistep=10, num_sampling_steps=4, use_second_view=True,
            use_proprio=True, action_type_adaln=True, use_causal_attention=True,
            use_cross_attn=True, use_adaln_cond=False, sampling_type='uniform',
            cfg_dropout=0., token_dropout=.1, dit_dim=1024, n_heads=16, n_layers=12,
            use_rope=True, query_seq_len=100, rope_theta=1000.,
            attn_pdrop=.1, resid_pdrop=.1, mlp_pdrop=.1)
    saved = mapped_state(torch.load(BASE / '360000_model_weights.pt', weights_only=True,
                                   mmap=True, map_location='cpu'))
    actual = model.state_dict()
    missing, extra = sorted(set(actual) - set(saved)), sorted(set(saved) - set(actual))
    mismatch = {k: [list(actual[k].shape), list(saved[k].shape)] for k in set(actual) & set(saved)
                if actual[k].shape != saved[k].shape}
    if missing or extra or mismatch:
        raise ValueError(json.dumps({'missing': missing, 'extra': extra, 'shapes': mismatch}))
    model.load_state_dict(saved, strict=True, assign=True)
    if any(not torch.equal(v, saved[k]) for k, v in model.state_dict().items()):
        raise ValueError('A retained pretrained tensor changed during load')
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Unmaterialized model parameter')
    model.device = torch.device('cpu')
    # The official processor adds 1024 Florence grounding tokens before Flow.
    # BART alone would incorrectly assign Flow50265 rather than native51289.
    tokenizer = AutoProcessor.from_pretrained(BASE / 'florence_assets_r69', local_files_only=True,
                                              trust_remote_code=True).tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': ['<Flow>']})
    if len(tokenizer) != 51290 or tokenizer.convert_tokens_to_ids('<Flow>') != 51289:
        raise ValueError('Native Flow-token ID no longer matches checkpoint')
    model.prompt_embeds = model.vlm.get_input_embeddings().weight[51289].detach()[None, None].clone()
    receipt = {'strict_exact_loaded_tensors': len(saved), 'missing': [], 'unexpected': [],
        'n_flow_blocks': len(model.dit), 'rope_theta': spec['rope_theta'],
        'foundation_core_not_randomly_extended_to_18_blocks': True,
        'unique_pretrained_parameters': sum(p.numel() for p in model.parameters()),
        'official_pretraining_code': str(SOURCE), 'official_checkpoint': str(BASE / '360000_model_weights.pt')}
    return model, tokenizer, receipt


def adapt(model, *, device='cuda', rank=8):
    # Explicitly replace only the EEF ZeroEncoder. No learned native EEF state
    # prefix exists to overwrite. Initially remains exactly the zero branch.
    dim = model.dit_dim
    projection = nn.Sequential(nn.Linear(67, dim), nn.SiLU(), nn.Linear(dim, dim))
    nn.init.zeros_(projection[-1].weight)
    nn.init.zeros_(projection[-1].bias)
    model.proprio_encoders['eef_delta'] = projection
    targets = [(name, module) for name, module in model.vlm.named_modules() if isinstance(module, nn.Linear)]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.vlm.parameters():
        parameter.requires_grad_(False)
    for name, module in targets:
        parent_name, _, leaf = name.rpartition('.')
        parent = model.vlm.get_submodule(parent_name) if parent_name else model.vlm
        setattr(parent, leaf, LoRALinear(module, rank))
    # Do not train unused action-domain slots or the unused action-space
    # embedder: they do not enter native EEF forward and would waste Adam RAM.
    for table in (model.action_encoders, model.action_decoders, model.proprio_encoders, model.adaln):
        for name, module in table.items():
            if name != 'eef_delta':
                module.requires_grad_(False)
    model.action_space_embedder.requires_grad_(False)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.device = torch.device(device)
    model.to(device)
    model.prompt_embeds = model.prompt_embeds.to(device)
    return {'backbone_lora_rank': rank, 'backbone_lora_layers': len(targets),
            'privileged_projection': '67->1024->1024_EEF_ZeroEncoder_replacement',
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'full_VLM_finetune_NOT_reproduced': True}


def image_tensor(rgb):
    if np.asarray(rgb).dtype != np.uint8 or np.asarray(rgb).shape[-1] != 3:
        raise ValueError('Actual uint8 RGB required, not BGR or already normalized pixels')
    t = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.
    t = TVF.resize(t, [224, 224], antialias=True)
    return TVF.normalize(t, MEAN, STD)


def native_batch(sample, tokenizer, stats, device='cuda'):
    # Full scene values travel losslessly as numeric67, not an overlength
    # textual list that would be cut off by the author's 50-token tokenizer.
    prompt = ('Agent Type: 1-arm UR5e Robotiq2F-85, Action Space: Delta End-Effector, '
              'Task Instruction: ' + sample['instruction'])
    ids = tokenizer(prompt, return_tensors='pt', truncation=False)
    if ids['input_ids'].shape[1] > 77:
        raise ValueError('Prompt exceeds declared budget; do not silently truncate instruction')
    mean, std = (np.asarray(stats['scene67'][k], np.float32) for k in ('mean', 'std'))
    state = np.asarray(sample['numeric67'], np.float32)
    if state.shape != (67,) or not np.isfinite(state).all():
        raise ValueError('All causal67 fields required')
    batch = {'images': torch.stack([image_tensor(sample[k]) for k in ('front', 'wrist', 'map')])[None].to(device),
        'input_ids': ids['input_ids'].to(device), 'attention_mask': ids['attention_mask'].to(device),
        'numeric': torch.from_numpy((state - mean) / (std + 1e-8))[None].to(device)}
    return batch


def action_targets(canonical, stats):
    """Meters/world translation, local Euler radians, native OPEN01 gripper."""
    value = np.asarray(canonical, np.float32).copy()
    if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
        raise ValueError('Finite canonical Hx7 labels required')
    value[:, 6] = (1. - value[:, 6]) / 2.
    lo, hi = (np.asarray(stats['adapter']['action'][k], np.float32) for k in ('q01', 'q99'))
    value[:, :6] = np.clip(2*(value[:, :6]-lo[:6])/(hi[:6]-lo[:6]+1e-8)-1., -1., 1.)
    return value


def canonical_actions(native, stats):
    value = np.asarray(native, np.float32).copy()
    if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
        raise ValueError('Finite neural Hx7 prediction required')
    lo, hi = (np.asarray(stats['adapter']['action'][k], np.float32) for k in ('q01', 'q99'))
    value[:, :6] = .5*(value[:, :6]+1.)*(hi[:6]-lo[:6]+1e-8)+lo[:6]
    value[:, 6] = 1. - 2.*np.clip(value[:, 6], 0., 1.)
    return value


if __name__ == '__main__':
    m, t, evidence = load_foundation()
    print(json.dumps(evidence, indent=2))
