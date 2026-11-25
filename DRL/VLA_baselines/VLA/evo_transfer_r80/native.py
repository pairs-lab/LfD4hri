"""Actual official Evo-1 with strict restore and explicit privileged PEFT.

Retain the released50x24 action network and native image/token processing.
Only extend the state encoder24->91, preserve all old weights exactly, and
add rank8 VLM LoRA. Full native action head is trained, not replaced.
"""
import json
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .common import SOURCE, BASE, VLM
from .data import state, targets, canonical


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features, dtype=torch.float32) * .02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        self.scale = 2.

    def forward(self, x):
        return self.base(x) + self.scale * nn.functional.linear(nn.functional.linear(x, self.A), self.B)


def expand_state(head):
    old = head.state_encoder.fc1.linear
    if old.in_features != 24:
        raise ValueError('Unexpected native state encoder')
    new = nn.Linear(91, old.out_features, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :24].copy_(old.weight.float())
        if old.bias is not None:
            new.bias.copy_(old.bias.float())
    if not torch.equal(new.weight[:, :24], old.weight.float()) or new.weight[:, 24:].count_nonzero():
        raise ValueError('Native state weights not preserved')
    head.state_encoder.fc1.linear = new
    return {'native_prefix': 24, 'appended_numeric': 67, 'new_state': 91,
            'native_columns_and_bias_preserved': True, 'new_columns_zero_initialized': True}


def load_foundation():
    from accelerate import init_empty_weights
    from transformers import AutoModel, AutoConfig
    import flash_attn
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(SOURCE / 'scripts'))
    # The simulator already has a different top-level config.py. Resolve the
    # official EvoConfig explicitly and restore the simulator module afterward.
    spec = importlib.util.spec_from_file_location('evo80_native_config', SOURCE / 'config.py')
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config_module
    spec.loader.exec_module(config_module)
    cfg = config_module.EvoConfig.from_dict(json.loads((BASE / 'config.json').read_text()))
    cfg.device = 'meta'
    cfg.vlm_name = str(VLM)
    cfg.enable_gradient_checkpointing = True
    cfg.num_inference_timesteps = 50
    if (cfg.horizon, cfg.per_action_dim, cfg.state_dim, cfg.num_layers) != (50, 24, 24, 8):
        raise ValueError('Do not silently resize the native released architecture')
    def architecture_only(*_args, **_kwargs):
        config = AutoConfig.from_pretrained(VLM, trust_remote_code=True, local_files_only=True)
        return AutoModel.from_config(config, trust_remote_code=True, use_flash_attn=True,
                                     torch_dtype=torch.bfloat16)
    with patch.dict(sys.modules, {'config': config_module}):
        from Evo1 import EVO1
        with patch.object(AutoModel, 'from_pretrained', architecture_only), init_empty_weights(include_buffers=False):
            model = EVO1(cfg)
    saved = torch.load(BASE / 'mp_rank_00_model_states.pt', weights_only=True,
                       mmap=True, map_location='cpu')['module']
    current = model.state_dict()
    missing, extra = sorted(set(current) - set(saved)), sorted(set(saved) - set(current))
    shapes = {k: [list(current[k].shape), list(saved[k].shape)] for k in set(current) & set(saved)
              if current[k].shape != saved[k].shape}
    if missing or extra or shapes:
        raise ValueError(json.dumps({'missing': missing, 'extra': extra, 'shapes': shapes}))
    model.load_state_dict(saved, strict=True, assign=True)
    restored_buffers = []
    for name, value in list(model.named_buffers()):
        if not value.is_meta:
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        original = getattr(parent, 'original_inv_freq', None)
        if (name != 'embedder.model.language_model.model.rotary_emb.inv_freq' or
                leaf != 'inv_freq' or original is None or original.is_meta or original.shape != value.shape):
            raise ValueError('Unknown unmaterialized native buffer: ' + name)
        # Qwen registers inv_freq as nonpersistent, so it is correctly absent
        # from checkpoint keys. Restore the original native constructor value,
        # not random data, zeros or a changed RoPE base/scaling formula.
        parent.inv_freq = original.detach().clone()
        restored_buffers.append(name)
    if any(value.is_meta for value in model.parameters()):
        raise ValueError('Unmaterialized native parameter')
    for key, value in model.state_dict().items():
        if value.dtype != saved[key].dtype or not torch.equal(value, saved[key]):
            raise ValueError('Retained foundation value differs: ' + key)
    model._device = model.embedder.device = 'cpu'
    cfg.device = 'cpu'
    return model, {'strict_loaded_tensors': len(saved), 'native_layers': 8,
        'native_horizon': 50, 'native_channels': 24,
        'all_pretrained_values_exact': True, 'flash_attention_version': flash_attn.__version__,
        'deterministic_native_nonpersistent_buffers_restored': restored_buffers,
        'unique_native_parameters': sum(p.numel() for p in model.parameters())}


class Runtime:
    def __init__(self, stats, checkpoint=None, *, device='cuda'):
        from safetensors.torch import load_file
        self.stats, self.device = stats, device
        torch.manual_seed(202680)
        self.model, self.base = load_foundation()
        self.model.requires_grad_(False)
        self.extension = expand_state(self.model.action_head)
        self.model.action_head.requires_grad_(True)
        self.model.action_head.float()
        targets_vlm = [(n, m) for n, m in self.model.embedder.model.named_modules() if isinstance(m, nn.Linear)]
        for name, module in targets_vlm:
            parent, _, leaf = name.rpartition('.')
            host = self.model.embedder.model.get_submodule(parent) if parent else self.model.embedder.model
            setattr(host, leaf, LoRALinear(module))
        self.trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if not self.trainable or any(p.dtype != torch.float32 for p in self.trainable.values()):
            raise ValueError('Actual learned masters must be FP32')
        if checkpoint is not None:
            values = load_file(str(Path(checkpoint) / 'learned.safetensors'), device='cpu')
            if set(values) != set(self.trainable):
                raise ValueError('Learned parameter keys differ')
            with torch.no_grad():
                for name, value in values.items():
                    p = self.trainable[name]
                    if value.dtype != torch.float32 or value.shape != p.shape:
                        raise ValueError('Learned precision/shape differs: ' + name)
                    p.copy_(value)
                    if not torch.equal(p, value):
                        raise ValueError('Learned value did not reload exactly')
        self.model.to(device)
        self.model._device = self.model.embedder.device = device
        self.model.config.device = device
        self.last_input = None
        self.capture_pixels = False
        self.last_pixels = None
        native_pixels = self.model.embedder._preprocess_images
        def checked_pixels(batch):
            pixels, tiles = native_pixels(batch)
            if pixels.shape != (3, 3, 448, 448) or not torch.isfinite(pixels).all():
                raise ValueError('Actual native pixel boundary differs')
            self.last_input['pixels'] = list(pixels.shape)
            if self.capture_pixels:
                self.last_pixels = pixels.detach().float().cpu()
            return pixels, tiles
        self.model.embedder._preprocess_images = checked_pixels
        original = self.model.embedder._prepare_and_fuse_embeddings
        def checked(prompts, vit, masks, tiles):
            lengths = [len(ids) for ids in self.model.embedder.tokenizer(
                prompts, truncation=False, padding=False)['input_ids']]
            if max(lengths) > self.model.embedder.max_text_length or tiles != [[1, 1, 1]]:
                raise ValueError('Native multimodal text/view truncated')
            self.last_input['prompt_tokens'] = lengths
            self.last_input['visual_tokens'] = int(vit.shape[1]) * int(vit.shape[0])
            self.last_input['native_tiles'] = tiles
            return original(prompts, vit, masks, tiles)
        self.model.embedder._prepare_and_fuse_embeddings = checked
        self.adaptation = {'VLM_LoRA_rank': 8, 'VLM_LoRA_linears': len(targets_vlm),
            'state_extension': self.extension,
            'trainable_parameters': sum(p.numel() for p in self.trainable.values()),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'native_action_head_preserved_and_finetuned': True,
            'paper_full_VLM_stage2_replaced_by_disclosed_PEFT': True}

    def embeddings(self, sample):
        images = [[torch.from_numpy(sample[k].copy()).permute(2, 0, 1).float().to(self.device) / 255
                   for k in ('front', 'wrist', 'map')]]
        s = torch.from_numpy(state(sample, self.stats))[None].to(self.device)
        self.last_input = {'views': ['front', 'wrist', 'causal_GT_map'], 'numeric': [1, 67],
            'state': list(s.shape), 'pixels': [3, 3, 448, 448],
            'future_labels_teacher_phase_or_success': False}
        emb = self.model.get_vl_embeddings(images, torch.ones((1, 3), dtype=torch.int32, device=self.device),
                                           [sample['instruction']], False)
        if emb.shape != (1, 1024, 896) or not torch.isfinite(emb).all():
            raise ValueError('Malformed actual native VLM embedding')
        return emb, s

    def learning_mode(self, *, joint):
        self.model.train()
        # Disable stochastic backbone dropout, not gradients through LoRA.
        self.model.embedder.model.eval()
        for name, p in self.trainable.items():
            p.requires_grad_(joint or not name.startswith('embedder.'))

    def loss(self, sample):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb, s = self.embeddings(sample)
            a, mask = targets(sample, self.stats)
            a = torch.from_numpy(a)[None].to(self.device)
            mask = torch.from_numpy(mask)[None].to(self.device)
            prediction, noise = self.model.predict_action(emb, s, actions_gt=a, action_mask=mask)
            velocity = a - noise
            error = (prediction.view_as(velocity).float() - velocity.float()).square()
            value = (error * mask).sum() / mask.sum()
        if not torch.isfinite(value):
            raise ValueError('Nonfinite native masked flow loss')
        return value

    def predict(self, sample, seed=202680):
        self.model.eval()
        with torch.random.fork_rng(devices=[torch.device(self.device).index or 0]), torch.inference_mode():
            torch.manual_seed(int(seed))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb, s = self.embeddings(sample)
                mask = torch.zeros((1, 50, 24), device=self.device)
                mask[:, :, :7] = 1
                output = self.model.predict_action(emb, s, action_mask=mask).float().reshape(50, 24)
            raw = output.cpu().numpy()
        return canonical(raw, self.stats)
