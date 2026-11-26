"""Native pretrained VLA runtimes shared by training and robot inference.

No geometric planner, privileged residual action, or oracle fallback. Each
runtime appends21 robot observations to native proprio and runs the real VLA.
"""
from functools import lru_cache
import json
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

from VLA.new_vlas.privileged_inputs import expand_xvla_proprio, expand_adapter_proprio, inject_native_lora

ROOT = Path(__file__).resolve().parents[2]


class Runtime:
    def parameters(self):
        return {name+'.'+key: value for name, module in self.modules.items()
                for key, value in module.named_parameters() if value.requires_grad}

    def to(self, device):
        self.device = str(device)
        for module in self.modules.values():
            module.to(device)
        return self

    def train(self):
        for module in self.modules.values():
            module.train()

    def eval(self):
        for module in self.modules.values():
            module.eval()

    def save_trainables(self, path):
        values = {name: value.detach().cpu().contiguous().clone() for name, value in self.parameters().items()}
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in values.values()):
            raise ValueError('Only finite FP32 trainable masters may be saved')
        torch.save(values, path)

    def load_trainables(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        actual = self.parameters()
        if set(saved) != set(actual):
            raise ValueError('Exact full trainable parameter key set required')
        with torch.no_grad():
            for name, value in saved.items():
                if value.shape != actual[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Saved precision/shape/finite mismatch: '+name)
                actual[name].copy_(value)
                if not torch.equal(actual[name].detach().cpu(), value):
                    raise ValueError('Non-exact saved trainable master reload: '+name)
        return len(saved)


class XVLARuntime(Runtime):
    family = 'xvla'
    rank = 8
    horizon = 30

    def __init__(self, stats, *, checkpoint=None, resource_root=ROOT):
        from lerobot.configs import PolicyFeature, FeatureType
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        self.stats = stats
        checkpoint = Path(checkpoint) if checkpoint else Path(resource_root)/'VLA/checkpoints/xvla-base'
        cfg = XVLAConfig.from_pretrained(str(checkpoint))
        cfg.device, cfg.dtype = 'cpu', torch.float32
        model = XVLAPolicy.from_pretrained(checkpoint, config=cfg,strict=True,local_files_only=True)
        self.extension = expand_xvla_proprio(model.model.transformer, cfg.max_state_dim)
        cfg.max_state_dim = 41
        cfg.input_features['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=(41,))
        model.model.dim_proprio = 41
        for value in model.parameters():
            value.requires_grad_(False)
        model = inject_native_lora(model, self.rank)
        for name in ('soft_prompt_hub', 'action_encoder', 'action_decoder'):
            for value in getattr(model.model.transformer, name).parameters():
                value.requires_grad_(True)
        self.model, self.config = model, cfg
        self.modules = {'policy': model}
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large',
            cache_dir=str(Path(resource_root)/'VLA/checkpoints/huggingface/hub'), local_files_only=True)

    @lru_cache(maxsize=512)
    def tokens(self, instruction):
        if len(self.tokenizer.encode(instruction)) > 50:
            raise ValueError('Do not silently truncate the task instruction')
        return self.tokenizer(instruction, return_tensors='pt', padding='max_length',
                              max_length=50, truncation=False)['input_ids']

    def batch(self, sample):
        if np.asarray(sample['state']).shape != (41,):
            raise ValueError('X-VLA requires native20 + robot21, not a historical state vector')
        result = {'observation.state': torch.from_numpy(sample['state'])[None].to('cuda'),
            'observation.language.tokens': self.tokens(sample['instruction']).to('cuda'),
            'domain_id': torch.tensor([12], device='cuda', dtype=torch.long)}
        mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
        std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
        for key, role in (('image', 'front'), ('image2', 'wrist')):
            from torchvision.transforms.functional import resize
            values = torch.as_tensor(sample[role], device='cuda').permute(2, 0, 1).float()/255
            values = resize(values, [224,224], antialias=True)
            result['observation.images.'+key] = ((values-mean)/std)[None]
        if 'action' in sample:
            result['action'] = torch.from_numpy(sample['action'])[None].to('cuda')
        return result

    def loss(self, sample):
        self.train()
        # Disable native backbone dropout, not gradients. Actual LoRA vision
        # and language modules still receive backprop through this eval mode.
        self.model.model.vlm.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = self.model(self.batch(sample))
        return loss, metrics

    def predict(self, sample):
        self.eval()
        self.config.device = 'cuda'
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(202643)
            return self.model.predict_action_chunk(self.batch(sample))[0].float().cpu().numpy()


class AdapterRuntime(Runtime):
    family = 'vla_adapter'
    rank = 64
    horizon = 8

    def __init__(self, stats, *, checkpoint=None, resource_root=ROOT):
        from VLA.new_vlas.adapter_loading import import_native, install_amp_query_cast
        from peft import get_peft_model, LoraConfig
        Config, Model, ImageProcessor, Processor, Head, Projector = import_native(Path(resource_root)/'VLA/third_party/vla_adapter')
        checkpoint = Path(checkpoint) if checkpoint else Path(resource_root)/'VLA/checkpoints/vla-adapter-spatial-pro'
        cfg = Config.from_pretrained(checkpoint)
        base, info = Model.from_pretrained(checkpoint, config=cfg, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation='eager', output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
            raise ValueError('Native checkpoint model/source mismatch: '+str(info))
        base.norm_stats = {'pybullet_train': stats['adapter']}
        base.vision_backbone.set_num_images_in_input(2)
        self.head = Head(input_dim=base.llm_dim, hidden_dim=base.llm_dim,
                        action_dim=7, num_task_tokens=512, use_pro_version=True)
        self.projector = Projector(llm_dim=base.llm_dim, proprio_dim=8)
        for module, filename in ((self.head, 'action_head--checkpoint.pt'),
                                 (self.projector, 'proprio_projector--checkpoint.pt')):
            saved = torch.load(checkpoint/filename, map_location='cpu', weights_only=True, mmap=True)
            values = {key.removeprefix('module.'): value for key, value in saved.items()}
            module.load_state_dict(values, strict=True)
            if any(not torch.equal(module.state_dict()[key], value.float()) for key, value in values.items()):
                raise ValueError('Checkpoint value promotion was not lossless')
        self.extension = expand_adapter_proprio(self.projector)
        self.amp_boundary = install_amp_query_cast(base)
        self.model = get_peft_model(base, LoraConfig(r=self.rank, lora_alpha=2*self.rank,
            lora_dropout=0., target_modules='all-linear', init_lora_weights='gaussian'))
        for name, value in self.model.named_parameters():
            if 'action_queries' in name:
                value.requires_grad_(True)
            if value.requires_grad:
                value.data = value.data.float()
        self.modules = {'policy': self.model, 'head': self.head, 'projector': self.projector}
        self.processor = Processor(ImageProcessor.from_pretrained(checkpoint),
                                   AutoTokenizer.from_pretrained(checkpoint, local_files_only=True))
        self.stats = stats
        self.training_capture = False
        self.captured = None
        original = self.head.predict_action

        def capture(this, features, **kwargs):
            # Same official head and Training perturbation. Retain the tensor
            # before native predict_action converts a detached copy to NumPy.
            result = original(features, **kwargs,
                              phase='Training' if self.training_capture else 'Inference')
            if self.training_capture:
                self.captured = result
            return result
        self.head.predict_action = MethodType(capture, self.head)

    def batch(self, sample):
        if np.asarray(sample['state']).shape != (29,):
            raise ValueError('VLA-Adapter requires native8 + robot21')
        instruction = sample['instruction'].lower()
        prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\nWhat action should the robot take to '+instruction+
            '?<|im_end|>\n<|im_start|>assistant\n')
        result = self.processor(prompt, Image.fromarray(sample['front']), return_tensors='pt')
        wrist = self.processor(prompt, Image.fromarray(sample['wrist']), return_tensors='pt')
        result['pixel_values'] = torch.cat((result['pixel_values'], wrist['pixel_values']), dim=1)
        if result['input_ids'].shape[1] > 512:
            raise ValueError('Task instruction unexpectedly long; no silent truncation allowed')
        return {key: value.to('cuda') for key, value in result.items()}

    def loss(self, sample):
        self.train()
        self.training_capture, self.captured = True, None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            self.model.predict_action(**self.batch(sample), unnorm_key='pybullet_train',
                proprio=sample['state'], proprio_projector=self.projector, action_head=self.head)
            prediction = self.captured
            target = torch.from_numpy(sample['action'])[None].to('cuda')
            if prediction is None or not prediction.requires_grad or prediction.shape != target.shape:
                raise ValueError('Native training feature graph or horizon detached/mismatched')
            loss = torch.nn.functional.l1_loss(prediction, target)
        self.training_capture, self.captured = False, None
        return loss, {'native_L1_loss': float(loss.detach())}

    def predict(self, sample):
        self.eval()
        self.training_capture = False
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return self.model.predict_action(**self.batch(sample), unnorm_key='pybullet_train',
                proprio=sample['state'], proprio_projector=self.projector, action_head=self.head)[0]


def create_runtime(family, stats):
    if family == 'xvla':
        return XVLARuntime(stats)
    if family == 'vla_adapter':
        return AdapterRuntime(stats)
    raise ValueError('Unknown native model family')
