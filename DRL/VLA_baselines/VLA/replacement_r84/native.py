"""Official VLANeXt-S flow/DCT/soft head. No learned-action shortcut or planner."""
import inspect
import sys
import numpy as np
from PIL import Image
from .common import BASE, REPO
from .data import denormalize_actions

class Runtime:
    horizon = 8
    def __init__(self, stats, *, checkpoint=None):
        import torch
        import yaml
        sys.path.insert(0, str(REPO))
        from src.models.VLANeXt import VLANeXt
        config = yaml.safe_load((REPO/'config/VLANeXt-S_config.yaml').read_text())
        settings = {k:v for k,v in config['model'].items() if k in inspect.signature(VLANeXt).parameters}
        settings.update(lmm_path=str(BASE), num_actions=8, num_history=8,
                        attn_implementation='sdpa')
        # SDPA is mathematically equivalent attention, avoiding the documented
        # Qwen3.5/FlashAttention 3D-position illegal-access bug. Not a smaller head.
        self.model = VLANeXt(**settings)
        old = self.model.action_projector
        if not isinstance(old, torch.nn.Linear) or old.in_features != 7:
            raise ValueError('Official linear native proprio7 prefix required')
        extended = torch.nn.Linear(74, old.out_features, dtype=old.weight.dtype)
        with torch.no_grad():
            extended.weight.zero_()
            extended.weight[:,:7].copy_(old.weight)
            extended.bias.copy_(old.bias)
        self.model.action_projector = extended
        if checkpoint:
            from safetensors.torch import load_model
            load_model(self.model, str(checkpoint), strict=True)
        self.model.to('cuda')
        self.model.lmm.config.use_cache = False
        self.stats = stats
        self.trainables = {n:p for n,p in self.model.named_parameters() if p.requires_grad}
        self.last_input = {}

    def batch(self, s, *, augment=False):
        import torch
        from torchvision.transforms import functional as F
        from torchvision.transforms import RandomResizedCrop
        images = []
        for name in ('front','wrist','map'):
            image = Image.fromarray(s[name]).resize((256,256), Image.Resampling.BILINEAR)
            if augment and name != 'map':
                i,j,h,w = RandomResizedCrop.get_params(image, (.8,1.), (.9,1.1))
                image = F.resized_crop(image,i,j,h,w,(256,256))
                image = F.adjust_brightness(image,float(torch.empty(()).uniform_(.8,1.2)))
                image = F.adjust_contrast(image,float(torch.empty(()).uniform_(.8,1.2)))
                image = F.adjust_saturation(image,float(torch.empty(()).uniform_(.8,1.2)))
                image = F.adjust_hue(image,float(torch.empty(()).uniform_(-.05,.05)))
            images.append(image)
        content = [{'type':'image','image':im} for im in images]
        content += [{'type':'text','text':s['instruction']}]
        text = self.model.processor.apply_chat_template([{'role':'user','content':content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.model.processor(text=[text], images=images, padding=True, return_tensors='pt')
        if len(inputs['image_grid_thw']) != 3 or 'mm_token_type_ids' not in inputs:
            raise ValueError('Missing native3-view or Qwen3.5 multimodal type IDs')
        valid = {'input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids','token_type_ids'}
        batch = {k:v.cuda() for k,v in inputs.items() if k in valid}
        batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
        batch['proprioception'] = torch.as_tensor(s['history74'][None], device='cuda')
        self.last_input = {'actual_views':3, 'image_grid_thw':inputs['image_grid_thw'].tolist(),
            'text_token_count':int(inputs['input_ids'].shape[1]), 'proprio_shape':[1,8,74],
            'history_includes_current_and_causal_past_only':True,
            'observation_numeric_action_planner':False}
        return batch

    def loss(self, s):
        import torch
        self.model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.model(**self.batch(s, augment=True), actions=torch.as_tensor(
                s['normalized_action'][None],device='cuda'), return_loss_components=True)

    def predict(self, s, seed):
        import torch
        self.model.eval()
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(int(seed))
            result = self.model.predict_action(**self.batch(s)).float().cpu().numpy()[0]
        if result.shape != (8,7) or not np.isfinite(result).all():
            raise ValueError('Native flow prediction malformed')
        return denormalize_actions(result,self.stats)
