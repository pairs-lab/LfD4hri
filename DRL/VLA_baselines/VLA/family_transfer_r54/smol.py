"""Official SmolVLA flow loss / inference with unchanged camera1/2 + GT3.

Only observation profile, TRAIN normalization and sampling change. Every
action is produced by the native policy, not a simulator-state controller.
"""
import numpy as np
from .common import SMOL, read


def config(parent=SMOL, device='cuda'):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    cfg = PreTrainedConfig.from_pretrained(str(parent))
    if cfg.type != 'smolvla' or cfg.max_state_dim != 67 or cfg.chunk_size != 5:
        raise ValueError('Own existing native67 / horizon5 SmolVLA required')
    cfg.device = device
    cfg.input_features['observation.images.camera3'] = PolicyFeature(type=FeatureType.VISUAL, shape=(3,384,384))
    cfg.freeze_vision_encoder = True
    cfg.n_action_steps = 5
    cfg.compile_model = False
    return cfg


def statistics(data):
    from .data import action_stats
    return {'observation.state': data.stats['scene67'], 'action': action_stats(data)}


def processors(cfg, stats):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    values = {name: {k:torch.tensor(v,dtype=torch.float32) for k,v in s.items() if k in ('mean','std','min','max')}
              for name,s in stats.items()}
    return make_pre_post_processors(cfg, dataset_stats=values)


def sensor(sample, *, train=False):
    import torch
    result = {'observation.images.camera'+str(i+1):
        torch.from_numpy(sample[k].copy()).permute(2,0,1).unsqueeze(0).float()/255
        for i,k in enumerate(('front','wrist','map'))}
    result['observation.state'] = torch.from_numpy(sample['numeric67'].copy()).unsqueeze(0)
    result['task'] = [sample['instruction']]
    if train:
        result['action'] = torch.from_numpy(sample['action'].copy()).unsqueeze(0)
        result['action_is_pad'] = torch.from_numpy(sample['action_is_pad'].copy()).unsqueeze(0)
    return result


def prepared(pre, sample, *, train=False):
    value = pre(sensor(sample, train=train))
    value['observation.state'] = value['observation.state'].clamp(-5.,5.)
    return value


class Runtime:
    def __init__(self, parent, stats):
        import torch
        from VLA.smolvla.precision import fp32_policy_class
        cfg = config(parent)
        master = read(SMOL/'fp32_master_contract.json')
        self.model = fp32_policy_class(master.get('train_visual_connector',False)).from_pretrained(str(parent), config=cfg)
        self.model = self.model.to('cuda')
        self.pre, self.post = processors(cfg, stats)
        self.stats = stats
        self.model.reset()
        if any(p.dtype != torch.float32 for p in self.model.parameters() if p.requires_grad):
            raise ValueError('Actual learned Smol masters must remain FP32')

    def loss(self, sample):
        batch = prepared(self.pre,sample,train=True)
        images, masks = self.model.prepare_images(batch)
        if len(images)!=3 or not all(bool(v.all()) for v in masks):
            raise ValueError('Actual Smol native forward must consume all three views')
        loss, _ = self.model(batch)
        return loss

    def predict(self,sample,seed):
        import torch
        self.model.eval(); self.model.reset()
        batch = prepared(self.pre,sample)
        images,masks = self.model.prepare_images(batch)
        if len(images)!=3 or not all(bool(v.all()) for v in masks): raise ValueError('Third native view dropped')
        generator = torch.Generator(device='cuda').manual_seed(int(seed))
        noise = torch.randn((1,5,self.model.config.max_action_dim), device='cuda', generator=generator)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            action = self.model.predict_action_chunk(batch,noise=noise)
            # Native action postprocessor accepts arbitrary leading dimensions.
            value = self.post(action).float().cpu().numpy()[0]
        if value.shape!=(5,7) or not np.isfinite(value).all(): raise ValueError('Malformed genuine Smol action chunk')
        return value


class BaselineRuntime(Runtime):
    """Preserved two-view/old-normalizer policy, no untrained map/clipping."""
    def __init__(self,parent):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from VLA.smolvla.precision import fp32_policy_class
        from VLA.common.provenance import sha256_file
        from safetensors import safe_open
        contract=read(parent/'scene67_contract.json')
        for path,digest in contract['input_bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Frozen baseline implementation changed')
        master=read(parent/'fp32_master_contract.json')
        self.model=fp32_policy_class(master.get('train_visual_connector',False)).from_pretrained(str(parent)).to('cuda')
        actual=dict(self.model.named_parameters())
        with safe_open(parent/'model.safetensors',framework='pt',device='cpu') as saved:
            for name in master['trainable_parameter_names']:
                if not torch.equal(saved.get_tensor(name),actual[name].detach().cpu()):raise ValueError('Lossy baseline reload')
        self.pre,self.post=make_pre_post_processors(self.model.config,str(parent),
            preprocessor_overrides={'device_processor':{'device':'cuda'}})

    def predict(self,sample,seed):
        import torch
        self.model.eval();self.model.reset()
        batch=sensor(sample);batch.pop('observation.images.camera3')
        batch=self.pre(batch)
        generator=torch.Generator(device='cuda').manual_seed(int(seed))
        noise=torch.randn((1,5,self.model.config.max_action_dim),device='cuda',generator=generator)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=self.post(self.model.predict_action_chunk(batch,noise=noise)).float().cpu().numpy()[0]
        if result.shape!=(5,7) or not np.isfinite(result).all():raise ValueError('Malformed native baseline actions')
        return result
