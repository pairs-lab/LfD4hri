"""SmolVLA shared processor helpers; current runtime is robot21/two-view.

Only observation profile, TRAIN normalization and sampling change. Every
action is produced by the native policy, not a simulator-state controller.
"""
import numpy as np
from .common import SMOL


def config(parent=SMOL, device='cuda'):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    cfg = SmolVLAConfig.from_pretrained(str(parent))
    if cfg.type != 'smolvla':
        raise ValueError('Native SmolVLA foundation required')
    cfg.device = device
    cfg.max_state_dim=21
    cfg.chunk_size=5
    cfg.input_features={f'observation.images.camera{i}':PolicyFeature(type=FeatureType.VISUAL,shape=(3,384,384)) for i in (1,2)}
    cfg.input_features['observation.state']=PolicyFeature(type=FeatureType.STATE,shape=(21,))
    cfg.freeze_vision_encoder = True
    cfg.n_action_steps = 5
    cfg.compile_model = False
    return cfg


def statistics(data):
    from .data import action_stats
    return {'observation.state': data.stats['robot21'], 'action': action_stats(data)}


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
        for i,k in enumerate(('front','wrist'))}
    state=np.asarray(sample['raw_state21'],np.float32)
    if state.shape!=(21,):raise ValueError('Expected raw_state21; historical numeric67 inputs are rejected')
    result['observation.state'] = torch.from_numpy(state.copy()).unsqueeze(0)
    result['task'] = [sample['instruction']]
    if train:
        result['action'] = torch.from_numpy(sample['action'].copy()).unsqueeze(0)
        result['action_is_pad'] = torch.from_numpy(sample['action_is_pad'].copy()).unsqueeze(0)
    return result


def prepared(pre, sample, *, train=False):
    value = pre(sensor(sample, train=train))
    return value


class Runtime:
    def __init__(self, parent, stats):
        from protocol21.native import Smol
        if 'robot21' not in stats:raise ValueError('Use TRAIN robot21 statistics, not an old-width normalizer')
        self.current=Smol(stats,parent,'cuda')
        self.model,self.pre,self.post=self.current.model,self.current.pre,self.current.post

    def loss(self, sample):
        return self.current.loss(sample)

    def predict(self,sample,seed):
        return self.current.predict(sample,seed)


class BaselineRuntime(Runtime):
    """Fail explicitly instead of silently loading the old observation protocol."""
    def __init__(self,parent):
        raise ValueError('Historical baseline is incompatible with robot21; use protocol21.native.create with new statistics')
