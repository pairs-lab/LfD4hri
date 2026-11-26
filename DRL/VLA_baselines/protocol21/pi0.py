"""OpenPI LoRA recipe: numeric robot21+zero11, raw named21 text, two RGB views."""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def payload(sample):
    state = np.asarray(sample['state'], np.float32)
    if state.shape != (32,) or np.any(state[21:]):
        raise ValueError('Expected normalized robot21 followed by eleven exact zeros')
    result = {'state': state.copy(), 'image': {'base_0_rgb': sample['front'], 'left_wrist_0_rgb': sample['wrist']},
              'image_mask': {'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.True_}, 'prompt': sample['prompt']}
    if 'action' in sample:
        action = np.asarray(sample['action']).copy(); action[:, 6] = (action[:, 6]+1)/2
        result['actions'] = action
    return result


def build_config(stats, output, updates=3000):
    from openpi import transforms
    from openpi.shared.normalize import NormStats
    from openpi.training import config, optimizer
    from VLA.pi0.training import make_config
    args = SimpleNamespace(model='pi0', steps=updates, gradient_accumulation=4,
                           action_horizon=5, repo_id='local/robot21', exp_name=Path(output).name,
                           batch_size=1, overwrite=False, resume=False, save_interval=250, asset_id='robot21')
    base = make_config(args)
    # Numeric state is already normalized by Inputs, and must not be normalized twice.
    actions = {k:np.asarray(v, np.float32).copy() for k,v in stats['actions'].items()}
    for key in ('mean','q01','q99'): actions[key][6] = (actions[key][6]+1)/2
    actions['std'][6] /= 2
    norm = {'actions': NormStats(**actions)}
    @dataclass(frozen=True)
    class InputTransform(transforms.DataTransformFn):
        def __call__(self, x): return payload(x)
    @dataclass(frozen=True)
    class OutputTransform(transforms.DataTransformFn):
        def __call__(self, x): return {'actions': np.asarray(x['actions'])[..., :7]}
    @dataclass(frozen=True)
    class Factory(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return config.DataConfig(repo_id=self.repo_id, asset_id='robot21', norm_stats=norm,
                use_quantile_norm=False, data_transforms=transforms.Group(inputs=[InputTransform()],outputs=[OutputTransform()]),
                model_transforms=config.ModelTransformFactory()(model_config))
    return replace(base, name='pi0_robot21', model=replace(base.model,max_token_len=1024),
                   data=Factory(repo_id='local/robot21'),
                   lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,decay_steps=updates,
                                                            peak_lr=2.5e-5,decay_lr=2.5e-6))


class Runtime:
    def __init__(self, stats, checkpoint):
        from openpi.policies import policy_config
        cfg=build_config(stats,checkpoint)
        norm=cfg.data.create(cfg.assets_dirs,cfg.model).norm_stats
        self.policy = policy_config.create_trained_policy(cfg, checkpoint, norm_stats=norm,
                                                         sample_kwargs={'num_steps':10})

    def predict(self, sample, seed=0):
        # The native policy owns its RNG; no synthetic action/noise fallback.
        result = self.policy.infer({k:v for k,v in sample.items() if k not in ('action','action_is_pad')})
        x = np.asarray(result['actions'], np.float32).copy()
        x[:,6] = 2*np.clip(x[:,6],0,1)-1
        return x
