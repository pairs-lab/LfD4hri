"""Official FAST32×32 action tokens; seven physical EEF controls, causal GT3."""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from VLA.family_transfer_r54.data import Data
from VLA.family_transfer_r54.pi0 import pi_sample, statistics
from .common import SCHEMA, native_contract, read

CODEC_SCALE = 30.0  # TRAIN-only fidelity calibration; no final-cohort outcomes.
DECODING_STEPS = 256


class FastData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, horizon=32, **kwargs)

    def __getitem__(self, index):
        return pi_sample(super().__getitem__(index))


def build_config(stat, output, updates):
    from openpi import transforms
    from openpi.shared.normalize import NormStats
    from openpi.training import config
    from VLA.pi0.eef_scene32_training import build_config as original
    from VLA.pi0.scene32_inputs import state32
    c = original('pi0_fast',stat,output,updates)
    c = replace(c, model=replace(c.model, fast_model_tokenizer_kwargs={
        **c.model.fast_model_tokenizer_kwargs, 'codec_scale': CODEC_SCALE}))
    norm={key:NormStats(**{k:np.asarray(v,np.float32) for k,v in values.items()}) for key,values in stat.items()}
    @dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self,x):
            out={'state':state32(x['pre_action_scene64']),
                 'image':{'base_0_rgb':x['image'],'base_1_rgb':x['map_image'],'wrist_0_rgb':x['wrist_image']},
                 'image_mask':{k:np.True_ for k in ('base_0_rgb','base_1_rgb','wrist_0_rgb')},
                 'prompt':x['prompt']}
            if 'actions' in x:
                a=x['actions']
                if a.shape!=(32,7):raise ValueError('Require32×7 actual physical EEF labels')
                out['actions']=np.pad(a,((0,0),(0,25)))
            return out
    @dataclass(frozen=True)
    class StateBound(transforms.DataTransformFn):
        def __call__(self,x):return {**x,'state':np.clip(x['state'],-5,5)}
    @dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self,x):return {'actions':np.asarray(x['actions'])[...,:7]}
    @dataclass(frozen=True)
    class Factory(config.DataConfigFactory):
        def create(self,assets_dirs,model_config):
            mt=config.ModelTransformFactory()(model_config)
            return config.DataConfig(repo_id=self.repo_id,asset_id='R65_FAST32_EEF_GT3',norm_stats=norm,
                use_quantile_norm=True,data_transforms=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]),
                model_transforms=transforms.Group(inputs=[StateBound(),*mt.inputs],outputs=mt.outputs))
    # Normalization expects32 channels before official FAST tokenization.
    # The25 nonphysical dimensions are zero labels with nondegenerate stats.
    from openpi.shared.normalize import NormStats
    for key in ('mean','std','q01','q99'):
        values=np.asarray(stat['actions'][key],np.float32)
        fill=1. if key in ('std','q99') else -1. if key=='q01' else 0.
        values=np.r_[values,np.full(25,fill,np.float32)]
        norm['actions']=replace(norm['actions'],**{key:values})
    return replace(c,name='pi0_fast_r65_native_EEF_GT3',data=Factory(repo_id='local/R65_FAST_R52_complete'),
        policy_metadata={**c.policy_metadata,'schema':SCHEMA,'full64_text_and_GT3':True,
            'physical_action_dimensions':7,'native_FAST_coefficient_layout':[32,32],
            'FAST_codec_scale':CODEC_SCALE,'native_max_decoding_steps':DECODING_STEPS})


class Runtime:
    def __init__(self, checkpoint):
        from flax import nnx
        from flax.traverse_util import flatten_dict
        from openpi.models import model as model_api
        from openpi.policies import policy_config
        self.contract=native_contract(checkpoint)
        self.cfg=build_config(self.contract['stats'],checkpoint,self.contract['actual_optimizer_updates'])
        precision=read(self.contract['precision']); restore=model_api.restore_params
        def exact(path,*args,**kwargs):
            if Path(path).resolve()!=(checkpoint/'params').resolve():raise ValueError('Wrong native FAST restore')
            kwargs['dtype']=None;return restore(path,*args,**kwargs)
        with patch.object(model_api,'restore_params',exact):
            self.native=policy_config.create_trained_policy(self.cfg,str(checkpoint),
                sample_kwargs={'max_decoding_steps':DECODING_STEPS})
        values=flatten_dict(nnx.state(self.native._model).filter(self.cfg.trainable_filter).to_pure_dict())
        for row in precision['trainable_tensors']:
            v=np.asarray(values[tuple(row['path'])])
            if str(v.dtype)!='float32' or hashlib.sha256(v.tobytes()).hexdigest()!=row['saved_sha256']:
                raise ValueError('Native FAST learned FP32 value changed on fresh reload')
        self.exact_values=True
        # This wrapper runs on the actual infer() path before Observation.from_dict.
        native_transform=self.native._input_transform
        def checked_transform(raw):
            from VLA.scene64_conditioning import parse_prompt
            np.testing.assert_allclose(parse_prompt(raw['prompt']),raw['pre_action_scene64'],rtol=0,atol=.000051)
            prepared=native_transform(raw)
            if (np.asarray(prepared['state']).shape!=(32,) or len(prepared['image'])!=3 or
                not all(bool(v) for v in prepared['image_mask'].values())):
                raise ValueError('Actual native FAST omitted privileged state/text/GT3')
            n=int(np.asarray(prepared['tokenized_prompt_mask']).sum())
            if n>=self.cfg.model.max_token_len:
                raise ValueError('Actual native FAST full64 feedback truncated')
            self.last_input_receipt={'actual_native_views':3,'actual_state_dimensions':32,
                'complete_scene64_text_verified':True,'actual_prefix_tokens':n}
            return prepared
        self.native._input_transform=checked_transform

    def predict(self,x,seed):
        import jax
        raw=pi_sample(x);raw.pop('actions',None)
        prepared=self.native._input_transform(dict(raw))
        if len(prepared['image'])!=3 or not all(bool(v) for v in prepared['image_mask'].values()):
            raise ValueError('Native FAST view missing')
        # Native Policy maintains its own sampling key. Reset it explicitly,
        # like the existing per-query scene/instruction RNG policy.
        self.native._rng=jax.random.key(int(seed)&0x7fffffff)
        out=np.asarray(self.native.infer(raw)['actions'],np.float32)
        if out.shape!=(32,7) or not np.isfinite(out).all():raise ValueError('Malformed genuine FAST32 chunk')
        out[:,6]=2*np.clip(out[:,6],0,1)-1
        return out
