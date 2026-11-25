"""Official pi0 LoRA / EEF flow objective, full64 text + native32 + GT3.

The original joint-output checkpoint remains intact. This recipe learns the
same canonical EEF commands as Smol; it does not relabel old weights as adapted.
"""
from dataclasses import dataclass,replace
import numpy as np
from .data import Data, action_stats


class PiData(Data):
    def __getitem__(self,index):
        x=super().__getitem__(index)
        return pi_sample(x)
    def at(self,ei,fi):
        return super().at(ei,fi)


def pi_sample(x):
    actions=x['action'].copy();actions[:,6]=(actions[:,6]+1)/2
    raw=x['raw64']
    return {**x,'image':x['front'],'wrist_image':x['wrist'],'map_image':x['map'],
        'actions':actions,'canonical_actions':x['action'].copy(),
        'joints':((raw[:6]+np.pi)%(2*np.pi)-np.pi).astype(np.float32),'gripper':raw[6:7].copy(),
        'pre_action_scene64':raw.copy()}


def statistics(data):
    from VLA.pi0.scene32_inputs import statistics as state_stats
    raw=np.concatenate([data.numeric(ei)['scene64'] for ei in range(len(data.rows))])
    state,regularization=state_stats(raw)
    # Prevent near-static contact/rotation/velocity channels from exploding.
    from VLA.pi0.scene32_inputs import RAW_SLOTS
    for i,raw_index in enumerate(RAW_SLOTS):
        if i<7 or raw_index in (10,17,28,29,30):continue
        floor=1. if raw_index==39 else .03 if 11<=raw_index<=16 else .01
        state['std'][i]=max(state['std'][i],floor)
    return {'state':state,'actions':action_stats(data,closure01=True)}


def build_config(stats,output,updates):
    from openpi import transforms
    from openpi.training import config
    from openpi.shared.normalize import NormStats
    from VLA.pi0.eef_scene32_training import build_config as original
    from VLA.pi0.scene32_inputs import state32
    cfg=original('pi0',stats,output,updates)
    norm={key:NormStats(**{k:np.asarray(v,np.float32) for k,v in values.items()}) for key,values in stats.items()}
    @dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self,raw):
            out={'state':state32(raw['pre_action_scene64']),
                'image':{'base_0_rgb':raw['image'],'left_wrist_0_rgb':raw['wrist_image'],'right_wrist_0_rgb':raw['map_image']},
                'image_mask':{k:np.True_ for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},'prompt':raw['prompt']}
            if 'actions' in raw:out['actions']=raw['actions'].copy()
            return out
    @dataclass(frozen=True)
    class ClipState(transforms.DataTransformFn):
        def __call__(self,raw):return {**raw,'state':np.clip(raw['state'],-5.,5.)}
    @dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self,raw):return {'actions':np.asarray(raw['actions'])[...,:7]}
    @dataclass(frozen=True)
    class Factory(config.DataConfigFactory):
        def create(self,assets_dirs,model_config):
            mt=config.ModelTransformFactory()(model_config)
            return config.DataConfig(repo_id=self.repo_id,asset_id='r54_pi0_GT3_EEF',norm_stats=norm,
                use_quantile_norm=False,data_transforms=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]),
                model_transforms=transforms.Group(inputs=[ClipState(),*mt.inputs],outputs=mt.outputs))
    return replace(cfg,name='pi0_r54_native_EEF_GT3',data=Factory(repo_id='local/pi0_R52_qualified119495_GT3'),
        policy_metadata={**cfg.policy_metadata,'extra_view':'causal_simulator_GT_not_real_right_wrist',
            'native32_and_full64_language_preserved':True,'standardized_state_bound':5.})
