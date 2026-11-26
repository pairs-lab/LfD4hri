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
    if 'robot21' not in stats:
        raise ValueError('Historical scene32/scene64 recipe is disabled here; use protocol21 and TRAIN robot21 statistics')
    from protocol21.pi0 import build_config as current
    return current(stats,output,updates)
