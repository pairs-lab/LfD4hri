"""Native RDT with learned, zero-initialized gains for49 new observations.

Legacy18 conditioning is unchanged at initialization; the49 newly observed
slots cannot perturb pretrained action/noise pathways before adaptation.
No added motor DOFs, teacher phase, planner or action correction.
"""
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import numpy as np
import torch
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import load_scene_projection
from VLA.scene64_conditioning import normalization
from VLA.native_corrective_dataset import NativeCorrectiveDataset,rdt_dataset
from VLA.rdt.training_data import PyBulletRDTDataset
from VLA.rdt.scene67_inputs import append_scene_conditioning
from VLA.rdt.checkpoint_loading import load_checkpoint as _native_load

SCHEMA='rdt_scene67_native7_learned_new49_input_gains_v1'
NEW_SLOTS=tuple(range(50,99))


def gated_state(model,state):
    result=state.clone()
    result[...,list(NEW_SLOTS)]=state[...,list(NEW_SLOTS)]*model.scene67_gain
    return result


def install_gains(model):
    if hasattr(model,'scene67_gain'):raise ValueError('Do not duplicate learned observation gains')
    model.register_parameter('scene67_gain',torch.nn.Parameter(torch.zeros(49,dtype=torch.float32)))
    for name in ('compute_loss','predict_action'):
        original=getattr(model,name);signature=inspect.signature(original)
        def call(self,*args,_original=original,_signature=signature,**kwargs):
            bound=_signature.bind(*args,**kwargs)
            bound.arguments['state_tokens']=gated_state(self,bound.arguments['state_tokens'])
            return _original(*bound.args,**bound.kwargs)
        setattr(model,name,MethodType(call,model))
    return model


def load(checkpoint,horizon,*,dtype=torch.float32):
    # The original loader still constructs the native RDTRunner and strictly
    # copies every retained tensor. Install only the extra parameter before
    # loading a new-format checkpoint containing it.
    from models.rdt_runner import RDTRunner
    original=RDTRunner.load_state_dict
    def restored(model,state,*args,**kwargs):
        if 'scene67_gain' in state:install_gains(model)
        return original(model,state,*args,**kwargs)
    with patch.object(RDTRunner,'load_state_dict',restored):model,adaptation=_native_load(checkpoint,horizon,dtype=dtype)
    if not hasattr(model,'scene67_gain'):install_gains(model)
    return model,adaptation


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self,source,projection,admission,**kwargs):
        self.base=PyBulletRDTDataset(source,**kwargs)
        raw,self.audit,rows=load_scene_projection(projection)
        if [r['file'] for r in rows]!=[r['file'] for r in self.base.index]:raise ValueError('Full source order differs')
        self.projected={};offset=0
        for row,proof in zip(rows,self.audit['rows'],strict=True):
            if sha256_file(Path(source)/row['file'])!=proof['source_payload_sha256']:raise ValueError('Native original source changed')
            self.projected[row['file']]=raw[offset:offset+row['steps']];offset+=row['steps']
        correction=NativeCorrectiveDataset(admission,10);self.correction_data=correction
        self.stats,self.normalization_contract=normalization(np.concatenate((raw,correction.states)))
        old,_=normalization(raw)
        for key in ('mean','std','min','max'):self.stats[key][:18]=old[key][:18]
        self.normalization_contract['legacy18_reference_preserved']=True
        self.corrective=rdt_dataset(correction,self.stats,**kwargs)
        self.index=self.base.index+self.corrective.index;self.original_length=len(self.base)
        self.recovery_contract={'admission':str(Path(admission).resolve()),'sha256':sha256_file(admission),
            'episodes':38,'frames':len(correction),'first18_reference_preserved':True}
    @property
    def prompts(self):return self.base.prompts+self.corrective.prompts
    def __len__(self):return len(self.index)
    def set_epoch(self,epoch):self.base.set_epoch(epoch);self.corrective.set_epoch(epoch)
    def __getitem__(self,index):
        if index>=self.original_length:return self.corrective[index-self.original_length]
        sample=self.base[index]
        raw=self.projected[sample.episode_file][sample.step_ids]
        return replace(sample,states=append_scene_conditioning(sample.states,raw,self.stats).numpy())
    def weights(self):
        # Rehearsal80% task/target balanced; corrective20% task balanced.
        from collections import Counter
        base_counts=Counter((r['task']['name'],r['task']['pick_name']) for r in self.base.index)
        task_targets=Counter(k[0] for k in base_counts)
        new_counts=Counter(r['task']['name'] for r in self.corrective.index)
        return [.8/(4*task_targets[r['task']['name']]*base_counts[r['task']['name'],r['task']['pick_name']]) for r in self.base.index]+[
            .2/(4*new_counts[r['task']['name']]) for r in self.corrective.index]
