"""Explicit RDT+SimState input preparation; native action layout is unchanged.

The eleven additional causal features occupy separate128-D conditioning
slots112..122. These are NOT action DOFs: the output/action mask remains the
original six UR5e joints plus gripper-open. No teacher actions/phase/reward.
This module by itself does NOT authorize training or deploy a base checkpoint.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.rdt.training_data import PyBulletRDTDataset, wrap_joints
from VLA.smolvla.simstate_dataset import load_projection

EXTRA_SLOTS = tuple(range(112,123))
SCHEMA = "rdt_simstate18_native7_plus_normalized11_conditioning_v1"


def append_causal_conditioning(native_states, state18, stats):
    original=torch.as_tensor(native_states,dtype=torch.float32)
    state=np.asarray(state18,dtype=np.float32)
    if state.ndim==1: state=state[None]
    if original.shape!=(len(state),1,128) or state.shape!=(len(state),18) or not np.isfinite(state).all():
        raise ValueError("Expected current finite state18 and native RDT Bx1x128 state")
    expected_q=torch.from_numpy(wrap_joints(state[:,:6]))
    if not torch.allclose(original[:,0,:6],expected_q,rtol=0,atol=1e-6):
        raise ValueError("Privileged and native joint observations differ")
    if not torch.allclose(original[:,0,10],torch.from_numpy(1-state[:,6]),rtol=0,atol=1e-6):
        raise ValueError("Native gripper-open and causal encoder-closure differ")
    if torch.any(original[:,:,list(EXTRA_SLOTS)]!=0):
        raise ValueError("Never overwrite existing native state fields")
    mean,std=(np.asarray(stats[k],dtype=np.float32) for k in ("mean","std"))
    if mean.shape!=(18,) or std.shape!=(18,) or np.any(std<0) or not np.isfinite(np.r_[mean,std]).all():
        raise ValueError("Full TRAIN-only state18 normalization required")
    normalized=(state[:,7:]-mean[7:])/(std[7:]+np.float32(1e-8))
    if not np.isfinite(normalized).all(): raise ValueError("Non-finite privileged conditioning")
    result=original.clone()
    result[:,0,list(EXTRA_SLOTS)]=torch.from_numpy(normalized)
    return result


class RDTSimStateDataset(PyBulletRDTDataset):
    def __init__(self,source,projection,**kwargs):
        super().__init__(source,**kwargs)
        values,audit,self.simstate_stats=load_projection(projection)
        if sha256_file(Path(source)/"index.json")!=audit["source_index_sha256"]:
            raise ValueError("RDT source must have the EXACT same full episode ordering")
        parent=Path(audit["source"])
        original=json.loads((parent/"index.json").read_text())
        if len(original)!=len(self.index) or [r["file"] for r in original]!=[r["file"] for r in self.index]:
            raise ValueError("No subset or remapping of the full causal state18 projection")
        self.projected_states={};offset=0
        for row,proof in zip(self.index,audit["rows"],strict=True):
            if sha256_file(Path(source)/row["file"])!=proof["source_payload_sha256"]:
                raise ValueError("RDT source payload differs from the real causal replay")
            count=int(row["steps"])
            self.projected_states[row["file"]]=values[offset:offset+count];offset+=count
        self.simstate_audit=audit

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        current=self.projected_states[sample.episode_file][sample.step_ids]
        states=append_causal_conditioning(sample.states,current,self.simstate_stats).numpy()
        # Dataclass replacement preserves native joint/action/gripper labels,
        # image history, masks, text and requested timestamps byte for byte.
        return replace(sample,states=states)


def prepare_privileged_sensor_tensors(payload,processor,embeddings,previous_images=None,*,stats):
    from VLA.rdt.server import prepare_sensor_tensors
    if payload.get("simstate_schema")!=SCHEMA or "privileged_state18" not in payload:
        raise ValueError("Explicit privileged input schema required; no silent base-model route")
    inputs=prepare_sensor_tensors(payload,processor,embeddings,target_domain=True,previous_images=previous_images)
    inputs["states"]=append_causal_conditioning(inputs["states"],payload["privileged_state18"],stats)
    return inputs
