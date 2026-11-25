"""Equivalent encoder-angle preprocessing, NOT a pose/controller correction.

Choose the2*pi-equivalent measurement closest to the immutable TRAIN mean.
Full TRAIN extrema MUST fit that interval, so every training value is kept
byte-identical; only out-of-interval deployment windings are changed.
"""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from VLA.common.provenance import sha256_file
from VLA.smolvla.simstate_adapter import SimStateSmolVLA


def range_proof(stats,means):
    means=np.asarray(means,dtype=np.float64)
    low,high=(np.asarray(stats[k][:6],dtype=np.float64) for k in ("min","max"))
    if means.shape!=(6,) or not np.isfinite(np.r_[means,low,high]).all():raise ValueError("Invalid TRAIN angle limits")
    if not np.all(low-means>=-np.pi) or not np.all(high-means<np.pi):
        raise ValueError("Mean-centered aliasing would change TRAIN samples; requires a new source/fit")
    return {"all_TRAIN_joint_values_UNCHANGED":True,"train_min_relative_to_mean":(low-means).tolist(),
        "train_max_relative_to_mean":(high-means).tolist(),"means":means.tolist(),
        "physical_pose_controller_or_actions_changed":False}


def aliases(values,means):
    values=np.asarray(values,dtype=np.float32);means=np.asarray(means,dtype=np.float64)
    if values.shape[-1]!=6 or means.shape!=(6,) or not np.isfinite(values).all():raise ValueError("Finite six-joint measurement required")
    turns=np.floor((values.astype(np.float64)-means+np.pi)/(2*np.pi)).astype(np.int64)
    result=values.copy();change=turns!=0
    # Do not add float roundoff to in-range training/deployment values.
    shifted=values.astype(np.float64)-turns*(2*np.pi)
    result[change]=shifted[change].astype(np.float32)
    return result,turns


class AngleAliasSimStatePolicy(SimStateSmolVLA):
    name="smolvla_simstate_angle_alias"
    input_modalities=SimStateSmolVLA.input_modalities+("TRAIN_equivalent_encoder_angle_aliases",)

    def __init__(self,env,checkpoint,projection,**kwargs):
        with safe_open(Path(checkpoint)/"policy_preprocessor_step_5_normalizer_processor.safetensors",framework="np") as data:
            self.angle_means=data.get_tensor("observation.state.mean")[:6].astype(np.float64)
        stats_path=Path(projection)/"stats.json"
        self.angle_proof={**range_proof(json.loads(stats_path.read_text()),self.angle_means),
            "TRAIN_stats_sha256":sha256_file(stats_path),"implementation_sha256":sha256_file(Path(__file__).resolve())}
        super().__init__(env,checkpoint,**kwargs)

    def build_sensor_batch(self,observation):
        import torch
        batch=super().build_sensor_batch(observation)
        values=batch["observation.state"].numpy().copy()
        values[:,:6],_=aliases(values[:,:6],self.angle_means)
        batch["observation.state"]=torch.from_numpy(values)
        return batch
