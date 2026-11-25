"""Declared source-preserving encoder conditioning, never motor correction."""
import json
from pathlib import Path

import numpy as np

from VLA.smolvla.joint_angle_alias import AngleAliasSimStatePolicy, aliases


def nearstatic_clip(values, stats):
    """Bound sub-.01rad excursions of axes whose TRAIN variance is tiny.

    In-range TRAIN values stay byte-identical. This is approximate sensor
    conditioning, NOT a claim that clipped angles describe an identical pose.
    Actual simulation pose and all privileged spatial observations are kept.
    """
    original = np.asarray(values,dtype=np.float32)
    if original.shape[-1] != 6 or not np.isfinite(original).all():
        raise ValueError("Finite joint observations required")
    mean, std, low, high = [np.asarray(stats[k][:6],dtype=np.float64) for k in ("mean","std","min","max")]
    eligible = (std < .001) & (high-low < .01)
    if not np.isfinite(np.r_[mean,std,low,high]).all() or np.any(std < 0) or np.any(low > high):
        raise ValueError("Invalid immutable TRAIN statistics")
    limited = np.clip(original,low,high).astype(np.float32)
    change = eligible & (np.abs(original-mean) <= .01) & (limited != original)
    result = original.copy(); result[change] = limited[change]
    return result,change


class RobustEncoderSimStatePolicy(AngleAliasSimStatePolicy):
    name = "smolvla_simstate_robust_encoder"
    input_modalities = AngleAliasSimStatePolicy.input_modalities+("bounded_nearstatic_encoder_conditioning",)

    def __init__(self,env,checkpoint,projection,**kwargs):
        self.encoder_stats = json.loads((Path(projection)/"stats.json").read_text())
        super().__init__(env,checkpoint,projection,**kwargs)

    def build_sensor_batch(self,observation):
        import torch
        batch = super().build_sensor_batch(observation)
        values = batch["observation.state"].numpy().copy()
        values[:,:6],_ = nearstatic_clip(values[:,:6],self.encoder_stats)
        batch["observation.state"] = torch.from_numpy(values)
        return batch
