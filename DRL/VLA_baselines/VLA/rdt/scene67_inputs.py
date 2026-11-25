"""Full67 feedback in RDT conditioning slots; still only7 motor DOFs.

Numeric extra information uses unused observation slots, NEVER extra action
labels or output DOFs. Legacy11 slots are kept;49 new slots are reserved.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from VLA.common.provenance import sha256_file
from VLA.rdt.training_data import PyBulletRDTDataset, wrap_joints
from VLA.scene64_dataset import load_scene_projection
from VLA.scene64_conditioning import encode_numeric, normalization

SCHEMA = "rdt_scene67_native7_plus_normalized60_conditioning_v1"
EXTRA_SLOTS = tuple(range(112, 123)) + tuple(range(50, 99))


def append_scene_conditioning(native_states, raw64, stats):
    original = torch.as_tensor(native_states, dtype=torch.float32)
    raw = np.asarray(raw64, np.float32)
    if raw.ndim == 1: raw = raw[None]
    if raw.ndim != 2 or raw.shape[1] != 64 or original.shape != (len(raw), 1, 128):
        raise ValueError("Native128 observation and matching causal64 required")
    if not torch.allclose(original[:, 0, :6], torch.from_numpy(wrap_joints(raw[:, :6])), atol=1e-6, rtol=0):
        raise ValueError("Native joints and projected pre-action observation differ")
    if not torch.allclose(original[:, 0, 10], torch.from_numpy(1-raw[:, 6]), atol=1e-6, rtol=0):
        raise ValueError("Native open/closure convention differs")
    if torch.any(original[:, :, list(EXTRA_SLOTS)] != 0): raise ValueError("Do not overwrite native fields")
    encoded = np.asarray([encode_numeric(v) for v in raw])
    mean, std = [np.asarray(stats[k], np.float32) for k in ("mean", "std")]
    if mean.shape != (67,) or std.shape != (67,) or np.any(std < 0) or not np.isfinite(np.r_[mean, std]).all():
        raise ValueError("Own full TRAIN67 normalization required")
    result = original.clone()
    result[:, 0, list(EXTRA_SLOTS)] = torch.from_numpy((encoded[:, 7:]-mean[7:]) / (std[7:]+np.float32(1e-8)))
    if not torch.isfinite(result).all(): raise ValueError("Nonfinite native conditioning")
    return result


class RDTScene67Dataset(PyBulletRDTDataset):
    def __init__(self, source, projection, **kwargs):
        super().__init__(source, **kwargs)
        raw, self.scene_audit, rows = load_scene_projection(projection)
        if [r["file"] for r in rows] != [r["file"] for r in self.index]: raise ValueError("Native full episode order differs")
        self.scene_stats, self.normalization_contract = normalization(raw)
        self.projected = {}; offset = 0
        for row, proof in zip(rows, self.scene_audit["rows"], strict=True):
            if sha256_file(Path(source)/row["file"]) != proof["source_payload_sha256"]:
                raise ValueError("Native source labels changed")
            count = row["steps"]; self.projected[row["file"]] = raw[offset:offset+count]; offset += count

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        current = self.projected[sample.episode_file][sample.step_ids]
        return replace(sample, states=append_scene_conditioning(sample.states, current, self.scene_stats).numpy())
