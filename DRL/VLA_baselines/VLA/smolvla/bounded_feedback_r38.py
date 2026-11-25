"""Opt-in finite-state transport ablation; no action repair or hidden planner.

Use the checkpoint's ACTUAL saved native mean/std, including its epsilon.
Raw scene64 remains recorded. Each normalized measured component is bounded
to +/-10; this does not assert that the resulting policy is better. Original
evaluators/checkpoints/training data are never modified.
"""
from pathlib import Path
import numpy as np
import torch
from VLA.smolvla.scene67_adapter import Scene67SmolVLA

SCHEMA = 'SmolVLA_simscene67_native20_saved_normalizer_bound10_r38'
LIMIT = 10.


def load_stats(checkpoint):
    from safetensors import safe_open
    with safe_open(Path(checkpoint)/'policy_preprocessor_step_5_normalizer_processor.safetensors', framework='np') as f:
        mean = f.get_tensor('observation.state.mean').copy()
        std = f.get_tensor('observation.state.std').copy()
    if mean.shape != (67,) or std.shape != (67,) or not np.isfinite(np.r_[mean, std]).all() or np.any(std < 0):
        raise ValueError('Actual finite native67 normalization required')
    return mean, std


def bound(values, mean, std, limit=LIMIT):
    values = np.asarray(values, np.float32)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if values.shape[-1:] != (67,) or mean.shape != (67,) or std.shape != (67,) or not np.isfinite(values).all():
        raise ValueError('Finite native67 measured input required')
    if not np.isfinite(limit) or limit <= 0 or np.any(std < 0):
        raise ValueError('Positive finite normalized bound required')
    radius = np.float32(limit)*(std+np.float32(1e-8))
    result = np.clip(values, mean-radius, mean+radius).astype(np.float32)
    return result, result != values


class BoundedFeedbackSmolVLA(Scene67SmolVLA):
    """Actual native NN policy, with explicitly declared observation clipping."""
    def __init__(self, env, checkpoint, **kwargs):
        super().__init__(env, checkpoint, **kwargs)
        self.bound_mean, self.bound_std = load_stats(checkpoint)

    def build_sensor_batch(self, observation):
        batch = super().build_sensor_batch(observation)
        raw = self.last_input['numeric67']
        bounded, changed = bound(raw, self.bound_mean, self.bound_std)
        batch['observation.state'] = torch.from_numpy(bounded).unsqueeze(0)
        self.last_input.update(bounded_numeric67=bounded.copy(), clipped_numeric_indices=np.flatnonzero(changed).tolist(),
            input_variant=SCHEMA, raw_numeric67_preserved=True)
        return batch
