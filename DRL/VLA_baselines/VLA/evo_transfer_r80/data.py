"""Qualified TRAIN, native bounds normalization and actual temporal masks."""
import numpy as np
from VLA.flower_rotation_r76.sampler import Data as Balanced
from VLA.new_vlas.action_contracts import native8
from VLA.scene64_conditioning import encode_numeric, validate_scene


class Data(Balanced):
    def __init__(self):
        super().__init__()
        self.horizon = 50

    def at(self, ei, fi):
        sample = super().at(ei, fi)
        sample['native8'] = self.numeric(ei)['native8'][fi].copy()
        return sample


def bounds(values):
    values = np.asarray(values, np.float64)
    low, high = values.min(0), values.max(0)
    center = (low + high) / 2
    radius = np.maximum((high - low) / 2, 1e-4)
    return {'min': (center - radius).tolist(), 'max': (center + radius).tolist()}


def fit_stats(data):
    actions = np.concatenate([data.numeric(i)['canonical7'] for i in range(len(data.rows))])
    # Native LIBERO checkpoint: OPEN01, not CLOSE01 and not a gripper delta.
    actions = actions.copy()
    actions[:, 6] = (1 - actions[:, 6]) / 2
    result = {'action': bounds(actions), 'native8': bounds(np.concatenate([
        data.numeric(i)['native8'] for i in range(len(data.rows))])), 'scene67': data.stats['scene67']}
    result['action']['min'][6], result['action']['max'][6] = 0., 1.
    result['schema'] = 'native_bounds_EEF7_OPEN01_native24_plus_causal67_R80'
    return result


def normalize(values, stat, *, clip=False):
    lo, hi = [np.asarray(stat[k], np.float32) for k in ('min', 'max')]
    result = 2 * (np.asarray(values, np.float32) - lo) / (hi - lo) - 1
    return np.clip(result, -1, 1) if clip else result


def denormalize(values, stat):
    lo, hi = [np.asarray(stat[k], np.float32) for k in ('min', 'max')]
    return (np.asarray(values, np.float32) + 1) / 2 * (hi - lo) + lo


def state(sample, stats):
    prefix = normalize(sample['native8'], stats['native8'], clip=True)
    mean, std = [np.asarray(stats['scene67'][k], np.float32) for k in ('mean', 'std')]
    numeric = np.clip((sample['numeric67'] - mean) / (std + 1e-8), -5, 5)
    result = np.r_[prefix, np.zeros(16, np.float32), numeric].astype(np.float32)
    if result.shape != (91,) or not np.isfinite(result).all():
        raise ValueError('Native24 prefix or complete causal67 is missing')
    return result


def targets(sample, stats):
    action = sample['action'].copy()
    if action.shape != (50, 7) or sample['action_is_pad'].shape != (50,):
        raise ValueError('Native50 horizon required')
    action[:, 6] = (1 - action[:, 6]) / 2
    output = np.zeros((50, 24), np.float32)
    mask = np.zeros_like(output)
    output[:, :7] = normalize(action, stats['action'])
    mask[:, :7] = (~sample['action_is_pad'])[:, None]
    output *= mask
    if not mask.any() or not np.isfinite(output).all():
        raise ValueError('No actual future action labels')
    return output, mask


def canonical(native, stats):
    native = np.asarray(native, np.float32)
    if native.shape != (50, 24) or not np.isfinite(native).all():
        raise ValueError('Malformed native Evo-1 output')
    result = denormalize(native[:, :7], stats['action'])
    result[:, 6] = 1 - 2 * np.clip(result[:, 6], 0, 1)
    # No geometric residual, waypoint, double-delta or post-hoc movement repair.
    return result.astype(np.float32)


def observed_sample(payload, front, wrist, scene_map):
    raw = validate_scene(payload['scene64'])
    return {'instruction': payload['instruction'], 'front': front, 'wrist': wrist,
            'map': scene_map, 'numeric67': encode_numeric(raw),
            'native8': native8(np.asarray(payload['ee_position'], np.float32),
                np.asarray(payload['ee_rotation_6d'], np.float32), raw[27])}
