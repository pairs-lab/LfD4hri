"""Supervised targets are kept separate from observation packing."""
import numpy as np


def quantile(action, stats):
    x = np.asarray(action, np.float32).copy()
    lo, hi = (np.asarray(stats[k], np.float32) for k in ('q01', 'q99'))
    if lo.shape != (7,) or hi.shape != (7,) or np.any(hi <= lo):
        raise ValueError('Finite nondegenerate TRAIN action bounds required')
    x[..., :6] = np.clip(2*(x[..., :6]-lo[:6])/(hi[:6]-lo[:6])-1, -1, 1)
    return x


def attach_targets(family, sample, stats):
    canonical = sample['action'].copy()
    if family == 'xvla':
        # Labels at future measured timestamps are supplied by Bank below, not
        # integrated from the first observation's pose.
        sample['canonical_action'] = canonical
    elif family == 'vla_adapter':
        from VLA.new_vlas.action_contracts import normalize_bounds
        sample['canonical_action'] = canonical
        canonical[:, 6] = (1-canonical[:, 6])/2
        sample['action'] = normalize_bounds(canonical, stats['adapter']['action'])
    elif family == 'vlanext_s':
        sample['normalized_action'] = quantile(canonical, stats['actions'])


def attach_seer_targets(sample, values, frame, stats):
    from .data import seer_labels
    local, previous, measured = seer_labels(values)
    n = len(local)
    history = np.maximum(np.arange(frame-6, frame+1), 0)
    future_actions = history[:, None]+np.arange(3)
    future_images = history+3
    sample.update(local_actions=local[np.minimum(future_actions, n-1)],
                  valid_actions=future_actions < n,
                  future_front_labels=values['front_rgb'][np.minimum(future_images, n-1)],
                  future_wrist_labels=values['wrist_rgb'][np.minimum(future_images, n-1)],
                  valid_future_images=future_images < n,
                  current_pose=measured[frame], previous_command=previous[frame])
