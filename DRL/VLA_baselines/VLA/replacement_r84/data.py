"""Causal native7 + appended67; real history8; zero-motion terminal padding.

No held-out observations, action labels, future flags or expert phases enter
policy inputs. Native LIBERO camera axes are NOT imposed on our PyBullet scene.
"""
import numpy as np
from VLA.flower_rotation_r76.sampler import Data
from VLA.family_transfer_r54.data import action_stats
from VLA.stable_native_r44.conditioning import bounded_append

def statistics(data):
    return {'actions':action_stats(data), 'scene67':data.stats['scene67'],
        'native_prefix':'measured EEF xyz/continuous axisangle/encoder closure',
        'numeric_dim':74, 'history':8, 'chunk':8}

def normalize_actions(a, stats):
    a = np.asarray(a, np.float32).copy()
    lo, hi = (np.asarray(stats['actions'][k], np.float32) for k in ('q01','q99'))
    a[..., :6] = np.clip(2*(a[..., :6]-lo[:6])/(hi[:6]-lo[:6])-1, -1, 1)
    return a

def denormalize_actions(a, stats):
    a = np.asarray(a, np.float32).copy()
    lo, hi = (np.asarray(stats['actions'][k], np.float32) for k in ('q01','q99'))
    a[..., :6] = (a[..., :6]+1)*.5*(hi[:6]-lo[:6])+lo[:6]
    # Native signed-close output, only a declaration of the common actuator convention.
    a[..., 6] = np.clip(a[..., 6], -1, 1)
    return a

def observation74(prefix6, raw, stats):
    return bounded_append(np.r_[prefix6, raw[6]].astype(np.float32), raw, stats['scene67'])

def sample(data, ei, fi, stats):
    data.horizon = 8
    s = data.at(ei, fi)
    n = data.numeric(ei)
    # Same reset repeat padding as native LeRobot history, includes current t.
    history = np.maximum(np.arange(fi-7, fi+1), 0)
    s['history74'] = np.stack([observation74(n['native8'][t,:6], n['scene64'][t], stats) for t in history])
    a = s['action'].copy()
    a[s['action_is_pad'], :6] = 0  # native padding, not repeating the final nonzero velocity
    s['normalized_action'] = normalize_actions(a, stats)
    s['history_source_indices'] = history
    return s
