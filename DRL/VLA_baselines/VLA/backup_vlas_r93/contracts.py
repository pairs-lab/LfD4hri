"""Representation-only candidate contracts. These do not implement a VLA policy."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose(position, rotation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, np.float64)
    result[:3, 3] = np.asarray(position, np.float64)
    return result


def seer_local_from_canonical(action, current_pose, previous_command=None):
    """Convert world-translation/local-Euler signed-CLOSE to Seer local signed-OPEN.

    The anchor is the previous requested pose, as in native real deployment,
    or the measured initial pose at the first step. No goal/object state is
    used to choose an action. Bounds/normalization remain separate contracts.
    """
    action = np.asarray(action, np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Finite canonical action7 required")
    current_pose = np.asarray(current_pose, np.float64)
    target = current_pose.copy()
    target[:3, 3] += action[:3]
    target[:3, :3] = current_pose[:3, :3] @ Rotation.from_euler("xyz", action[3:6]).as_matrix()
    anchor = current_pose if previous_command is None else np.asarray(previous_command, np.float64)
    relative = np.linalg.solve(anchor, target)
    return np.r_[relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_euler("xyz"), -action[6]]


def canonical_from_seer_local(action, current_pose, previous_command=None):
    action = np.asarray(action, np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Finite unnormalized native Seer action7 required")
    current_pose = np.asarray(current_pose, np.float64)
    anchor = current_pose if previous_command is None else np.asarray(previous_command, np.float64)
    target = anchor @ pose(action[:3], Rotation.from_euler("xyz", action[3:6]).as_matrix())
    relative_rotation = current_pose[:3, :3].T @ target[:3, :3]
    return np.r_[target[:3, 3] - current_pose[:3, 3],
                 Rotation.from_matrix(relative_rotation).as_euler("xyz"), -action[6]]


def pad_state(values, width):
    values = np.asarray(values, np.float32)
    if not np.isfinite(values).all() or values.shape[-1] > width:
        raise ValueError("State would be truncated or is nonfinite")
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, width - values.shape[-1])])


def temporal_indices(frame, frames, history, horizon):
    if not 0 <= frame < frames or history < 1 or horizon < 1:
        raise ValueError("Invalid within-trajectory temporal contract")
    past = np.maximum(np.arange(frame - history + 1, frame + 1), 0)
    future = np.arange(frame, frame + horizon)
    return past, np.minimum(future, frames - 1), future < frames
