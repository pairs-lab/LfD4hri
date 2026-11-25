"""Representation conversions only. No target planner or simulator controller.

X-VLA uses native absolute20 (single physical arm plus fixed dummy second arm).
Adapter uses the original 7D delta-EEF action, with its native [0,1] OPEN channel.
Future labels are never included in observations.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from config import BOUNDS


def rotation_matrix(columns):
    columns = np.asarray(columns, np.float64)
    if columns.shape[-1] != 6 or not np.isfinite(columns).all():
        raise ValueError('Finite first-two-column rotation6D required')
    first, second = columns[..., :3], columns[..., 3:]
    # Source rotations are already orthonormal; model predictions need the
    # standard differentiable-representation Gram-Schmidt decoding instead.
    first = first/np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second-first*np.sum(first*second, axis=-1, keepdims=True)
    second = second/np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
    if np.any(np.abs(np.linalg.det(matrix)-1) > 1e-4):
        raise ValueError('Degenerate rotation prediction cannot define an action')
    return matrix


def pack_native20(position, columns, closure):
    position, columns, closure = map(np.asarray, (position, columns, closure))
    # UR foundation domain12 uses R[:,:2].reshape(6), not the repository's
    # R[:,:2].T.reshape(6). LIBERO has a DIFFERENT domain-specific converter.
    # Match the selected actual pretrained UR handler, not an internal-only
    # roundtrip that could pass while losing its pretrained orientation prior.
    interleaved = columns[..., [0, 3, 1, 4, 2, 5]]
    pose = np.concatenate((position, interleaved, closure[..., None]), axis=-1)
    dummy = np.zeros_like(pose)
    result = np.concatenate((pose, dummy), axis=-1).astype(np.float32)
    if result.shape[-1] != 20 or not np.isfinite(result).all():
        raise ValueError('Invalid native absolute20 layout')
    return result


def absolute20_from_canonical(position, columns, canonical):
    """Exact desired pose supplied to the SAME shared environment.step()."""
    action = np.asarray(canonical, np.float64)
    if action.shape[-1] != 7 or not np.isfinite(action).all():
        raise ValueError('Finite canonical7 action required')
    target_position = np.clip(np.asarray(position)+np.clip(action[..., :3], -.05, .05),
        (BOUNDS[0, 0]-.12, BOUNDS[1, 0]-.05, .10),
        (BOUNDS[0, 1]+.12, BOUNDS[1, 1]+.05, .72))
    current = Rotation.from_matrix(rotation_matrix(columns))
    target = (current*Rotation.from_euler('xyz', np.clip(action[..., 3:6], -.25, .25))).as_matrix()
    six = np.swapaxes(target[..., :, :2], -1, -2).reshape(*target_position.shape[:-1], 6)
    return pack_native20(target_position, six, np.clip((action[..., 6]+1)/2, 0, 1))


def canonical_from_absolute20(native, current_position, current_columns):
    native = np.asarray(native, np.float64)
    if native.shape[-1] != 20 or not np.isfinite(native).all():
        raise ValueError('Finite native absolute20 prediction required')
    source_columns = native[..., 3:9][..., [0, 2, 4, 1, 3, 5]]
    relative = (Rotation.from_matrix(rotation_matrix(current_columns)).inv()
                *Rotation.from_matrix(rotation_matrix(source_columns))).as_euler('xyz')
    return np.concatenate((native[..., :3]-np.asarray(current_position), relative,
        (2*np.clip(native[..., 9], 0, 1)-1)[..., None]), axis=-1).astype(np.float32)


def native8(position, columns, jaw_gap):
    """PyBullet EEF xyz/axis-angle and signed symmetric half jaw aperture.

    Last two are an explicitly declared Robotiq aperture representation, not
    falsely described as real Franka joint encoders. Same formula at runtime.
    """
    rotvec = Rotation.from_matrix(rotation_matrix(columns)).as_rotvec()
    # UR home is close to pi around world-Y. SciPy's principal rotvec flips
    # sign at pi, causing artificial ~2pi state jumps despite smooth motion.
    # Choose the equivalent branch closest to fixed home, no future/history
    # or target-dependent information. The actual rotation is unchanged.
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    alternate = rotvec-2*np.pi*rotvec/np.maximum(angle, 1e-12)
    home = np.array([0., np.pi, 0.])
    choose = np.linalg.norm(alternate-home, axis=-1) < np.linalg.norm(rotvec-home, axis=-1)
    rotvec = np.where(choose[..., None], alternate, rotvec)
    gap = np.asarray(jaw_gap)
    return np.concatenate((position, rotvec, gap[..., None]/2, -gap[..., None]/2), axis=-1).astype(np.float32)


def bounds_stats(values, *, mask=None):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite TRAIN matrix required')
    low, high = np.quantile(values, (.01, .99), axis=0)
    center = (low+high)/2
    radius = np.maximum((high-low)/2, 1e-4)
    low, high = center-radius, center+radius
    return {'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(),
        'min': values.min(0).tolist(), 'max': values.max(0).tolist(),
        'q01': low.tolist(), 'q99': high.tolist(),
        'mask': list(mask) if mask is not None else [True]*values.shape[1],
        'quantile_span_floor': .0002,
        'fraction_below_q01': (values < low).mean(0).tolist(),
        'fraction_above_q99': (values > high).mean(0).tolist()}


def normalize_bounds(values, stats):
    values = np.asarray(values, np.float32)
    low, high = (np.asarray(stats[k], np.float32) for k in ('q01', 'q99'))
    mask = np.asarray(stats['mask'], bool)
    return np.clip(np.where(mask, 2*(values-low)/(high-low+1e-8)-1, values), -1, 1).astype(np.float32)
