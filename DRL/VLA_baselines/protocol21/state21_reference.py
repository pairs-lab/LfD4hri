"""Reference for the locked sensor/FK observation contract, not a VLA runner."""
import numpy as np

NAMES = (
    *('q' + str(i) + '_rad' for i in range(1, 7)),
    'closure01', 'pad_midpoint_world_x_m', 'pad_midpoint_world_y_m',
    'pad_midpoint_world_z_m', 'previous_signed_close',
    'tool_R11', 'tool_R21', 'tool_R31', 'tool_R12', 'tool_R22', 'tool_R32',
    'surface_jaw_gap_m', 'left_external_contact', 'right_external_contact',
    'total_pad_normal_force_N',
)
BINARY = (18, 19)


def pack(*, q, closure01, pad_midpoint_world, previous_signed_close,
         tool_to_world_rotation, surface_jaw_gap_m,
         left_external_normal_force_N, right_external_normal_force_N):
    """Forces aggregate all non-self contacts at each pad, without body labels."""
    q = np.asarray(q, np.float64)
    p = np.asarray(pad_midpoint_world, np.float64)
    R = np.asarray(tool_to_world_rotation, np.float64)
    if q.shape != (6,) or p.shape != (3,) or R.shape != (3, 3):
        raise ValueError('Expected q6, world XYZ3 and tool-to-world rotation3x3')
    if not np.isfinite(np.r_[q, p, R.ravel()]).all():
        raise ValueError('Non-finite robot observation')
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
        raise ValueError('Invalid tool rotation')
    if not (0 <= closure01 <= 1 and -1 <= previous_signed_close <= 1):
        raise ValueError('Invalid closure/previous-command convention')
    forces = np.asarray([left_external_normal_force_N, right_external_normal_force_N], np.float64)
    if not np.isfinite(forces).all() or np.any(forces < 0) or surface_jaw_gap_m < 0:
        raise ValueError('Invalid contact force/aperture')
    result = np.r_[q, closure01, p, previous_signed_close, R[:, :2].T.ravel(),
                   surface_jaw_gap_m, (forces > .1).astype(np.float64), forces.sum()]
    if result.shape != (21,) or not np.isfinite(result).all():
        raise ValueError('Exactly21 finite causal values required')
    return result.astype(np.float32)


def fit_train_stats(raw_train):
    """Fit once per training bank before any oversampling; population std."""
    x = np.asarray(raw_train, np.float64)
    if x.ndim != 2 or x.shape[1] != 21 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty TRAIN Nx21 required')
    if not np.isin(x[:, BINARY], (0, 1)).all():
        raise ValueError('Contact channels must be binary')
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0, ddof=0), 1e-6)
    mean[list(BINARY)], std[list(BINARY)] = 0., 1.
    return {'mean': mean.tolist(), 'std': std.tolist(), 'count': len(x),
            'fit_scope': 'TRAIN only; unweighted unique aligned timesteps',
            'schema': 'vla21-v1', 'clip': None}


def normalize(raw, stats):
    x = np.asarray(raw, np.float32)
    mean, std = (np.asarray(stats[k], np.float32) for k in ('mean', 'std'))
    if x.shape[-1] != 21 or mean.shape != (21,) or std.shape != (21,):
        raise ValueError('Do not reuse old-width normalizers')
    if not np.isfinite(x).all() or not np.isfinite(np.r_[mean, std]).all() or np.any(std <= 0):
        raise ValueError('Invalid finite normalization record')
    if not np.isin(x[..., BINARY], (0, 1)).all():
        raise ValueError('Contact channels must remain binary')
    return ((x - mean) / std).astype(np.float32)


def model_state(raw, stats, family, *, native_prefix=None):
    """Native prefix is already in the separately declared native convention."""
    tail = normalize(raw, stats)
    if tail.ndim != 1:
        raise ValueError('Pack one timestamp; history is handled by the caller')
    widths = {'xvla': 20, 'vla_adapter': 8, 'dita': 7, 'vlanext_s': 7, 'seer': 7}
    if family == 'pi0':
        return np.r_[tail, np.zeros(11, np.float32)]
    if family in ('flower', 'smolvla', 'vla_jepa', 'falcon_esm'):
        return tail
    if family not in widths:
        raise ValueError('Unknown method')
    prefix = np.asarray(native_prefix, np.float32)
    if prefix.shape != (widths[family],) or not np.isfinite(prefix).all():
        raise ValueError('Missing/malformed native prefix')
    if family == 'seer':
        return {'native7': prefix.copy(), 'robot21': tail}
    return np.r_[prefix, tail]


def pi0_state_text(raw):
    x = np.asarray(raw, np.float64)
    if x.shape != (21,) or not np.isfinite(x).all():
        raise ValueError('Finite raw21 required')
    return '; '.join(name + '=' + format(float(value), '.4f') for name, value in zip(NAMES, x))
