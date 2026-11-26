"""One causal observation path shared by offline data and online execution."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from .state21_reference import normalize, model_state, pi0_state_text
from .specs import SPECS


def validate_raw(value):
    x = np.asarray(value, np.float32)
    if x.shape != (21,) or not np.isfinite(x).all():
        raise ValueError('raw_state21 must contain exactly 21 finite channels; old67/74 are not accepted')
    if not 0 <= x[6] <= 1 or not -1 <= x[10] <= 1:
        raise ValueError('Invalid measured closure or preceding signed-close command')
    if x[17] < 0 or x[20] < 0 or not np.isin(x[18:20], (0, 1)).all():
        raise ValueError('Invalid aperture or generic pad feedback')
    R = rotation(x)
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-5) or np.linalg.det(R) < .9999:
        raise ValueError('Malformed measured tool rotation')
    return x.copy()


def rotation(x):
    a, b = np.asarray(x)[11:14], np.asarray(x)[14:17]
    return np.column_stack((a, b, np.cross(a, b)))


@dataclass(frozen=True)
class Frame:
    raw_state21: np.ndarray
    ee_position: np.ndarray
    front: np.ndarray
    wrist: np.ndarray
    instruction: str
    time_s: float
    depth_z_m: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    extrinsics_OpenCV: np.ndarray | None = None

    def checked(self):
        raw = validate_raw(self.raw_state21)
        p = np.asarray(self.ee_position, np.float32)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Measured tool-link XYZ3 required for native prefixes')
        images = []
        for value in (self.front, self.wrist):
            value = np.asarray(value)
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError('Unnormalized uint8 HWC front/wrist RGB required')
            images.append(value.copy())
        if not self.instruction.strip() or 'simulation observation before action:' in self.instruction.lower():
            raise ValueError('Task instruction only; legacy numeric scene text is rejected')
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError('Finite causal simulation timestamp required')
        optional = (self.depth_z_m, self.intrinsics, self.extrinsics_OpenCV)
        if any(v is not None for v in optional):
            d, K, E = map(lambda v: np.asarray(v, np.float32), optional)
            if (d.shape != images[0].shape[:2] or K.shape != (3, 3) or E.shape != (4, 4)
                    or not all(np.isfinite(v).all() for v in (d, K, E)) or np.any(d <= 0)
                    or K[0, 0] <= 0 or K[1, 1] <= 0
                    or not np.allclose(E[3], [0, 0, 0, 1])):
                raise ValueError('Synchronized positive metric front depth, K3x3 and world-to-camera4x4 required')
            optional = tuple(v.copy() for v in (d, K, E))
        return Frame(raw, p.copy(), *images, self.instruction.strip(), float(self.time_s), *optional)


def prefix(frame, family):
    from VLA.new_vlas.action_contracts import native8, pack_native20, normalize_bounds
    x, pos = frame.raw_state21, frame.ee_position
    if family == 'xvla':
        return pack_native20(pos, x[11:17], np.asarray(x[6]))
    if family in ('vla_adapter', 'dita', 'vlanext_s'):
        native = native8(pos, x[11:17], x[17])
        return native if family == 'vla_adapter' else np.r_[native[:6], x[6]].astype(np.float32)
    if family == 'seer':
        return np.r_[pos, Rotation.from_matrix(rotation(x)).as_euler('xyz'), float(x[6] < .5)].astype(np.float32)
    return None


class Inputs:
    """Call push at every 10 Hz control tick, including ticks without a new policy query."""
    def __init__(self, family, stats):
        self.family, self.stats = family, stats
        self.spec = SPECS[family]
        self.frames = deque(maxlen=self.spec.history)
        # Reject old normalizers immediately, including all-binary-only fixtures.
        record = stats['robot21']
        if (record.get('schema') != 'vla21-v1' or np.asarray(record['mean']).shape != (21,)
                or not np.array_equal(np.asarray(record['mean'])[18:20], [0, 0])
                or not np.array_equal(np.asarray(record['std'])[18:20], [1, 1])):
            raise ValueError('TRAIN-derived robot21 statistics with unchanged contact bits required')

    def reset(self):
        self.frames.clear()

    def push(self, frame):
        frame = frame.checked()
        if self.frames:
            if not np.isclose(frame.time_s - self.frames[-1].time_s, .1, atol=1e-5):
                raise ValueError('History must contain contiguous 10 Hz observations, not query-only history')
            if frame.instruction != self.frames[-1].instruction:
                raise ValueError('Reset history between instructions/episodes')
        self.frames.append(frame)

    def sample(self):
        if not self.frames:
            raise ValueError('Push the reset observation before querying')
        history = [self.frames[0]] * (self.spec.history - len(self.frames)) + list(self.frames)
        frame = history[-1]
        family, stats = self.family, self.stats
        native = prefix(frame, family)
        if family == 'vla_adapter':
            from VLA.new_vlas.action_contracts import normalize_bounds
            native = normalize_bounds(native, stats['adapter']['proprio'])
        state = model_state(frame.raw_state21, stats['robot21'], family, native_prefix=native)
        sample = {'front': frame.front, 'wrist': frame.wrist, 'instruction': frame.instruction,
                  'raw_state21': frame.raw_state21.copy(), 'state': state,
                  'ee_position': frame.ee_position.copy(), 'time_s': frame.time_s,
                  'history_times': [v.time_s for v in history]}
        if family == 'pi0':
            sample['prompt'] = frame.instruction + '; robot state: ' + pi0_state_text(frame.raw_state21)
        if family in ('dita', 'vlanext_s'):
            sample['history28'] = np.stack([model_state(v.raw_state21, stats['robot21'], family,
                                             native_prefix=prefix(v, family)) for v in history])
        if family == 'dita':
            sample['history_images'] = {key: [getattr(v, key) for v in history] for key in ('front', 'wrist')}
        if family == 'seer':
            sample.update(state_history=np.stack([prefix(v, family) for v in history]),
                          robot21_history=np.stack([normalize(v.raw_state21, stats['robot21']) for v in history]),
                          front_history=np.stack([v.front for v in history]),
                          wrist_history=np.stack([v.wrist for v in history]))
        if self.spec.depth:
            if frame.depth_z_m is None:
                raise ValueError('FALCON-ESM requires actual aligned front depth and calibration')
            for key in ('depth_z_m', 'intrinsics', 'extrinsics_OpenCV'):
                sample[key] = getattr(frame, key).copy()
        return sample
