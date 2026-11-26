"""Aligned episode bank for robot21; old task-conditioned caches are rejected."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np

from .inputs import Frame, Inputs, prefix
from .state21_reference import fit_train_stats
from .specs import SPECS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class Bank:
    """Manifest has schema, split and rows: episode_id, scene_id, file, sha256,
    instruction, task, source_kind, target_name (sampling label only).
    Payload arrays: raw_state21, ee_position, front_rgb, wrist_rgb, canonical7,
    time_s; FALCON also depth_z_m, intrinsics, extrinsics_OpenCV.
    """
    def __init__(self, manifest, *, reserved_scenes=()):
        self.path = Path(manifest).resolve()
        record = json.loads(self.path.read_text())
        if record.get('schema') != 'vla21-episodes-v1' or record.get('split') not in ('TRAIN', 'VAL', 'TEST'):
            raise ValueError('Explicit robot21 schema and split required; no legacy scene64 fallback')
        self.split, self.rows = record['split'], record['rows']
        if not self.rows:
            raise ValueError('Empty episode bank')
        ids = [r['episode_id'] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate episodes would bias statistics')
        reserved = set(reserved_scenes)
        for row in self.rows:
            if row['scene_id'] in reserved:
                raise ValueError('Scene-group overlap with reserved validation/evaluation instances')
            if not row.get('sha256') or digest(self.path.parent / row['file']) != row['sha256']:
                raise ValueError('Missing or changed aligned episode payload')

    @lru_cache(maxsize=2)
    def arrays(self, episode):
        row = self.rows[episode]
        with np.load(self.path.parent / row['file'], allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        required = {'raw_state21', 'ee_position', 'front_rgb', 'wrist_rgb', 'canonical7', 'time_s'}
        if not required <= values.keys() or {'scene64', 'numeric67', 'map'} & values.keys():
            raise ValueError('Robot21 payload required, without legacy scene conditioning')
        n = len(values['time_s'])
        if n < 1 or values['raw_state21'].shape != (n, 21) or values['canonical7'].shape != (n, 7):
            raise ValueError('Invalid aligned state/action shapes')
        if (values['ee_position'].shape != (n, 3) or not np.isfinite(values['canonical7']).all()
                or not np.allclose(np.diff(values['time_s']), .1, atol=1e-5)
                or not np.isclose(values['time_s'][0], 0)):
            raise ValueError('Episode timestamps/EEF/action alignment differs')
        if not np.isclose(values['raw_state21'][0, 10], -1):
            raise ValueError('Reset previous gripper command must be -1')
        # Records store the previous applied actuator command, not the future label.
        if n > 1:
            command=values['canonical7'][:-1,6]
            # Match the shared Robotiq actuator's signed command/hysteresis.
            applied=np.where(command>.2,1.,np.where(command<-.2,-1.,values['raw_state21'][:-1,10]))
            if not np.allclose(values['raw_state21'][1:,10],applied,atol=1e-5):
                raise ValueError('Preceding applied gripper command is not causally aligned')
        for i in range(n):
            self.frame_from(values, row, i).checked()
        return values

    @staticmethod
    def frame_from(v, row, i):
        extra = {}
        for key in ('depth_z_m', 'intrinsics', 'extrinsics_OpenCV'):
            if key in v:
                extra[key] = v[key][i]
        return Frame(v['raw_state21'][i], v['ee_position'][i], v['front_rgb'][i],
                     v['wrist_rgb'][i], row['instruction'], float(v['time_s'][i]), **extra)

    def statistics(self, *, parent_robot21=None):
        if self.split != 'TRAIN':
            raise ValueError('Never fit normalization on VAL/TEST')
        from VLA.new_vlas.action_contracts import bounds_stats
        # Keep only small numerical arrays; never retain the complete RGB bank in RAM.
        values = [{k:self.arrays(ei)[k] for k in ('raw_state21','ee_position','canonical7')}
                  for ei in range(len(self.rows))]
        raw = np.concatenate([v['raw_state21'] for v in values])
        stats = {'robot21': parent_robot21 if parent_robot21 is not None else fit_train_stats(raw)}
        actions = np.concatenate([v['canonical7'] for v in values]).astype(np.float64)
        mean, std = actions.mean(0), np.maximum(actions.std(0), [.005]*6+[1.])
        mean[6], std[6] = 0., 1.
        lo, hi = np.quantile(actions, (.01, .99), axis=0)
        radius = np.maximum((hi-lo)/2, [.03]*6+[1.])
        center = (hi+lo)/2
        lo, hi = center-radius, center+radius
        lo[6], hi[6] = -1., 1.
        stats['actions'] = {k: v.tolist() for k, v in [('mean', mean), ('std', std), ('q01', lo), ('q99', hi)]}
        from VLA.new_vlas.action_contracts import native8
        states = [native8(v['ee_position'], v['raw_state21'][:,11:17], v['raw_state21'][:,17]) for v in values]
        adapter_action = actions.copy(); adapter_action[:, 6] = (1-adapter_action[:, 6])/2
        stats['adapter'] = {'proprio': bounds_stats(np.concatenate(states)),
                            'action': bounds_stats(adapter_action, mask=[True]*6+[False])}
        stats['action'] = dict(stats['actions'])  # JEPA canonical signed-close moments.
        stats['manifest_sha256'] = digest(self.path)
        stats['fit_split'] = 'TRAIN'
        stats['frames'] = len(raw)
        # Seer must use previous requested poses, not measured-pose local labels.
        maximum = np.zeros(6)
        for v in values:
            local, _, _ = seer_labels(v)
            maximum = np.maximum(maximum, np.abs(local[:, :6]).max(0))
        stats['local_action_scale6'] = [max(.02, float(maximum[:3].max())*1.05)]*3 + [max(.05, float(maximum[3:].max())*1.05)]*3
        return stats

    def sample(self, family, episode, frame, stats):
        values, row = self.arrays(episode), self.rows[episode]
        n = len(values['time_s']); spec = SPECS[family]
        if not 0 <= frame < n:
            raise IndexError(frame)
        inputs = Inputs(family, stats)
        for t in range(max(0, frame-spec.history+1), frame+1):
            inputs.push(self.frame_from(values, row, t))
        sample = inputs.sample()
        ids = frame + np.arange(spec.horizon)
        action = values['canonical7'][np.minimum(ids, n-1)].copy()
        action[ids >= n, :6] = 0
        sample.update(action=action, action_is_pad=ids >= n)
        from .targets import attach_targets
        attach_targets(family, sample, stats)
        if family == 'xvla':
            from VLA.new_vlas.action_contracts import absolute20_from_canonical
            at = np.minimum(ids, n-1)
            sample['action'] = absolute20_from_canonical(values['ee_position'][at],
                values['raw_state21'][at, 11:17], action)
        if family == 'dita':
            # Native denoiser includes one history slot plus ten future actions.
            indices = frame-1 + np.arange(11)
            valid = (indices >= 0) & (indices < n)
            labels = values['canonical7'][np.clip(indices, 0, n-1)].copy()
            labels[~valid, :6] = 0
            from .targets import quantile
            labels = quantile(labels, stats['actions']); labels[:, 6] = (1-labels[:, 6])/2
            sample.update(normalized_action11=labels, action_valid11=valid)
        if family == 'seer':
            from .targets import attach_seer_targets
            attach_seer_targets(sample, values, frame, stats)
        return sample


def seer_labels(values):
    from .inputs import rotation
    from VLA.new_vlas.action_contracts import absolute20_from_canonical
    from VLA.admission_r100.seer_data import poses
    from scipy.spatial.transform import Rotation
    raw = values['raw_state21']
    desired = absolute20_from_canonical(values['ee_position'], raw[:, 11:17], values['canonical7'])
    target = poses(desired)
    measured = np.broadcast_to(np.eye(4), target.shape).copy()
    measured[:, :3, 3] = values['ee_position']
    measured[:, :3, :3] = np.stack([rotation(x) for x in raw])
    previous = np.concatenate((measured[:1], target[:-1]))
    inv = previous[:, :3, :3].transpose(0, 2, 1)
    local = np.c_[np.einsum('nij,nj->ni', inv, target[:, :3, 3]-previous[:, :3, 3]),
                  Rotation.from_matrix(inv @ target[:, :3, :3]).as_euler('xyz'), -values['canonical7'][:, 6]]
    return local.astype(np.float32), previous, measured
