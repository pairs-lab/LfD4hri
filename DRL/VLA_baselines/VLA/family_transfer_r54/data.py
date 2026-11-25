"""Same complete qualified TRAIN bank, causal observations, native labels.

No source RGB duplication, held-out frames, teacher phases or target waypoints.
"""
from bisect import bisect_right
from functools import lru_cache
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.new_vlas.dataset import NativeData
from VLA.fit_feedback_r52 import draw, gate_anchors, SAMPLER
from VLA.scene64_conditioning import encode_numeric, augment_prompt
from VLA.privileged_visual_r53.scene_map import render
from .common import DATA


class Data(NativeData):
    def __init__(self, root=DATA, *, verify_rgb_hashes=False, horizon=5):
        super().__init__(root, verify_rgb_hashes=verify_rgb_hashes)
        self.horizon = int(horizon)
        if self.horizon < 1: raise ValueError('Positive native horizon required')
        self.ends = np.cumsum([r['frames'] for r in self.rows]).tolist()
        if len(self.rows) != 2185 or self.ends[-1] != 119495:
            raise ValueError('Require the complete qualified R52 bank, not a small pilot')

    draw = draw

    def __len__(self): return self.ends[-1]

    def address(self, index):
        if not 0 <= int(index) < len(self): raise IndexError(index)
        ei = bisect_right(self.ends, int(index))
        return ei, int(index)-(self.ends[ei-1] if ei else 0)

    @lru_cache(maxsize=2)
    def rgb(self, episode):
        row = self.rows[episode]
        if sha256_file(row['source']) != row['source_sha256']:
            raise ValueError('Actual sampled RGB/label source changed')
        return super().rgb(episode)

    def at(self, ei, fi):
        row, number, rgb = self.rows[ei], self.numeric(ei), self.rgb(ei)
        raw = number['scene64'][fi].copy()
        ids = fi+np.arange(self.horizon)
        actions = number['canonical7'][np.minimum(ids, row['frames']-1)].copy()
        return {'front': rgb['front_rgb'][fi].copy(), 'wrist': rgb['wrist_rgb'][fi].copy(),
            'map': render(raw, row['task']), 'raw64': raw, 'numeric67': encode_numeric(raw),
            'action': actions, 'action_is_pad': ids >= row['frames'],
            'instruction': row['instruction'], 'prompt': augment_prompt(row['instruction'], raw),
            'task': row['task'], 'episode': int(ei), 'frame': int(fi)}

    def __getitem__(self, index): return self.at(*self.address(index))

    def __getitems__(self, indices): return [self[int(i)] for i in indices]


def action_stats(data, *, closure01=False):
    a = np.concatenate([data.numeric(i)['canonical7'] for i in range(len(data.rows))]).astype(np.float64)
    if closure01: a[:, 6] = (a[:, 6]+1)/2
    mean, std = a.mean(0), a.std(0)
    std = np.maximum(std, [.005]*6+[.5 if closure01 else 1.])
    mean[6] = .5 if closure01 else 0.
    low, high = np.quantile(a, .01, axis=0), np.quantile(a, .99, axis=0)
    for i in range(6):
        radius = max((high[i]-low[i])/2, .03)
        center = (high[i]+low[i])/2
        low[i], high[i] = center-radius, center+radius
    low[6], high[6] = (0., 1.) if closure01 else (-1., 1.)
    return {k: np.asarray(v, np.float32).tolist() for k,v in
        [('mean',mean), ('std',std), ('q01',low), ('q99',high)]}


def anchors(data):
    return gate_anchors(data)


def phase_anchors(data):
    result=[];events={}
    for task in ('touch','pick','move','pick_place'):
        for kind in ('complete_original','wrist_tilt','closed_empty'):
            ei=next(i for i,r in enumerate(data.rows) if r['task']==task and
                (r['source_kind']=='complete_original' if kind=='complete_original' else r.get('recovery_kind')==kind))
            n=data.numeric(ei);raw,action=n['scene64'],n['canonical7']
            chosen={'initial_pre_action':0,'terminal_pre_action':len(raw)-1}
            for name,flag in {'lateral_label_audit_only':np.linalg.norm(action[:,:2],axis=1)>.005,
                'closed_empty':(raw[:,6]>.8)&(raw[:,31]<.5),'pad_contact':(raw[:,28:30]>.5).any(axis=1),
                'correct_bilateral_hold':raw[:,17]>.5}.items():
                ids=np.flatnonzero(flag)
                if len(ids):chosen[name]=int(ids[0])
            for name,fi in chosen.items():
                if (ei,fi) not in events:result.append((ei,fi));events[ei,fi]=[]
                events[ei,fi].append(name)
    return result,events
