"""Read-only complete-source native dataset, task/target-balanced sampling."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import normalize_bounds
from VLA.new_vlas.privileged_inputs import append_numeric67


class NativeData:
    def __init__(self, root, *, verify_rgb_hashes=True):
        self.root = Path(root).resolve()
        self.proof = json.loads((self.root/'audit.json').read_text())
        if (not self.proof.get('passed') or not self.proof.get('original_full_source_coverage')
                or not self.proof.get('all_frames_actual_URDF_FK_checked')
                or self.proof.get('future_labels_in_observations') is not False):
            raise ValueError('Complete-source native action qualification required')
        for name in ('index', 'stats'):
            if sha256_file(self.root/(name+'.json')) != self.proof[name+'_sha256']:
                raise ValueError('Native index or TRAIN statistics changed')
        for path, digest in self.proof['bound_files'].items():
            if sha256_file(path) != digest:
                raise ValueError('Action conversion/URDF source changed: '+path)
        self.rows = json.loads((self.root/'index.json').read_text())
        self.stats = json.loads((self.root/'stats.json').read_text())
        self.groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, row in enumerate(self.rows):
            if sha256_file(self.root/row['cache']) != row['cache_sha256']:
                raise ValueError('Native numeric cache changed')
            if verify_rgb_hashes and sha256_file(row['source']) != row['source_sha256']:
                raise ValueError('Original RGB/action source changed')
            self.groups[row['task']][row['source_kind']][row['pick_name']].append(index)
        if sum(r['frames'] for r in self.rows) != self.proof['frames'] or len(self.groups) != 4:
            raise ValueError('Incomplete original4-task dataset')

    @lru_cache(maxsize=4)
    def numeric(self, episode):
        with np.load(self.root/self.rows[episode]['cache'], allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    @lru_cache(maxsize=2)
    def rgb(self, episode):
        with np.load(self.rows[episode]['source'], allow_pickle=False) as data:
            return {key: data[key] for key in ('front_rgb', 'wrist_rgb')}

    def draw(self, rng):
        # Equal task probability,20% corrective rehearsal within each task,
        # uniform requested object then episode. No held-out feedback used.
        task = rng.choice(sorted(self.groups))
        kind = 'fresh_corrective' if rng.random() < .20 else 'complete_original'
        choices = self.groups[task][kind]
        target = rng.choice(sorted(choices))
        ei = int(rng.choice(choices[target]))
        canonical = self.numeric(ei)['canonical7']
        if rng.random() < .10:
            fi = 0
        else:
            weights = 1.+2*(np.linalg.norm(canonical[:, :3], axis=1) > .001)
            fi = int(rng.choice(len(canonical), p=weights/weights.sum()))
        return ei, fi

    def sample(self, episode, frame, family):
        row, data, rgb = self.rows[episode], self.numeric(episode), self.rgb(episode)
        horizon = 30 if family == 'xvla' else 8
        ids = np.minimum(frame+np.arange(horizon), row['frames']-1)
        if family == 'xvla':
            prefix = data['state20'][frame]
            actions = data['action20'][ids]
        elif family == 'vla_adapter':
            prefix = normalize_bounds(data['native8'][frame], self.stats['adapter']['proprio'])
            actions = normalize_bounds(data['adapter_action7'][ids], self.stats['adapter']['action'])
        else:
            raise ValueError('Unknown native family')
        state = append_numeric67(prefix, data['scene64'][frame], self.stats['scene67'])
        return {'state': state, 'action': actions, 'instruction': row['instruction'],
            'front': rgb['front_rgb'][frame].copy(), 'wrist': rgb['wrist_rgb'][frame].copy(),
            'episode': episode, 'frame': frame, 'task': row['task']}
