"""Qualified new TRAIN examples; original task/target probabilities unchanged."""
from collections import Counter
import os
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.replacement_r84.data import Data as OriginalData
from .audit import OUT, read


class Data(OriginalData):
    def __init__(self):
        super().__init__()
        root = Path(os.environ.get('VLA_R90_RECOVERY_DATA', str(OUT/'qualified_recovery'))).resolve()
        proof, rows = read(root/'ADMISSION.json'), read(root/'index.json')
        if not proof['passed'] or proof['split'] != 'train' or not proof['pre_action_no_future_labels']:
            raise ValueError('Qualified new TRAIN recovery receipt required')
        if proof['minimum_targets_per_task_kind'] < 10:
            raise ValueError('A two-target pilot is NOT sufficient for production adaptation')
        if sha256_file(root/'index.json') != proof['new_index_sha256']:
            raise ValueError('Recovery index changed')
        if sha256_file(self.root/'index.json') != proof['original_index_sha256'] or sha256_file(self.root/'stats.json') != proof['stats_sha256_preserved']:
            raise ValueError('Original bank/statistics changed')
        for path, digest in proof['binding'].items():
            if sha256_file(path) != digest: raise ValueError('Recovery qualification changed')
        self.corrective = {}
        for row in rows:
            if sha256_file(row['cache']) != row['cache_sha256']:
                raise ValueError('Recovery numeric cache changed')
            ei = len(self.rows); self.rows.append(row)
            self.corrective.setdefault((row['task'], row['pick_name']), []).append(ei)
        self.ends = np.cumsum([r['frames'] for r in self.rows]).tolist()
        self.exposure = Counter()
        type(self).last_instance = self

    def draw(self, rng):
        # First draw from the original task/target-balanced sampler. Replacing
        # within THAT task+target retains the original target prior exactly.
        ei, fi = super().draw(rng)
        row = self.rows[ei]
        options = self.corrective.get((row['task'], row['pick_name']), ())
        if options and rng.random() < .35:
            ei = int(rng.choice(options))
            n = self.rows[ei]['frames']
            fi = int(rng.integers(min(n, 8) if rng.random() < .6 else n))
        self.exposure[self.rows[ei]['source_kind']] += 1
        return int(ei), int(fi)
