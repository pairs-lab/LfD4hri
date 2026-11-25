"""Bound normalized privileged observations, never actions or robot geometry.

R43 preserves a near-constant q4 standard deviation of0.000299rad. A real
0.007rad pose error then becomes23standard deviations, driving a feedback
runaway. Keep every native proprio value and all67 feature slots; saturate
only the standardized67 tail to[-5,5], identically for fitting and inference.
Values outside this bound are explicitly no longer losslessly represented.
"""
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.privileged_inputs import append_numeric67 as original_append

SCHEMA = 'native_prefix_plus_causal67_standardized_clip5_r44_v1'
LIMIT = 5.


def bounded_append(prefix, raw, stats):
    result = original_append(prefix, raw, stats)
    result[len(prefix):] = np.clip(result[len(prefix):], -LIMIT, LIMIT)
    return result


def install():
    # Local process only. No old source file or live old server is modified.
    import VLA.new_vlas.dataset as dataset
    dataset.append_numeric67 = bounded_append


def contract():
    return {'schema': SCHEMA, 'limit_standard_deviations': LIMIT,
        'native_proprio_prefix_unchanged': True, 'all67_feature_slots_preserved': True,
        'out_of_bound_numeric_values_saturated_NOT_lossless': True,
        'train_and_runtime_identical': True, 'no_action_correction_or_oracle': True}


def add_contract(checkpoint, *, parent=None):
    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint/'config.json').read_text())
    cfg['observation_conditioning'] = contract()
    cfg['code'].update({str(p.resolve()): sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    if parent is not None:
        cfg['same_weights_parent_no_extra_optimizer_updates'] = str(Path(parent).resolve())
    atomic_json(checkpoint/'config.json', cfg)
    saved = json.loads((checkpoint/'saved_values.json').read_text())
    saved['config_sha256'] = sha256_file(checkpoint/'config.json')
    atomic_json(checkpoint/'saved_values.json', saved)


def require_contract(checkpoint):
    cfg = json.loads((Path(checkpoint)/'config.json').read_text())
    if cfg.get('observation_conditioning') != contract():
        raise ValueError('Bounded-input policy must have its own explicit conditioning contract')
    for path, digest in cfg['code'].items():
        if sha256_file(path) != digest:
            raise ValueError('Bounded/original native source changed: '+path)
    return cfg
