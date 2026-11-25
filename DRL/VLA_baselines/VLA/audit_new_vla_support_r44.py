"""Quantify support for real grasp recovery and privileged input learning.

Small CPU-only audit; no running trainer, source dataset or robot is changed.
An upward command with an empty hand is not automatically a bad label:
recentring a moving Robotiq pad and collision-free retreat can require it.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from VLA.audit_new_vla_quality_r44 import DATA, ROOT, read
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import NUMERIC_NAMES


def main(args):
    rows = read(DATA / 'index.json')
    counts = Counter()
    per_object = Counter()
    candidates = []
    for ei, row in enumerate(rows):
        with np.load(DATA / row['cache'], allow_pickle=False) as z:
            state, action = z['scene64'], z['canonical7']
        with np.load(row['source'], allow_pickle=False) as z:
            next_held = z['next_grasped_object'].astype(bool)
        attached, verified = state[:, 31] > .5, state[:, 17] > .5
        measured_empty_closed = (state[:, 6] > .95) & ~attached
        reopen = measured_empty_closed & (action[:, 6] < -.5)
        # Conservative candidate only: excludes acquisition within this step.
        empty_move = (~attached & ~next_held & (action[:, 6] > .5))
        large_empty_lift = empty_move & (action[:, 2] > .01)
        large_empty_xy = empty_move & (np.linalg.norm(action[:, :2], axis=1) > .02)
        drop = attached & ~next_held
        counts.update({'measured95pct_closed_empty_frames': int(measured_empty_closed.sum()),
            'measured95pct_closed_empty_reopen_labels': int(reopen.sum()),
            'episodes_with_measured95pct_closed_empty_reopen': int(reopen.any()),
            'wrong_attached_frames': int((state[:, 30] > .5).sum()),
            'any_attached_to_target_unheld_transitions': int(drop.sum()),
            'episodes_with_hold_loss_transitions': int(drop.any()),
            'correct_bilateral_hold_frames': int(verified.sum()),
            'closed_empty_lift_over1cm_candidates': int(large_empty_lift.sum()),
            'closed_empty_XY_over2cm_candidates': int(large_empty_xy.sum())})
        if reopen.any():
            per_object[row['task'] + '/' + row['pick_name']] += 1
        if large_empty_lift.any() or large_empty_xy.any():
            candidates.append({'episode': ei, 'source': row['source'],
                'lift_frames': np.flatnonzero(large_empty_lift).tolist(),
                'xy_frames': np.flatnonzero(large_empty_xy).tolist()})
    stats = read(DATA / 'stats.json')['scene67']
    minimum, maximum = np.asarray(stats['min']), np.asarray(stats['max'])
    constants = [NUMERIC_NAMES[i] for i in np.flatnonzero(maximum == minimum)]
    # Read saved trainable masters, not an independently instantiated model.
    cp = args.checkpoint.resolve()
    config = read(cp / 'config.json')
    if config['family'] != 'xvla' or config['data_audit_sha256'] != sha256_file(DATA / 'audit.json'):
        raise ValueError('Own saved X-VLA and exact audited TRAIN required')
    values = torch.load(cp / 'trainable_masters.pt', map_location='cpu', mmap=True, weights_only=True)
    embedding = values['policy.model.transformer.action_encoder.fc.weight']
    # Official DomainAwareLinear packs [domain, input*1024]. 20 action and
    # 20 original-proprio columns precede the appended67; time columns follow.
    matrix = embedding.reshape(embedding.shape[0], -1, 1024)[12]
    norms = matrix[40:107].norm(dim=1)
    if len(norms) != 67 or not torch.isfinite(norms).all():
        raise ValueError('Native input packing/finite master mismatch')
    weights = {name: float(norms[i]) for i, name in enumerate(NUMERIC_NAMES)}
    result = {'scope': 'all_TRAIN_recovery_support_and_saved_native_input_columns_NOT_SR',
        'counts': dict(counts), 'measured_empty_reopen_episode_counts_by_target': dict(per_object),
        'untrained_constant_numeric_features': constants,
        'saved_xvla_checkpoint': str(cp), 'actual_optimizer_updates': config['actual_optimizer_updates'],
        'causal67_saved_input_column_L2_norms': weights,
        'zero_input_columns': [name for name, norm in weights.items() if norm == 0],
        'input_weight_update_does_NOT_prove_correct_causal_use': True,
        'empty_move_candidates_are_NOT_automatically_invalid_labels': True,
        'candidate_motions': candidates,
        'current_wrongheld_pilot_not_part_of_TRAIN': True,
        'training_mutated': False, 'test_or_validation_labels_used': False,
        'bound_files': {str(p): sha256_file(p) for p in (Path(__file__), cp / 'config.json',
            cp / 'trainable_masters.pt', DATA / 'stats.json', DATA / 'index.json')}}
    atomic_json(args.output / 'recovery_and_native_input_support.json', result)
    print(json.dumps({'counts': dict(counts), 'constant_inputs': constants,
        'zero_input_columns': result['zero_input_columns'], 'candidate_episodes': len(candidates)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
