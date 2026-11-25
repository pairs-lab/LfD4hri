"""Full TRAIN action imbalance and ACTUAL sampled-frame exposure, CPU only.

This is read-only diagnosis, not source admission or an SR measurement. The
exact first-epoch native sampler is reconstructed from the saved counters.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def summarize(actions, states):
    xyz = np.linalg.norm(actions[:, :3], axis=1)
    xy = np.linalg.norm(actions[:, :2], axis=1)
    yaw = np.abs(actions[:, 5])
    held = states[:, 17] > 0
    closing = actions[:, 6] > .5
    # Labels/observed hold are offline diagnostic/sampler strata, never model
    # inputs derived from a teacher phase or desired future state.
    return {'frames': len(actions), 'xy_active_over_5mm': int((xy > .005).sum()),
        'xy_active_over_10mm': int((xy >= .01).sum()),
        'xyz_under_1mm_and_yaw_under_1mrad': int(((xyz < .001) & (yaw < .001)).sum()),
        'yaw_active_over_5mrad': int((yaw > .005).sum()),
        'open_request': int((actions[:, 6] < 0).sum()), 'close_request': int(closing.sum()),
        'close_request_without_current_correct_hold': int((closing & ~held).sum()),
        'current_correct_hold': int(held.sum()), 'current_wrong_hold': int((states[:, 30] > 0).sum())}


def run(checkpoint, output):
    if output.exists():
        raise FileExistsError('Retain actual coverage audits')
    cfg = json.loads((checkpoint/'train_config.json').read_text())
    saved = json.loads((checkpoint/'training_state.json').read_text())
    if saved['epoch'] != 0 or cfg['batch_size'] != 1 or saved['sample_offset'] != saved['total_micro_steps']:
        raise ValueError('Exact current first-epoch batch1 sampler required')
    contract = json.loads((Path(cfg['dataset'])/'privileged_dataset.json').read_text())
    source = Path(contract['original_source'])
    projection = Path(contract['projection'])
    rows = json.loads((source/'index.json').read_text())
    proof = json.loads((projection/'audit.json').read_text())
    if proof['source_index_sha256'] != sha256_file(source/'index.json'):
        raise ValueError('Source index changed')
    actions, states, tasks, targets, hashes = [], [], [], [], {}
    for row, ref in zip(rows, proof['rows'], strict=True):
        if row['file'] != ref['file']:
            raise ValueError('Source order changed')
        path = projection/'states'/row['file']
        if sha256_file(path) != ref['state_payload_sha256']:
            raise ValueError('Native causal source-state payload changed')
        with np.load(source/row['file'], allow_pickle=False) as z:
            command = z['canonical_actions'].astype(np.float32)
        with np.load(path, allow_pickle=False) as z:
            state = z['state'].copy()
        if command.shape != (row['steps'], 7) or state.shape != (row['steps'], 64):
            raise ValueError('Full source timeline differs')
        actions.append(command)
        states.append(state)
        hashes[row['file']] = hashlib.sha256(command.tobytes()).hexdigest()
        tasks.extend([row['task']['name']] * row['steps'])
        targets.extend([row['task']['pick_name']] * row['steps'])
    actions, states, tasks, targets = np.concatenate(actions), np.concatenate(states), np.asarray(tasks), np.asarray(targets)
    if len(actions) != saved['dataset_size']:
        raise ValueError('Native checkpoint/dataset length differs')
    ids = torch.randperm(len(actions), generator=torch.Generator().manual_seed(cfg['seed'])).numpy()[:saved['sample_offset']]
    groups = {}
    for name, selected in (('full_TRAIN', np.arange(len(actions))), ('actual_optimizer_presentations', ids)):
        selected_tasks = tasks[selected]
        task_counts = Counter(selected_tasks.tolist())
        groups[name] = {'all': summarize(actions[selected], states[selected]),
            'by_task': {task: summarize(actions[selected][selected_tasks == task], states[selected][selected_tasks == task]) for task in sorted(task_counts)},
            'task_target_counts': {task: dict(Counter(targets[selected][selected_tasks == task].tolist())) for task in sorted(task_counts)}}
    atomic_json(output, {'scope': 'FULL_TRAIN_action_imbalance_and_actual_sampler_exposure_NOT_SR',
        'groups': groups, 'actual_optimizer_updates': saved['optimizer_step'],
        'actual_microsteps': saved['total_micro_steps'], 'unique_sampled_frames': len(set(ids.tolist())),
        'full_source_frames': len(actions), 'fraction_original_epoch_seen': len(ids)/len(actions),
        'source_index_sha256': sha256_file(source/'index.json'), 'projection_audit_sha256': sha256_file(projection/'audit.json'),
        'canonical_array_sha256': hashes, 'checkpoint_training_state_sha256': sha256_file(checkpoint/'training_state.json'),
        'source_data_or_runtime_changed': False, 'model_failure_cause_proven': False,
        'diagnostic_strata_are_NOT_new_policy_inputs': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint.resolve(), args.output.resolve())
