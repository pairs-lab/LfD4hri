"""Bounded real greedy neural probes: seen versus unseen TRAIN frames.

Uses an already running stateless OpenVLA server, no new model/GPU allocation,
no simulator actions, no held-out fitting and no policy/runtime modifications.
Both groups are TRAIN; unseen means not in this checkpoint's optimizer batches.
"""
import argparse
import base64
from bisect import bisect_right
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import augment_prompt, PROMPT_SCHEMA


def png(image):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode()


def run(args):
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint/'train_config.json').read_text())
    saved = json.loads((checkpoint/'training_state.json').read_text())
    if saved['epoch'] != 0 or saved['total_micro_steps'] != saved['sample_offset'] or config['batch_size'] != 1:
        raise ValueError('This bounded reconstruction requires the actual first-epoch batch1 recipe')
    dataset = Path(config['dataset'])
    contract = json.loads((dataset/'privileged_dataset.json').read_text())
    projection = Path(contract['projection'])
    source = Path(contract['original_source'])
    index = json.loads((source/'index.json').read_text())
    proof = json.loads((projection/'audit.json').read_text())
    if proof['source_index_sha256'] != sha256_file(source/'index.json'):
        raise ValueError('Actual original TRAIN index differs')
    references = {row['file']: row for row in proof['rows']}
    ends = np.cumsum([r['steps'] for r in index]).tolist()
    if ends[-1] != saved['dataset_size']:
        raise ValueError('Actual training sampler length differs')
    order = torch.randperm(ends[-1], generator=torch.Generator().manual_seed(config['seed'])).tolist()
    seen_count = saved['sample_offset']
    selections = []
    for group, candidates in (('actually_seen_by_optimizer', order[:seen_count]),
        ('TRAIN_not_yet_seen_by_optimizer', order[seen_count:])):
        counts, used, archive_cache = Counter(), set(), {}
        for position in candidates:
            ei = bisect_right(ends, position)
            row = index[ei]
            task = row['task']['name']
            if counts[task] >= 2 or row['file'] in used:
                continue
            fi = position - (ends[ei-1] if ei else 0)
            if row['file'] not in archive_cache:
                with np.load(source/row['file'], allow_pickle=False) as z:
                    archive_cache[row['file']] = z['canonical_actions'].copy()
            target = archive_cache[row['file']][fi]
            # Predetermined moving-frame criterion, not neural outcome selection.
            if np.linalg.norm(target[:2]) < .01:
                continue
            counts[task] += 1
            used.add(row['file'])
            selections.append({'group': group, 'index': position, 'file': row['file'],
                'frame': fi, 'task': task, 'target': target.tolist()})
            if sum(counts.values()) == 8:
                break
        if sum(counts.values()) != 8:
            raise ValueError('Each group needs exactly two moving frames per task')
    args.output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    response = session.get(f'http://127.0.0.1:{args.port}/metadata', timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata.get('checkpoint') != str(checkpoint) or metadata.get('model') != 'openvla' or not metadata.get('all_saved_adapter_values_exact'):
        raise ValueError('Exact actual OpenVLA target checkpoint required')
    atomic_json(args.output/'plan.json', {'selection': selections, 'metadata': metadata,
        'source_train_frames': ends[-1], 'actual_optimizer_sample_presentations': seen_count,
        'original_epoch_exposure_fraction': seen_count/ends[-1], 'physical_robot_actions': 0,
        'max_neural_requests': 17, 'held_out_data_used': False})
    reports = []
    first = None
    try:
        for number, selected in enumerate(selections):
            file = selected['file']
            ref = references[file]
            if sha256_file(source/file) != ref['source_payload_sha256'] or sha256_file(projection/'states'/file) != ref['state_payload_sha256']:
                raise ValueError('Actual selected pre-action source changed')
            with np.load(projection/'states'/file, allow_pickle=False) as z:
                state = z['state'][selected['frame']].copy()
            with np.load(source/file, allow_pickle=False) as z:
                front, wrist = z['front_rgb'][selected['frame']], z['wrist_rgb'][selected['frame']]
                instruction = str(z['prompt'].item())
            Image.fromarray(front).save(args.output/f'case{number:02d}_front.png')
            prompt = augment_prompt(instruction, state)
            payload = {'schema': PROMPT_SCHEMA, 'instruction': instruction, 'scene64': state.tolist(),
                'prompt': prompt, 'front_rgb_png': png(front), 'wrist_rgb_png': png(wrist),
                'seed': 20261015, 'inference_index': number}
            reply = session.post(f'http://127.0.0.1:{args.port}/act', json=payload, timeout=900)
            reply.raise_for_status()
            data = reply.json()
            if data['action_source'] != 'native_neural_model' or data['action_kind'] != 'canonical_delta_ee_7d':
                raise ValueError('Actual neural canonical output required')
            prediction = np.asarray(data['actions'][0], np.float32)
            target = np.asarray(selected['target'], np.float32)
            if prediction.shape != (7,) or not np.isfinite(prediction).all():
                raise ValueError('Malformed actual neural output')
            expected_hash = __import__('hashlib').sha256(prompt.encode()).hexdigest()
            if data['prompt_sha256'] != expected_hash:
                raise ValueError('Actual full input identity differs')
            result = {**selected, 'prediction': prediction.tolist(),
                'translation_error_m': float(np.linalg.norm(prediction[:3]-target[:3])),
                'xy_error_m': float(np.linalg.norm(prediction[:2]-target[:2])),
                'xy_dot_target': float(np.dot(prediction[:2], target[:2])),
                'gripper_direction_matches': bool((prediction[6] > 0) == (target[6] > 0)),
                'full_prompt_sha256': expected_hash}
            reports.append(result)
            if first is None:
                first = (payload, prediction.copy())
            atomic_json(args.output/'progress.json', {'completed_neural_requests': len(reports), 'planned_neural_requests': 17,
                'rows': reports, 'model_SR_established': False})
        reply = session.post(f'http://127.0.0.1:{args.port}/act', json=first[0], timeout=900)
        reply.raise_for_status()
        np.testing.assert_array_equal(first[1], np.asarray(reply.json()['actions'][0], np.float32))
        groups = {}
        for group in sorted({r['group'] for r in reports}):
            rows = [r for r in reports if r['group'] == group]
            groups[group] = {'samples': len(rows), 'translation_mean_error_m': float(np.mean([r['translation_error_m'] for r in rows])),
                'xy_mean_error_m': float(np.mean([r['xy_error_m'] for r in rows])),
                'xy_direction_positive': sum(r['xy_dot_target'] > 0 for r in rows),
                'gripper_direction_matches': sum(r['gripper_direction_matches'] for r in rows)}
        atomic_json(args.output/'audit.json', {'scope': 'actual_seen_vs_unseen_TRAIN_neural_moving_frame_diagnostic_NOT_SR',
            'actual_neural_requests': 17, 'identical_repeated_input_prediction': True, 'groups': groups, 'rows': reports,
            'checkpoint_manifest_sha256': sha256_file(checkpoint/'checkpoint_manifest.json'),
            'saved_training_state_sha256': sha256_file(checkpoint/'training_state.json'),
            'implementation_sha256': sha256_file(Path(__file__)), 'physical_robot_actions': 0,
            'training_or_evaluator_changed': False, 'model_SR_established': False})
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
