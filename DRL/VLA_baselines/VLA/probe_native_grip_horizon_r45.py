"""Inspect the WHOLE learned gripper horizon, not just its first action.

Fixed TRAIN fixture, no robot control and no gold sent to the model. Tests
whether first-action-only replanning discards predicted future CLOSE actions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import requests

from VLA.audit_new_vla_quality_r44 import DATA, ROOT, read
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5845)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = ROOT/'VLA/audits/scene67_deadline_r31_20261015/native_TRAIN_action_probe_r37_v2'
    proof = read(fixture/'audit.json')
    if sha256_file(fixture/'requests_with_LOCAL_gold.json') != proof['request_file_sha256']:
        raise ValueError('Fixed TRAIN fixture changed')
    items = [r for r in read(fixture/'requests_with_LOCAL_gold.json') if r['stage'] == 'near_unheld_close']
    source = {r['source']: r for r in read(DATA/'index.json')}
    session = requests.Session()
    url = f'http://127.0.0.1:{args.port}'
    metadata = session.get(url+'/health', timeout=10).json()
    if metadata['family'] != 'xvla' or metadata['actual_optimizer_updates'] < 7000:
        raise ValueError('Require the real current fine-tuned X-VLA')
    rows = []
    for item in items:
        row, fi = source[item['source_file']], item['source_frame']
        with np.load(row['source'], allow_pickle=False) as z:
            xyz, six = z['ee_position'][fi], z['ee_rotation_6d'][fi]
        payload = {'instruction': row['instruction'], 'scene64': item['payload']['scene64'],
            'ee_position': xyz.tolist(), 'ee_rotation_6d': six.tolist(),
            'front_png': item['payload']['front_rgb_png'], 'wrist_png': item['payload']['wrist_rgb_png']}
        reply = session.post(url+'/act', json=payload, timeout=900)
        reply.raise_for_status()
        result = reply.json()
        if not result['actual_neural_model']: raise ValueError('Non-neural result')
        action = np.asarray(result['actions'])
        closed = np.flatnonzero(action[:, 9] > .5)
        rows.append({'task': item['task'], 'target': item['target'], 'source': row['source'], 'frame': fi,
            'native_gripper_closed_probability_all30': action[:, 9].tolist(),
            'first_CLOSE_index_zero_based': int(closed[0]) if len(closed) else None,
            'first_action_closes': bool(action[0, 9] > .5), 'some_action_in_chunk_closes': bool(len(closed)),
            'full_actual_native_chunk': action.tolist()})
        atomic_json(args.output/'audit.json', {'status': 'running', 'metadata': metadata, 'rows': rows,
            'robot_controls_executed': 0, 'gold_sent_to_server': False, 'task_SR_established': False})
    atomic_json(args.output/'audit.json', {'status': 'completed6_TRAIN_horizon_probes_NOT_SR',
        'metadata': metadata, 'rows': rows, 'first_action_closes_count': sum(r['first_action_closes'] for r in rows),
        'any_future_action_closes_count': sum(r['some_action_in_chunk_closes'] for r in rows),
        'robot_controls_executed': 0, 'gold_sent_to_server': False, 'task_SR_established': False,
        'fixture_sha256': proof['request_file_sha256']})
    print(json.dumps({'first_CLOSE_indices': [r['first_CLOSE_index_zero_based'] for r in rows],
        'first_closes': sum(r['first_action_closes'] for r in rows),
        'any_future_closes': sum(r['some_action_in_chunk_closes'] for r in rows)}))
