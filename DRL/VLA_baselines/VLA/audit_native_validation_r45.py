"""Independent8-layout validation replay and whole-horizon command probes.

Never used for optimization or normalizer fitting. All original RGB/joint
observations are checked during physical replay before extracting causal64.
No future labels, replay controls or gold are sent in an inference request.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import requests

from VLA.audit_new_vla_quality_r44 import ROOT, SOURCE, physical_signature, read
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import (verify_source_audit, select_episodes,
    reset_observation_errors, validate_reset_observation)
from VLA.new_vlas.action_contracts import absolute20_from_canonical
from VLA.new_vlas.evaluate import png
from VLA.simstate_scene64 import scene_state


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = SOURCE/'validation_candidate'
    verify_source_audit(source, SOURCE/'validation_audit/audit.json')
    originals = read(source/'index.json')
    selected = select_episodes([r for r in originals if '_closedhover' not in r['file']], per_task=2)
    train = {physical_signature(r['task']) for r in read(SOURCE/'qualified_train/index.json')}
    if any(physical_signature(r['task']) in train for r in selected):
        raise ValueError('Physical TRAIN/validation leakage')
    atomic_json(args.output/'plan.json', {'split': 'validation_NOT_TRAIN', 'source': str(source),
        'selected_files': [r['file'] for r in selected], 'selected_before_neural_predictions': True,
        'source_index_sha256': sha256_file(source/'index.json'), 'normalizer_fitted': False,
        'optimization_authorized': False, 'reserved_final_test_opened': False,
        'extractor_sha256': sha256_file(Path(__file__))})
    items, reports = [], []
    for row in selected:
        with np.load(source/row['file'], allow_pickle=False) as z:
            arrays = {k: z[k] for k in ('front_rgb', 'wrist_rgb', 'joint_positions', 'gripper_closure',
                'ee_position', 'ee_rotation_6d', 'canonical_actions')}
        states, maxima = [], {}
        with audit_environment(source, image_size=384, control_hz=10, seed=0,
                required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
            obs = reset_source_episode(env, source, row)
            for fi, action in enumerate(arrays['canonical_actions']):
                errors = reset_observation_errors(obs, env.normalized_gripper_closure(),
                    {k: arrays[k][fi:fi+1] for k in ('front_rgb', 'wrist_rgb', 'joint_positions', 'gripper_closure')})
                validate_reset_observation(errors)
                for key, value in errors.items(): maxima[key] = max(maxima.get(key, 0), value)
                np.testing.assert_allclose(obs.ee_position, arrays['ee_position'][fi], atol=1e-6, rtol=0)
                np.testing.assert_allclose(obs.ee_rotation_6d, arrays['ee_rotation_6d'][fi], atol=1e-6, rtol=0)
                states.append(scene_state(env, obs))
                obs, _, _, info = env.step(action)
                if info.get('safety_violation'): raise ValueError('Validation replay safety failure')
            if not info['strict_raw_success']: raise ValueError('Original strict terminal no longer reproduced')
        states = np.asarray(states)
        action = arrays['canonical_actions']
        xy = np.flatnonzero((np.linalg.norm(action[:, :2], axis=1) > .008) & (action[:, 6] < -.5))
        if not len(xy): raise ValueError('No real alignment stratum, do not replace scene')
        anchors = [('xy_alignment', int(xy[0]))]
        if row['task']['name'] != 'touch':
            close = np.flatnonzero((action[:, 6] > .5) & (states[:, 17] < .5))
            carry = np.flatnonzero((states[:, 17] > .5) & (np.linalg.norm(action[:, :3], axis=1) > .003))
            if not len(close) or not len(carry): raise ValueError('Missing actual acquisition/held stratum')
            anchors += [('near_unheld_close', int(close[0])), ('verified_hold_transport', int(carry[len(carry)//2]))]
        labels = absolute20_from_canonical(arrays['ee_position'], arrays['ee_rotation_6d'], action)
        for stage, fi in anchors:
            ids = np.minimum(fi+np.arange(30), len(states)-1)
            items.append({'task': row['task']['name'], 'target': row['task']['pick_name'], 'stage': stage,
                'source': str(source/row['file']), 'source_sha256': sha256_file(source/row['file']), 'frame': fi,
                'native_gold_LOCAL_ONLY': labels[ids].tolist(),
                'payload': {'instruction': row['task']['prompt'], 'scene64': states[fi].tolist(),
                    'ee_position': arrays['ee_position'][fi].tolist(), 'ee_rotation_6d': arrays['ee_rotation_6d'][fi].tolist(),
                    'front_png': png(arrays['front_rgb'][fi]), 'wrist_png': png(arrays['wrist_rgb'][fi])}})
        reports.append({'file': row['file'], 'frames': len(states), 'task': row['task']['name'],
            'sensor_maxima': maxima, 'strict_terminal_reproduced': True})
        atomic_json(args.output/'status.json', {'status': 'replaying_validation', 'episodes': len(reports), 'planned_episodes': 8})
        print(json.dumps({'validation_replayed': len(reports)}), flush=True)
    atomic_json(args.output/'requests_with_LOCAL_gold.json', items)
    atomic_json(args.output/'audit.json', {'passed': True, 'split': 'validation_NOT_TRAIN', 'episodes': 8,
        'observations': len(items), 'strata': dict(Counter(r['stage'] for r in items)),
        'rows': reports, 'train_geometry_overlap': 0, 'source_images_and_labels_modified': False,
        'normalizer_fitted': False, 'training_authorized': False, 'reserved_test_used': False,
        'request_file_sha256': sha256_file(args.output/'requests_with_LOCAL_gold.json')})
    atomic_json(args.output/'status.json', {'status': 'completed_validation_fixture_NOT_training_data',
        'episodes': 8, 'observations': len(items)})


def probe(args):
    proof = read(args.output/'audit.json')
    requests_file = args.output/'requests_with_LOCAL_gold.json'
    if not proof['passed'] or proof['split'] != 'validation_NOT_TRAIN' or sha256_file(requests_file) != proof['request_file_sha256']:
        raise ValueError('Actual independently replayed held-out fixture required')
    session = requests.Session()
    url = f'http://127.0.0.1:{args.port}'
    response = session.get(url+'/health', timeout=10); response.raise_for_status()
    metadata = response.json()
    if metadata['family'] != 'xvla' or metadata['actual_optimizer_updates'] < 7000:
        raise ValueError('Own actual fine-tuned X-VLA required')
    results = []
    for item in read(requests_file):
        response = session.post(url+'/act', json=item['payload'], timeout=900); response.raise_for_status()
        prediction = response.json()
        if prediction.get('actual_neural_model') is not True: raise ValueError('Non-neural result')
        action = np.asarray(prediction['actions'])
        gold = np.asarray(item['native_gold_LOCAL_ONLY'])
        closed = np.flatnonzero(action[:, 9] > .5)
        results.append({k: item[k] for k in ('task', 'target', 'stage', 'source', 'frame')})
        results[-1].update(first_translation_error_m=float(np.linalg.norm(action[0, :3]-gold[0, :3])),
            all30_mean_translation_error_m=float(np.linalg.norm(action[:, :3]-gold[:, :3], axis=1).mean()),
            first_gripper_match=bool((action[0, 9] > .5) == (gold[0, 9] > .5)),
            all30_gripper_accuracy=float(np.mean((action[:, 9] > .5) == (gold[:, 9] > .5))),
            first_predicted_CLOSE_index=int(closed[0]) if len(closed) else None)
        atomic_json(args.output/'probe.json', {'status': 'running', 'metadata': metadata, 'rows': results,
            'gold_sent_to_model': False, 'robot_controls_executed': 0, 'task_SR_established': False})
    atomic_json(args.output/'probe.json', {'status': 'completed20_HELDOUT_command_diagnostics_NOT_task_SR',
        'metadata': metadata, 'rows': results, 'gold_sent_to_model': False, 'robot_controls_executed': 0,
        'validation_used_for_optimization_or_normalizer': False,
        'first_translation_error_mean_m': float(np.mean([r['first_translation_error_m'] for r in results])),
        'whole30_translation_error_mean_m': float(np.mean([r['all30_mean_translation_error_m'] for r in results])),
        'task_SR_established': False})
    print(json.dumps({'heldout_observations': len(results),
        'mean_first_translation_error_m': float(np.mean([r['first_translation_error_m'] for r in results]))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'probe'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5845)
    args = parser.parse_args()
    globals()[args.mode](args)
