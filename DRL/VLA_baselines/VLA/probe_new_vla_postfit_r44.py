"""Bounded TRAIN-only postfit fit/sensitivity probes on existing native servers.

Uses no extra GPU model, sends no robot action, never sends gold labels, and
does not modify the ongoing 40-scene evaluation. Server requests are serial.
Counterfactual geometry is a sensitivity diagnostic, NOT a physical episode.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import requests

from VLA.audit_new_vla_quality_r44 import DATA, ROOT, read
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import canonical_from_absolute20
from VLA.new_vlas.policy_outputs import native_gripper_command

FIXTURE = ROOT / 'VLA/audits/scene67_deadline_r31_20261015/native_TRAIN_action_probe_r37_v2'


def decode(family, result, payload):
    native = np.asarray(result['actions'], np.float32)
    if family == 'xvla':
        action = canonical_from_absolute20(native[0], payload['ee_position'], payload['ee_rotation_6d'])
        action[6] = native_gripper_command(native[0, 9], convention='xvla_closed01')
    else:
        action = native[0].copy()
        action[6] = native_gripper_command(action[6], convention='adapter_open01')
    if not np.isfinite(action).all(): raise ValueError('Nonfinite actual native control')
    return action


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    proof, items = read(FIXTURE / 'audit.json'), read(FIXTURE / 'requests_with_LOCAL_gold.json')
    if not proof['passed'] or sha256_file(FIXTURE / 'requests_with_LOCAL_gold.json') != proof['request_file_sha256']:
        raise ValueError('Existing fixed TRAIN-only fixture changed')
    caches = {row['source']: row for row in read(DATA / 'index.json')}
    session = requests.Session()
    for family, port in (('xvla', 5843), ('vla_adapter', 5844)):
        results, sensitivity = [], []
        def status(state, **extra):
            atomic_json(args.output / (family + '.json'), {'status': state, 'family': family,
                'TRAIN_observations': len(results), 'observations': results, 'sensitivity': sensitivity,
                'additional_GPU_models_loaded': False, 'robot_actions_executed': 0,
                'gold_sent_to_model': False, 'dataset_or_training_modified': False,
                'task_SR_established': False, 'counterfactuals_are_NOT_physical_trials': True, **extra})
        status('waiting_own_finetuned_server')
        url = f'http://127.0.0.1:{port}'
        until = time.monotonic() + args.hours*3600
        metadata = None
        while time.monotonic() < until:
            model_root = ROOT / 'VLA' / family / 'privileged_r43_native_v3_20261016/test1'
            eval_progress = model_root / 'development40/progress.json'
            if eval_progress.exists():
                state = read(eval_progress)['status']
                if state.startswith(('failed', 'completed40')):
                    break  # Never resurrect an exited server or change that cohort.
            try:
                response = session.get(url + '/health', timeout=3)
                response.raise_for_status()
                health = response.json()
                if (health['family'] == family and health['actual_optimizer_updates'] == 6000
                        and Path(health['checkpoint']).is_relative_to(model_root)):
                    metadata = health
                    break
            except requests.RequestException:
                pass
            time.sleep(10)
        if metadata is None:
            status('not_probed_server_unavailable_NO_fake_failure')
            continue
        try:
            def predict(payload):
                response = session.post(url + '/act', json=payload, timeout=900)
                response.raise_for_status()
                result = response.json()
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if result.get('actual_neural_model') is not True or result['input_sha256'] != digest:
                    raise ValueError('Native neural/input identity mismatch')
                return decode(family, result, payload)
            for fixture in items:
                row = caches[fixture['source_file']]
                fi = fixture['source_frame']
                with np.load(DATA / row['cache'], allow_pickle=False) as z:
                    raw = z['scene64'][fi]
                    gold = z['canonical7'][fi]
                with np.load(row['source'], allow_pickle=False) as z:
                    xyz, six = z['ee_position'][fi], z['ee_rotation_6d'][fi]
                payload = {'instruction': row['instruction'], 'scene64': raw.tolist(),
                    'ee_position': xyz.tolist(), 'ee_rotation_6d': six.tolist(),
                    'front_png': fixture['payload']['front_rgb_png'],
                    'wrist_png': fixture['payload']['wrist_rgb_png']}
                np.testing.assert_array_equal(raw, fixture['payload']['scene64'])
                predicted = predict(payload)
                error = predicted - gold
                results.append({'task': fixture['task'], 'target': fixture['target'], 'stage': fixture['stage'],
                    'source': row['source'], 'frame': fi, 'canonical_prediction': predicted.tolist(),
                    'LOCAL_gold_not_transmitted': gold.tolist(),
                    'translation_L2_error_m': float(np.linalg.norm(error[:3])),
                    'gripper_binary_match': bool(predicted[6] == gold[6])})
                if fixture['stage'] == 'xy_alignment' and fixture['task'] not in [s['task'] for s in sensitivity]:
                    counterfactual = copy.deepcopy(payload)
                    counterfactual['scene64'][11] *= -1
                    counterfactual['scene64'][12] *= -1
                    changed = predict(counterfactual)
                    direction = np.asarray(counterfactual['scene64'][11:13]) - raw[11:13]
                    sensitivity.append({'task': fixture['task'], 'counterfactual_target_xy_sign_flip_only': True,
                        'prediction_translation_change_m': (changed[:3]-predicted[:3]).tolist(),
                        'xy_response_dot_target_displacement': float(np.dot(changed[:2]-predicted[:2], direction)),
                        'physically_executed': False, 'interpretation': 'sensitivity_NOT_correctness_or_SR'})
                status('probing_actual_loaded_postfit_native_VLA', metadata=metadata)
            status('completed_TRAIN20_not_SR', metadata=metadata,
                mean_translation_L2_error_m=float(np.mean([r['translation_L2_error_m'] for r in results])),
                binary_gripper_matches=sum(r['gripper_binary_match'] for r in results),
                request_fixture_sha256=proof['request_file_sha256'])
        except Exception as error:
            status('failed_probe_preserve_evidence_NO_fake_SR', metadata=metadata,
                   error=f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.output.exists() or args.output.with_name(args.output.name+'_owner.json').exists():
            raise FileExistsError('Do not duplicate a native probe')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'VLA.probe_new_vla_postfit_r44',
                '--output', str(args.output.resolve()), '--hours', str(args.hours)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': child.pid})
        print(json.dumps({'pid': child.pid, 'output': str(args.output)}))
    else:
        run(args)
