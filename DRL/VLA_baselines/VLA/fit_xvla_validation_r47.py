"""Unchanged native bounded/acquisition optimizer plus held-out monitoring.

Validation is prediction-only, never backpropagated or used to fit stats.
The actual X-VLA runtime forks/restores its RNG for prediction, so probes
do not consume the training sampling or action-noise streams.
"""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import pack_native20
from VLA.stable_native_r44.conditioning import bounded_append


def monitor_rows(runtime, stats, items, samples):
    """Gold remains local and is used only AFTER the prediction returns."""
    rows = []
    for item, images in zip(items, samples, strict=True):
        payload = item['payload']
        raw = np.asarray(payload['scene64'], np.float32)
        prefix = pack_native20(np.asarray(payload['ee_position']), np.asarray(payload['ee_rotation_6d']), np.asarray(raw[6]))
        state = bounded_append(prefix, raw, stats['scene67'])
        prediction = np.asarray(runtime.predict({**images, 'instruction': payload['instruction'], 'state': state}))
        gold = np.asarray(item['native_gold_LOCAL_ONLY'], np.float32)
        if prediction.shape != (30, 20) or gold.shape != prediction.shape:
            raise ValueError('Held-out monitor requires the actual full native30x20 horizon')
        if not np.isfinite(prediction).all() or not np.isfinite(gold).all():
            raise ValueError('Non-finite held-out command diagnostics')
        p = np.clip(prediction[:, 9], 1e-7, 1-1e-7)
        xyz_mse = float(np.mean((prediction[:, :3]-gold[:, :3])**2))
        rotation6d_mse = float(np.mean((prediction[:, 3:9]-gold[:, 3:9])**2))
        bce = float(np.mean(-gold[:, 9]*np.log(p)-(1-gold[:, 9])*np.log(1-p)))
        rows.append({'task': item['task'], 'target': item['target'], 'stage': item['stage'],
            'physical_arm_xyz_mse': xyz_mse, 'physical_arm_rotation6d_mse': rotation6d_mse,
            'physical_gripper_BCE': bce,
            'mean30_translation_L2_m': float(np.linalg.norm(prediction[:, :3]-gold[:, :3], axis=1).mean()),
            'mean30_gripper_accuracy': float(np.mean((p > .5) == (gold[:, 9] > .5))),
            'single_physical_arm_weighted_diagnostic': 500*xyz_mse+10*rotation6d_mse+bce})
    return rows


def main(args):
    fixture = args.validation_fixture.resolve()
    proof = json.loads((fixture/'audit.json').read_text())
    path = fixture/'requests_with_LOCAL_gold.json'
    if (not proof['passed'] or proof['split'] != 'validation_NOT_TRAIN'
            or proof['train_geometry_overlap'] != 0 or sha256_file(path) != proof['request_file_sha256']):
        raise ValueError('Independent real replayed validation fixture required')
    items = json.loads(path.read_text())
    samples = []
    for item in items:
        sample = {role: np.array(Image.open(BytesIO(base64.b64decode(item['payload'][role+'_png']))))
                  for role in ('front', 'wrist')}
        samples.append(sample)
    import VLA.new_vlas.train as native
    from VLA.stable_native_r44.fit_balanced import main as fit, draw as expected_draw
    original_save = native.save_checkpoint
    def save(out, runtime, optimizer, data, step, rng, config, probe):
        if getattr(data.draw, '__func__', None) is not expected_draw:
            raise ValueError('Actual sampler identity is not acquisition balanced')
        cp = original_save(out, runtime, optimizer, data, step, rng, config, probe)
        rows = monitor_rows(runtime, data.stats, items, samples)
        atomic_json(cp/'heldout_monitor.json', {'scope': '20_observations_from8_replayed_VALIDATION_layouts_NOT_SR',
            'actual_optimizer_updates': step, 'rows': rows, 'optimizer_or_stats_fit_used_validation': False,
            'future_gold_not_given_to_prediction': True, 'inference_RNG_restored_by_native_runtime': True,
            'mean30_translation_L2_m': float(np.mean([r['mean30_translation_L2_m'] for r in rows])),
            'mean30_gripper_accuracy': float(np.mean([r['mean30_gripper_accuracy'] for r in rows])),
            'single_physical_arm_weighted_diagnostic_mean': float(np.mean([r['single_physical_arm_weighted_diagnostic'] for r in rows])),
            'checkpoint_selected_on_this_score': False})
        cfg = json.loads((cp/'config.json').read_text())
        cfg['actual_acquisition_sampler_function_identity_verified'] = True
        cfg['validation_monitor'] = {'fixture': str(fixture), 'optimization_or_normalizer_use': False,
            'checkpoint_selection': 'fixed_last20000_NOT_selected_on_validation_or_development_SR'}
        for frozen in (Path(__file__).resolve(), fixture/'audit.json', path):
            cfg['code'][str(frozen)] = sha256_file(frozen)
        atomic_json(cp/'config.json', cfg)
        saved = json.loads((cp/'saved_values.json').read_text())
        saved['config_sha256'] = sha256_file(cp/'config.json')
        atomic_json(cp/'saved_values.json', saved)
        return cp
    native.save_checkpoint = save
    fit(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla',), default='xvla')
    parser.add_argument('--mode', choices=('gate', 'long'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--total-updates', type=int, default=20000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--reload-proof', type=Path)
    parser.add_argument('--validation-fixture', type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((args.resume/'config.json').read_text())['actual_optimizer_updates']
    if args.mode == 'gate' and (args.updates-previous != 2 or args.accumulation != 4):
        parser.error('Gate must run2 real new updates /8 micros')
    main(args)
