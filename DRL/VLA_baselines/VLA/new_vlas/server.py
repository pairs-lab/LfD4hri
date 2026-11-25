"""Loopback-only native VLA server. The simulator stays in its original env."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import pack_native20, native8, normalize_bounds
from VLA.new_vlas.privileged_inputs import append_numeric67
from VLA.new_vlas.runtime import create_runtime


def main(args):
    from flask import Flask, jsonify, request
    torch.set_num_threads(2)
    torch.manual_seed(202643)
    cp = args.checkpoint.resolve()
    config = json.loads((cp/'config.json').read_text())
    saved = json.loads((cp/'saved_values.json').read_text())
    if sha256_file(cp/'trainable_masters.pt') != saved['trainable_master_sha256']:
        raise ValueError('Checkpoint saved master bytes changed')
    if sha256_file(cp/'config.json') != saved['config_sha256']:
        raise ValueError('Checkpoint configuration changed')
    runtime = create_runtime(config['family'], config['stats'])
    runtime.load_trainables(cp/'trainable_masters.pt')
    runtime.to('cuda')
    runtime.eval()
    metadata = {'family': config['family'], 'checkpoint': str(cp),
        'action_source': 'actual_native_pretrained_VLA_with_target_domain_LoRA',
        'native_horizon': runtime.horizon, 'numeric_privileged_fields': 67,
        'actual_optimizer_updates': config['actual_optimizer_updates'],
        'target_domain_finetuned': config['actual_optimizer_updates'] > 2,
        'oracle_actions_or_privileged_geometric_action_correction': False,
        'low_level_robot_controller': 'UNCHANGED shared PyBullet delta-EEF motor interface',
        'checkpoint_master_sha256': saved['trainable_master_sha256']}
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8*2**20

    @app.get('/health')
    def health():
        return jsonify(metadata)

    @app.post('/act')
    def act():
        payload = request.get_json()
        expected = {'instruction', 'scene64', 'ee_position', 'ee_rotation_6d', 'front_png', 'wrist_png'}
        if set(payload) != expected:
            raise ValueError('Only declared causal observations allowed')
        raw = np.asarray(payload['scene64'], np.float32)
        xyz = np.asarray(payload['ee_position'], np.float32)
        six = np.asarray(payload['ee_rotation_6d'], np.float32)
        if xyz.shape != (3,) or six.shape != (6,):
            raise ValueError('Native measured EEF pose missing')
        if config['family'] == 'xvla':
            prefix = pack_native20(xyz, six, np.asarray(raw[6]))
        else:
            prefix = normalize_bounds(native8(xyz, six, raw[27]), config['stats']['adapter']['proprio'])
        sample = {'instruction': payload['instruction'],
            'state': append_numeric67(prefix, raw, config['stats']['scene67'])}
        for role in ('front', 'wrist'):
            pixels = np.asarray(Image.open(BytesIO(base64.b64decode(payload[role+'_png']))).convert('RGB'))
            if pixels.shape != (384, 384, 3) or pixels.dtype != np.uint8:
                raise ValueError('Frozen camera resolution/format changed')
            sample[role] = pixels
        prediction = runtime.predict(sample)
        shape = (30, 20) if config['family'] == 'xvla' else (8, 7)
        if prediction.shape != shape or not np.isfinite(prediction).all():
            raise ValueError('Actual native prediction malformed')
        kind = 'native_absolute_ee20_closure' if config['family'] == 'xvla' else 'delta_ee7_open01'
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return jsonify({'actions': prediction.tolist(), 'action_kind': kind,
            'actual_neural_model': True, 'input_sha256': digest})

    app.run(host='127.0.0.1', port=args.port, threaded=False, use_reloader=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    main(parser.parse_args())
