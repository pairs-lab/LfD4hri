import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.autonomy_r83.common import read, verify, PORT
from VLA.autonomy_r83.runtime import Runtime
from VLA.common.provenance import sha256_file


def main(a):
    import torch
    from flask import Flask, request, jsonify
    from VLA.privileged_visual_r53.scene_map import render, task_from_instruction
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.84)
    method = read(a.method)
    verify(method['code'])
    verify(method['weights_sha256'])
    stats = method.get('stats')
    runtime = Runtime(method['family'], method['parent'], stats, method['variant'],
                      checkpoint=method.get('checkpoint'))
    meta = {'method_sha256': sha256_file(a.method), 'family': method['family'],
            'checkpoint': method.get('checkpoint', method['parent']), 'variant': method['variant'],
            'actual_neural_model': True, 'native_horizon': runtime.horizon,
            'action_kind': 'canonical_world_translation_local_rotation_signed_close7',
            'Oracle_or_geometric_action_residual': False}
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8 * 2**20

    @app.get('/health')
    def health():
        return jsonify(meta)

    @app.post('/act')
    def act():
        payload = request.get_json()
        expected = {'instruction', 'scene64', 'ee_position', 'ee_rotation_6d',
                    'front_png', 'wrist_png', 'seed', 'inference_index'}
        if set(payload) != expected:
            raise ValueError('Undeclared policy information')
        sample = runtime.observed(payload)
        seed = int.from_bytes(hashlib.sha256(json.dumps([payload['seed'], payload['instruction'],
            payload['inference_index']]).encode()).digest()[:8], 'little') & ((1 << 63) - 1)
        result = runtime.predict(sample, seed)
        gt = render(np.asarray(payload['scene64'], np.float32), task_from_instruction(payload['instruction']))
        return jsonify(actions=result.tolist(), actual_neural_model=True,
            input_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            map_RGB_sha256=hashlib.sha256(gt.tobytes()).hexdigest(), receipt=runtime.last_receipt)
    app.run(host='127.0.0.1', port=a.port, threaded=False, use_reloader=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=Path, required=True)
    p.add_argument('--port', type=int, default=PORT)
    main(p.parse_args())
