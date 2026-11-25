"""Independent process, strict FP32 native-adapter reload and prediction parity."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.dataset import NativeData
from VLA.new_vlas.runtime import create_runtime
from VLA.new_vlas.train import wait_gpu


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cp = args.checkpoint.resolve()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(202643)
        config = json.loads((cp/'config.json').read_text())
        saved = json.loads((cp/'saved_values.json').read_text())
        for filename, key in (('trainable_masters.pt', 'trainable_master_sha256'),
                              ('reload_reference.npy', 'reference_prediction_sha256'),
                              ('config.json', 'config_sha256')):
            if sha256_file(cp/filename) != saved[key]:
                raise ValueError('Own saved checkpoint evidence changed')
        gate = json.loads((cp.parents[1]/'gate.json').read_text())
        if not gate['passed'] or gate['checkpoint'] != str(cp):
            raise ValueError('Own completed optimizer gate required')
        data = NativeData(config['data_root'])
        if sha256_file(data.root/'audit.json') != config['data_audit_sha256']:
            raise ValueError('Different complete source')
        wait_gpu(out)
        runtime = create_runtime(config['family'], data.stats)
        keys = runtime.load_trainables(cp/'trainable_masters.pt')
        runtime.to('cuda')
        sample = data.sample(config['probe']['episode'], config['probe']['frame'], config['family'])
        actual = runtime.predict(sample)
        reference = np.load(cp/'reload_reference.npy', allow_pickle=False)
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise ValueError('Reload native prediction shape/finite differs')
        error = float(np.abs(actual-reference).max())
        if error > 1e-6:
            raise ValueError(f'Fresh-process prediction parity failed: {error}')
        atomic_json(out/'audit.json', {'passed': True, 'family': config['family'],
            'checkpoint': str(cp), 'data_audit_sha256': config['data_audit_sha256'],
            'FP32_trainable_values_exact': True, 'trainable_keys': keys,
            'fresh_process_native_prediction_max_difference': error,
            'native_prediction_shape': list(actual.shape), 'pid': os.getpid(),
            'gate_optimizer_updates': config['actual_optimizer_updates'],
            'model_task_SR_established': False})
        atomic_json(out/'status.json', {'status': 'completed', 'pid': os.getpid()})
    except Exception as error:
        atomic_json(out/'status.json', {'status': 'failed_preserve_evidence', 'pid': os.getpid(),
            'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
