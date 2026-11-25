"""Compare native head Training/Inference on identical real TRAIN features.

No optimization, simulation episodes, controller modification or SR claim.
The official Training branch injects random input perturbations; measure its
effect rather than inferring that a low training loss predicts control quality.
"""
import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.dataset import NativeData
from VLA.new_vlas.runtime import create_runtime
from VLA.new_vlas.train import wait_gpu
from VLA.stable_native_r44.conditioning import install, require_contract


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    install()
    cfg = require_contract(args.checkpoint)
    if cfg['family'] != 'vla_adapter':
        raise ValueError('Adapter only')
    data = NativeData(cfg['data_root'], verify_rgb_hashes=False)
    wait_gpu(args.output)
    runtime = create_runtime('vla_adapter', data.stats)
    runtime.load_trainables(args.checkpoint/'trainable_masters.pt')
    runtime.to('cuda'); runtime.eval()
    captured = {}
    official = type(runtime.head).predict_action
    def paired(head, features, **kwargs):
        deterministic = official(head, features, **kwargs, phase='Inference')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(202652)
            training = official(head, features, **kwargs, phase='Training')
        captured.update(inference=deterministic.detach().float().cpu().numpy()[0],
                        training=training.detach().float().cpu().numpy()[0])
        return deterministic
    runtime.head.predict_action = MethodType(paired, runtime.head)
    rows = []
    for task in sorted(data.groups):
        for kind in ('complete_original', 'fresh_corrective'):
            names = data.groups[task][kind]
            ei = names[sorted(names)[0]][0]
            canonical = data.numeric(ei)['canonical7']
            fi = int(np.argmax(np.linalg.norm(canonical[:, :2], axis=1)))
            sample = data.sample(ei, fi, 'vla_adapter')
            if sha256_file(data.rows[ei]['source']) != data.rows[ei]['source_sha256']:
                raise ValueError('Real TRAIN RGB changed')
            actual = runtime.predict(sample)
            lo, hi = (np.asarray(data.stats['adapter']['action'][k]) for k in ('q01', 'q99'))
            def physical(norm):
                return np.where(data.stats['adapter']['action']['mask'], .5*(norm+1)*(hi-lo)+lo, norm)
            predicted = physical(captured['inference'])
            np.testing.assert_allclose(actual[:, :6], predicted[:, :6], atol=1e-7, rtol=0)
            rows.append({'task': task, 'kind': kind, 'source': data.rows[ei]['source'], 'frame': fi,
                'target_normalized': sample['action'].tolist(), 'native_normalized': {k:v.tolist() for k,v in captured.items()},
                'normalized_L1_xyz': {k:float(abs(v[:, :3]-sample['action'][:, :3]).mean()) for k,v in captured.items()},
                'normalized_L1_all7': {k:float(abs(v-sample['action']).mean()) for k,v in captured.items()},
                'native_first_action_physical': actual[0].tolist(),
                'GT_first_action_physical': data.numeric(ei)['adapter_action7'][fi].tolist()})
            atomic_json(args.output/'progress.json', {'scope': 'TRAIN_head_phase_diagnostic_NOT_environment_SR', 'rows': rows})
    atomic_json(args.output/'audit.json', {'passed': True, 'scope': '8_real_TRAIN_max_XY_motion_samples_NOT_generalization_or_SR',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256_file(args.checkpoint/'config.json'),
        'identical_VLM_features_proprio_and_weights_for_both_phases': True,
        'actual_physical_inference_vs_normalized_head_decode_checked': True,
        'no_optimizer_steps_and_no_policy_or_simulator_file_changed': True,
        'mean_normalized_L1_xyz': {k:float(np.mean([r['normalized_L1_xyz'][k] for r in rows])) for k in ('training','inference')},
        'rows': rows})
    print(json.dumps({'samples':len(rows),'mean_normalized_L1_xyz':{k:float(np.mean([r['normalized_L1_xyz'][k] for r in rows])) for k in ('training','inference')}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
