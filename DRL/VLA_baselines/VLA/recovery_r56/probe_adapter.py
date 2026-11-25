"""Actual TRAIN-only native-head overfit and gradient-path diagnostic.

Not an alternative controller or benchmark policy. Never exports a servable
checkpoint; cached visual/language features are used ONLY for this diagnosis.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
import os
import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.common import DATA


def main(a):
    import torch
    from VLA.stable_native_r44.conditioning import install
    from VLA.privileged_visual_r53.contract import require_visual
    from VLA.privileged_visual_r53.data import MapData
    from VLA.privileged_visual_r53.runtime import create_runtime
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = require_visual(a.checkpoint)
    if cfg['family'] != 'vla_adapter':
        raise ValueError('Only genuine native Adapter')
    torch.set_num_threads(2)
    torch.manual_seed(202656)
    install()
    data = MapData(DATA)
    runtime = create_runtime('vla_adapter', data.stats)
    runtime.load_trainables(a.checkpoint/'trainable_masters.pt')
    torch.cuda.set_per_process_memory_fraction(14000*2**20/torch.cuda.get_device_properties(0).total_memory)
    runtime.to('cuda')
    official = type(runtime.head).predict_action
    cached, gradients = [], {}
    def status(stage, **kw):
        atomic_json(a.output/'status.json', {'status': 'running', 'stage': stage,
            'pid': os.getpid(), 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **kw})
    status('native_full_VLM_feature_extraction')
    def capture(head, features, **kwargs):
        cached.append({'features': features.detach().cpu(),
            'proprio': kwargs['proprio'].detach().cpu()})
        return official(head, features, **kwargs, phase='Inference')
    runtime.head.predict_action = MethodType(capture, runtime.head)
    anchors = []
    for task in sorted(data.groups):
        for kind in ('complete_original', 'pose_corrective'):
            choices = data.groups[task][kind]
            ei = choices[sorted(choices)[0]][0]
            fi = int(np.argmax(np.linalg.norm(data.numeric(ei)['canonical7'][:, :2], axis=1)))
            sample = data.sample(ei, fi, 'vla_adapter')
            runtime.predict(sample)
            cached[-1]['target'] = torch.from_numpy(sample['action'])[None]
            anchors.append({'episode': ei, 'frame': fi, 'task': task, 'kind': kind,
                'source': data.rows[ei]['source']})
    if len(cached) != 8:
        raise ValueError('All four tasks and both source branches required')
    # Check the REAL full-model gradient once before freezing cached VLM
    # features for the small diagnostic. No native parameter is detached here.
    runtime.head.predict_action = MethodType(lambda head, features, **kw:
        official(head, features, **kw, phase='Training' if runtime.training_capture else 'Inference'), runtime.head)
    def training_capture(head, features, **kw):
        value = official(head, features, **kw, phase='Training' if runtime.training_capture else 'Inference')
        if runtime.training_capture:
            runtime.captured = value
        return value
    runtime.head.predict_action = MethodType(training_capture, runtime.head)
    loss, _ = runtime.loss(data.sample(anchors[0]['episode'], anchors[0]['frame'], 'vla_adapter'))
    loss.backward()
    for group in ('policy', 'head', 'projector'):
        values = [p.grad.detach().float() for p in runtime.modules[group].parameters() if p.grad is not None]
        gradients[group] = {'tensors_with_gradient': len(values),
            'absolute_gradient_sum': sum(float(v.abs().sum()) for v in values),
            'nonzero_elements': sum(int(torch.count_nonzero(v)) for v in values)}
    tail = runtime.projector.fc1.weight.grad
    gradients['projector_appended67_abs_sum'] = None if tail is None else float(tail[:, 8:].abs().sum())
    if any(gradients[g]['nonzero_elements'] == 0 for g in ('policy', 'head', 'projector')):
        raise ValueError('A required native learned branch receives no actual gradient')
    for p in runtime.parameters().values():
        p.grad = None
    parameters = list(runtime.head.parameters()) + list(runtime.projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .99), weight_decay=0., foreach=False)
    def prediction(row, proprio=None):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return official(runtime.head, row['features'].to('cuda'),
                proprio=row['proprio'].to('cuda') if proprio is None else proprio,
                proprio_projector=runtime.projector, phase='Inference')
    def measure():
        errors, magnitudes, sensitivity = [], [], []
        with torch.no_grad():
            for i, row in enumerate(cached):
                value = prediction(row).float()
                target = row['target'].to('cuda')
                errors.append(float(abs(value[:, :, :2]-target[:, :, :2]).mean()))
                magnitudes.append(float(value[:, :, :2].abs().mean()))
                swapped = prediction(row, cached[(i+1)%len(cached)]['proprio'].to('cuda')).float()
                sensitivity.append(float(abs(value[:, :, :2]-swapped[:, :, :2]).mean()))
        return {'normalized_XY_L1': float(np.mean(errors)),
            'mean_abs_normalized_predicted_XY': float(np.mean(magnitudes)),
            'proprio_swap_XY_sensitivity': float(np.mean(sensitivity))}
    initial = measure()
    history = []
    for step in range(1, 65):
        optimizer.zero_grad(set_to_none=True)
        row = cached[(step-1)%len(cached)]
        value = prediction(row).float()
        loss = torch.nn.functional.l1_loss(value, row['target'].to('cuda'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        optimizer.step()
        if step%16 == 0:
            history.append({'actual_diagnostic_head_updates': step, **measure()})
            status('TRAIN_cached_feature_native_head_diagnostic_ONLY', history=history, initial=initial)
    result = {'passed': True, 'scope': 'TRAIN_ONLY_diagnostic_NOT_environment_SR_or_production_fit',
        'checkpoint': str(a.checkpoint.resolve()), 'all_four_tasks_both_sources': True,
        'native_full_model_gradient': gradients, 'initial': initial, 'history': history,
        'anchors': anchors, 'exported_policy_checkpoint': False,
        'no_simulator_camera_controller_changed': True,
        'XY_train_error_reduction_fraction': 1-history[-1]['normalized_XY_L1']/max(initial['normalized_XY_L1'], 1e-8)}
    atomic_json(a.output/'audit.json', result)
    atomic_json(a.output/'status.json', {'status': 'completed', 'pid': os.getpid(), **result})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
