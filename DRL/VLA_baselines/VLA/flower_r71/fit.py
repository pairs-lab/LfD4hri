"""One bounded128-update rehearsal, same native network/labels/loss/inputs.

This is a separate checkpoint with a fresh low-LR Adam state, NOT an exact
optimizer resume or a claim of new recovery demonstrations.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r69.native import load_foundation, adapt, native_batch, action_targets
from VLA.flower_r69.gate import prediction, anchors
from VLA.flower_r69.loss import loss as masked_loss
from VLA.flower_r70.common import checkpoint, verify, freeze_test_code
from VLA.flower_r71.sampler import GraspData, RECIPE


def main(a):
    if a.updates != 128:
        raise ValueError('Predeclared single128-update rehearsal only')
    source = checkpoint(a.source)
    if source['actual_new_optimizer_updates'] != 4096:
        raise ValueError('Wait for the untouched main4096 fit')
    if sha256_file(source['base']['official_checkpoint']) != 'f654fe4adc6a348e9c06424a3444fddf700820818b0b81b917db0ccec94eff60':
        raise ValueError('Pinned native foundation changed')
    code = dict(source['code']); code.update(freeze_test_code())
    code.update({str(p.resolve()): sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    verify(code)
    torch.set_num_threads(2); torch.manual_seed(202671)
    torch.cuda.set_per_process_memory_fraction(.79)
    a.output.mkdir(parents=True, exist_ok=False)
    data = GraspData()
    inherited_stats = json.loads((a.source / 'stats.json').read_text())
    if data.stats != inherited_stats:
        raise ValueError('Never silently renormalize a loaded native checkpoint')
    model, tokenizer, _ = load_foundation(); adapt(model)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    saved = torch.load(a.source / 'trainables.pt', weights_only=True, mmap=True, map_location='cpu')
    if set(saved) != set(params): raise ValueError('Wrong source master keys')
    with torch.no_grad():
        for name, p in params.items():
            if saved[name].dtype != torch.float32 or saved[name].shape != p.shape:
                raise ValueError('Wrong source master dtype/shape')
            p.copy_(saved[name])
            if not torch.equal(p.detach().cpu(), saved[name]):
                raise ValueError('Source master not loaded exactly')
    del saved
    plan = {'model': source['model'], 'source': str(a.source.resolve()),
        'base': source['base'], 'adaptation': source['adaptation'],
        'source_trainables_sha256': source['trainables_sha256'], 'code': code,
        'completed': False, 'sampler': RECIPE, 'transition_coverage': data.coverage(),
        'fresh_low_LR_Adam_NOT_exact_optimizer_resume': True,
        'parent_actual_new_optimizer_updates': 4096,
        'planned_additional_rehearsal_updates': 128,
        'TRAIN_episodes': len(data.rows), 'TRAIN_frames': len(data),
        'constant_learning_rates': [2e-6, 4e-6], 'accumulation': 16,
        'model_input_action_normalization_and_native_masked_RF_unchanged': True,
        'no_new_demonstrations_collected': True, 'read_final800_outcomes': False}
    atomic_json(a.output / 'PLAN.json', plan); atomic_json(a.output / 'stats.json', data.stats)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in params.items() if n.startswith('vlm.')], 'lr': 2e-6},
        {'params': [p for n, p in params.items() if not n.startswith('vlm.')], 'lr': 4e-6},
    ], betas=(.9, .95), weight_decay=.05, foreach=False)
    rng = np.random.default_rng(202671); counts = Counter(); losses = []; started = time.monotonic()
    with (a.output / 'updates.jsonl').open('x') as journal:
        for step in range(128):
            verify(code); optimizer.zero_grad(set_to_none=True); micro = []
            for _ in range(16):
                ei, fi = data.draw(rng); sample = data.at(ei, fi)
                batch = native_batch(sample, tokenizer, data.stats)
                target = torch.from_numpy(action_targets(sample['action'], data.stats))[None].to('cuda')
                pad = torch.from_numpy(sample['action_is_pad'])[None].to('cuda')
                model.train(); model.vlm.eval()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    cond = model.encode_observations(batch); value = masked_loss(model, cond, target, pad)
                if not torch.isfinite(value): raise ValueError('Nonfinite rehearsal loss')
                (value / 16).backward(); micro.append(float(value.detach()))
                counts[sample['task'] + '/' + data.last_sampling_mode] += 1
                del batch, target, pad, cond, value
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(np.mean(micro)))
            row = {'additional_optimizer_updates': step + 1, 'loss': losses[-1],
                'pre_clip_gradient_norm': float(norm), 'sampler_counts': dict(counts),
                'elapsed_s': time.monotonic() - started}
            journal.write(json.dumps(row) + '\n'); journal.flush()
            atomic_json(a.output / 'status.json', {'status': 'running_actual_rehearsal', **row})
    optimizer.zero_grad(set_to_none=True); verify(code)
    temp = a.output / 'trainables.pt.pending'
    with temp.open('xb') as stream:
        torch.save({n: p.detach().cpu().contiguous().clone() for n, p in params.items()}, stream)
    os.replace(temp, a.output / 'trainables.pt')
    selected = anchors(data)[::2]
    predictions = [prediction(model, native_batch(data.at(ei, fi), tokenizer, data.stats)) for ei, fi in selected]
    np.save(a.output / 'PREDICTIONS.npy', np.stack(predictions), allow_pickle=False)
    result = {**plan, 'completed': True, 'actual_new_optimizer_updates': 4096 + 128,
        'actual_additional_rehearsal_updates': 128, 'accumulation': 16,
        'trainables_sha256': sha256_file(a.output / 'trainables.pt'), 'anchors': selected,
        'loss_first32': float(np.mean(losses[:32])), 'loss_last32': float(np.mean(losses[-32:])),
        'sampler_counts': dict(counts), 'robot_success_rate_measured': False,
        'elapsed_s': time.monotonic() - started}
    atomic_json(a.output / 'TRAINING.json', result)
    atomic_json(a.output / 'status.json', {'status': 'completed_requires_fresh_reload_and_real_trials',
        'actual_additional_rehearsal_updates': 128})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=128); main(p.parse_args())
