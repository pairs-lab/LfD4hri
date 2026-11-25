"""Fresh, disclosed balanced/masked adaptation, with actual gate and exact resume."""
import argparse
from collections import Counter
import math
from pathlib import Path
import time
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.autonomy_r83.common import OUT, read, code, verify, TASKS
from VLA.autonomy_r83.runtime import Runtime


def main(a):
    import torch
    from VLA.flower_rotation_r76.sampler import Data, RECIPE
    from VLA.family_transfer_r54.smol import statistics
    from VLA.family_transfer_r54.data import anchors
    if not read(OUT / 'CPU_AUDIT.json')['passed']:
        raise ValueError('CPU data/split qualification missing')
    torch.set_num_threads(2)
    torch.manual_seed(202683)
    torch.cuda.set_per_process_memory_fraction(.84)
    a.output.mkdir(parents=True, exist_ok=False)
    bound = code()
    data = Data()
    data.horizon = 5 if a.family == 'smolvla' else 8
    stats = statistics(data) if a.family == 'smolvla' else data.stats
    runtime = Runtime(a.family, a.parent, stats, checkpoint=a.resume)
    params = runtime.params
    optimizer = torch.optim.AdamW(params.values(), lr=2.5e-5 if a.family == 'smolvla' else 5e-5,
        betas=(.9, .95), eps=1e-8, weight_decay=1e-8, foreach=False)
    rng = np.random.default_rng(202683)
    step = 0
    if a.resume:
        old = read(a.resume / 'ADAPTATION.json')
        verify(old['code'])
        if not old['completed'] or old['family'] != a.family or old['stats'] != stats:
            raise ValueError('Incompatible new-stage resume')
        if sha256_file(a.resume / 'optimizer.pt') != old['optimizer_sha256']:
            raise ValueError('Optimizer changed')
        saved = torch.load(a.resume / 'optimizer.pt', weights_only=True, map_location='cpu')
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        step = int(saved['step'])
        del saved
    if not step < a.updates <= 1536 or (a.updates == 2 and a.resume):
        raise ValueError('Finite stage budget must add actual optimizer updates')
    initial = {n: p.detach().cpu().clone() for n, p in params.items()} if a.updates == 2 else None
    selected = anchors(data)
    counts = Counter()
    feedback_gradients = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (a.output / 'updates.jsonl').open('x') as journal:
        while step < a.updates:
            verify(bound)
            optimizer.zero_grad(set_to_none=True)
            peak = 2.5e-5 if a.family == 'smolvla' else 5e-5
            lr = peak * min(1., (step + 1) / 64) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / 1536)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            losses = []
            for micro in range(4):
                ei, fi = selected[(4 * step + micro) % len(selected)] if a.updates == 2 else data.draw(rng)
                sample = runtime.sample(data, ei, fi)
                value = runtime.loss(sample)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite native masked training loss')
                (value / 4).backward()
                losses.append(float(value.detach()))
                counts[data.rows[ei]['task']] += 1
            grad = sum(float(p.grad.detach().abs().sum()) for p in params.values() if p.grad is not None)
            if not np.isfinite(grad) or grad <= 0:
                raise ValueError('No actual native gradient')
            if a.family == 'vla_adapter':
                feedback = runtime.native.projector.fc1.weight.grad
                feedback_grad = float(feedback[:, 8:].abs().sum()) if feedback is not None else 0.
            else:
                feedback_grad = sum(float(p.grad[:, :67].abs().sum()) for n, p in params.items()
                    if 'state_proj' in n and n.endswith('weight') and p.grad is not None)
            feedback_gradients.append(feedback_grad)
            torch.nn.utils.clip_grad_norm_(list(params.values()), 10., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            state = {'stage': 'actual_native_optimizer_updates', 'pid': __import__('os').getpid(),
                     'actual_optimizer_updates': step, 'target_updates': a.updates,
                     'loss': float(np.mean(losses)), 'LR': lr, 'elapsed_s': time.monotonic() - started}
            journal.write(__import__('json').dumps(state) + '\n')
            journal.flush()
            atomic_json(a.output / 'status.json', state)
    optimizer.zero_grad(set_to_none=True)
    path = runtime.save_weights(a.output)
    torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'numpy_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, a.output / 'optimizer.pt')
    changed = sum(not torch.equal(initial[n], p.detach().cpu()) for n, p in params.items()) if initial else None
    if initial is not None and (not changed or set(counts) != set(TASKS)):
        raise ValueError('Two-update native gate changed no weights or missed a task')
    if a.updates == 2 and (not np.isfinite(feedback_gradients).all() or max(feedback_gradients) <= 0):
        raise ValueError('Privileged numeric branch received no actual optimizer gradient')
    steps = {int(s['step']) for s in optimizer.state.values() if 'step' in s}
    if steps != {step} or any(v.dtype != torch.float32 for s in optimizer.state.values()
                             for k, v in s.items() if k in ('exp_avg', 'exp_avg_sq')):
        raise ValueError('Actual saved optimizer counters or moment precision differ')
    expected = [runtime.predict(runtime.sample(data, ei, fi), 202683 + i).tolist()
                for i, (ei, fi) in enumerate(selected)]
    atomic_json(a.output / 'RELOAD_ANCHORS.json', {'anchors': selected, 'predictions': expected, 'seed': 202683})
    atomic_json(a.output / 'ADAPTATION.json', {'family': a.family, 'variant': 'map3',
        'parent': str(a.parent.resolve()), 'parent_weights_unchanged_new_Adam_stage_declared': True,
        'source_resume': str(a.resume.resolve()) if a.resume else None,
        'stats': stats, 'completed': True, 'actual_new_optimizer_updates': step,
        'TRAIN_episodes': len(data.rows), 'TRAIN_frames': len(data), 'sampler': RECIPE,
        'batch_size': 1, 'accumulation': 4, 'native_horizon': runtime.horizon,
        'padding': 'native Smol action_is_pad; Adapter L1 uses real-timestep mask',
        'weights': str(path), 'weights_sha256': sha256_file(path),
        'optimizer_sha256': sha256_file(a.output / 'optimizer.pt'), 'code': bound,
        'changed_tensors_gate': changed, 'counts': dict(counts), 'actual_Adam_steps': sorted(steps),
        'numeric_branch_gradient_L1_max': max(feedback_gradients),
        'peak_GPU_MiB': torch.cuda.max_memory_allocated() / 2**20,
        'no_final_or_confirmation_used_in_gradients': True})


def reload(a):
    import torch
    from VLA.flower_rotation_r76.sampler import Data
    meta = read(a.checkpoint / 'ADAPTATION.json')
    verify(meta['code'])
    if sha256_file(meta['weights']) != meta['weights_sha256']:
        raise ValueError('Learned bytes changed')
    torch.set_num_threads(2)
    runtime = Runtime(meta['family'], meta['parent'], checkpoint=a.checkpoint)
    if meta['family'] == 'smolvla':
        from safetensors import safe_open
        actual = runtime.native.model.state_dict()
        with safe_open(meta['weights'], framework='pt', device='cpu') as saved:
            if set(saved.keys()) != set(actual):
                raise ValueError('Fresh Smol key set differs')
            for name in saved.keys():
                if not torch.equal(saved.get_tensor(name), actual[name].detach().cpu()):
                    raise ValueError('Fresh Smol weight not exact: ' + name)
    data = Data()
    data.horizon = runtime.horizon
    expected = read(a.checkpoint / 'RELOAD_ANCHORS.json')
    errors = []
    for i, ((ei, fi), value) in enumerate(zip(expected['anchors'], expected['predictions'], strict=True)):
        actual = runtime.predict(runtime.sample(data, ei, fi), expected['seed'] + i)
        errors.append(float(abs(actual - np.asarray(value, np.float32)).max()))
    if len(errors) != 8 or max(errors) > 1e-6:
        raise ValueError('Fresh native reload prediction mismatch')
    atomic_json(a.output, {'passed': True, 'checkpoint': str(a.checkpoint.resolve()),
        'actual_fresh_predictions': 8, 'max_errors': errors, 'weights_sha256': meta['weights_sha256']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='operation', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--family', choices=('smolvla', 'vla_adapter'), required=True)
    for name in ('parent', 'output'):
        fit.add_argument('--' + name, type=Path, required=True)
    fit.add_argument('--resume', type=Path)
    fit.add_argument('--updates', type=int, required=True)
    fresh = sub.add_parser('reload')
    fresh.add_argument('--checkpoint', type=Path, required=True)
    fresh.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args) if args.operation == 'fit' else reload(args)
