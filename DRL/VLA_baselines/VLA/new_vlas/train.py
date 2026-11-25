"""Bounded native-VLA fine-tuning with real optimizer and fresh-reload gates.

Gate mode uses all four actions AND both complete-source/corrective branches.
Long mode requires this family's own FP32 optimizer/save/fresh-reload proof.
No success-rate assertion follows from a training loss or a CUDA smoke test.
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.dataset import NativeData
from VLA.new_vlas.runtime import create_runtime

ROOT = Path(__file__).resolve().parents[2]
N = ROOT/'VLA/audits/new_vlas_r43_20261016'


def wait_gpu(out):
    from VLA.run_cross_vla120 import gpu_state
    until = time.monotonic()+3600
    while True:
        gpu = gpu_state()
        if not gpu['compute_pids'] and gpu['free_mib'] >= 14000:
            return
        if time.monotonic() > until:
            raise TimeoutError('Exclusive GPU slot unavailable; no other job interrupted')
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': 'running', 'stage': 'waiting_exclusive_GPU'})
        time.sleep(5)


def gate_anchors(data):
    anchors = []
    for task in sorted(data.groups):
        for kind in ('complete_original', 'fresh_corrective'):
            target = sorted(data.groups[task][kind])[0]
            ei = data.groups[task][kind][target][0]
            values = data.numeric(ei)['canonical7']
            moving = np.flatnonzero(np.linalg.norm(values[:, :3], axis=1) > .001)
            anchors.append((ei, int(moving[len(moving)//2]) if len(moving) else 0))
    return anchors


def save_checkpoint(out, runtime, optimizer, data, step, rng, args, probe):
    cp = out/'checkpoints'/f'step-{step:06d}'
    cp.mkdir(parents=True, exist_ok=False)
    runtime.save_trainables(cp/'trainable_masters.pt')
    torch.save(optimizer.state_dict(), cp/'optimizer.pt')
    torch.save({'CPU': torch.get_rng_state(), 'CUDA': torch.cuda.get_rng_state()}, cp/'torch_rng.pt')
    atomic_json(cp/'config.json', {'family': args.family, 'native_horizon': runtime.horizon,
        'LoRA_rank': runtime.rank, 'native_extension': runtime.extension,
        'actual_optimizer_updates': step, 'accumulation': args.accumulation,
        'learning_rate': args.learning_rate, 'schedule_total_updates': args.total_updates,
        'sampler_rng': rng.bit_generator.state, 'data_root': str(data.root),
        'data_audit_sha256': sha256_file(data.root/'audit.json'), 'stats': data.stats,
        'probe': {'episode': probe[0], 'frame': probe[1]},
        'model_input': 'two_real_RGB_original_task_native_proprio_plus_ALL67_causal_observations',
        'future_or_teacher_action_in_input': False, 'oracle_policy': False,
        'trainable_saved_precision': 'float32', 'compute': 'bfloat16_autocast',
        'own_full_source_numeric_gate_not_final_model_SR': args.mode == 'gate',
        'base_download_proof': str(N/('xvla_download' if args.family == 'xvla' else 'vla_adapter_download')/'source.json'),
        'code': {str(path): sha256_file(path) for path in sorted(Path(__file__).parent.glob('*.py'))}})
    prediction = runtime.predict(data.sample(*probe, args.family))
    np.save(cp/'reload_reference.npy', prediction)
    atomic_json(cp/'saved_values.json', {'trainable_master_sha256': sha256_file(cp/'trainable_masters.pt'),
        'reference_prediction_sha256': sha256_file(cp/'reload_reference.npy'),
        'optimizer_sha256': sha256_file(cp/'optimizer.pt'),
        'config_sha256': sha256_file(cp/'config.json'), 'shape': list(prediction.shape),
        'finite_prediction': bool(np.isfinite(prediction).all()), 'optimizer_updates': step})
    return cp


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage, step = 'checking_complete_source', 0
    started = time.monotonic()

    def status(state='running', **extra):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'family': args.family, 'actual_optimizer_updates': step,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'task_success_rate_established': False, **extra})

    try:
        torch.set_num_threads(2)
        status()
        data = NativeData(args.data)
        if args.mode == 'long':
            if not args.resume or not args.reload_proof:
                raise ValueError('Long fit requires own gate checkpoint and fresh-process reload proof')
            proof = json.loads(args.reload_proof.read_text())
            if (not proof.get('passed') or proof['family'] != args.family
                    or proof['checkpoint'] != str(args.resume.resolve())
                    or proof['data_audit_sha256'] != sha256_file(data.root/'audit.json')):
                raise ValueError('Own model/source/reload proof does not match')
        wait_gpu(out)
        stage = 'native_model_and_FP32_Adam_initialization'
        status()
        torch.manual_seed(202643)
        rng = np.random.default_rng(202643)
        runtime = create_runtime(args.family, data.stats)
        if args.resume:
            saved_cfg = json.loads((args.resume/'config.json').read_text())
            if (saved_cfg['family'] != args.family or saved_cfg['stats'] != data.stats
                    or saved_cfg['accumulation'] != args.accumulation
                    or saved_cfg['learning_rate'] != args.learning_rate):
                raise ValueError('Resume family/stats/effective batch/learning rate changed')
            runtime.load_trainables(args.resume/'trainable_masters.pt')
            step = saved_cfg['actual_optimizer_updates']
            rng.bit_generator.state = saved_cfg['sampler_rng']
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(14000*2**20/total_memory)
        runtime.to('cuda')
        trainable = runtime.parameters()
        if any(p.dtype != torch.float32 for p in trainable.values()):
            raise ValueError('Trainable master values must stay FP32')
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate,
            betas=(.9, .99), eps=1e-8, weight_decay=0., foreach=False)
        if args.resume:
            saved = json.loads((args.resume/'saved_values.json').read_text())
            if sha256_file(args.resume/'optimizer.pt') != saved['optimizer_sha256']:
                raise ValueError('Resume optimizer bytes changed')
            optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt', map_location='cpu',
                                                 weights_only=True, mmap=True))
            torch_rng = torch.load(args.resume/'torch_rng.pt', weights_only=True)
            torch.set_rng_state(torch_rng['CPU'])
            torch.cuda.set_rng_state(torch_rng['CUDA'])
        anchors = gate_anchors(data)
        probe = anchors[0]
        atomic_json(out/'plan.json', {'family': args.family, 'mode': args.mode,
            'actual_optimizer_updates_target': args.updates, 'resumed_updates': step,
            'accumulation': args.accumulation, 'microbatch': 1,
            'learning_rate': args.learning_rate, 'schedule_total_updates': args.total_updates,
            'warmup_updates': 200, 'TRAIN_episodes': len(data.rows), 'TRAIN_frames': data.proof['frames'],
            'sampling': 'equal_task_then20percent_corrective_then_uniform_target_episode_motion_weighted_frame',
            'native_horizon': runtime.horizon, 'LoRA_rank': runtime.rank,
            'trainable_parameters': sum(p.numel() for p in trainable.values()),
            'data_audit': str(data.root/'audit.json'), 'new_optimizer_not_microbatch_counter': True,
            'no_RL_or_simulator_code_edits': True, 'no_final_test_used_for_training': True})
        recent, seen, seen_tasks = deque(maxlen=100), set(), Counter()
        torch.cuda.reset_peak_memory_stats()
        stage = 'actual_native_optimizer_updates'
        status()
        with (out/'exposure.jsonl').open('x') as exposure, (out/'loss.jsonl').open('x') as loss_log:
            while step < args.updates:
                optimizer.zero_grad(set_to_none=True)
                lr_scale = .1+.9*min((step+1)/200, 1.)
                if step >= 200:
                    fraction = min((step-200)/max(args.total_updates-200, 1), 1.)
                    lr_scale = .1+.9*.5*(1+math.cos(math.pi*fraction))
                for group in optimizer.param_groups:
                    group['lr'] = args.learning_rate*lr_scale
                losses = []
                for micro in range(args.accumulation):
                    anchor = anchors[(step*args.accumulation+micro) % len(anchors)] if args.mode == 'gate' else data.draw(rng)
                    sample = data.sample(*anchor, args.family)
                    if args.mode == 'gate':
                        for role in ('front', 'wrist'):
                            Image.fromarray(sample[role]).save(out/f'gate_{step}_{micro}_{sample["task"]}_{role}.png')
                    loss, metrics = runtime.loss(sample)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite actual native supervised loss')
                    (loss/args.accumulation).backward()
                    losses.append(float(loss.detach()))
                    seen.add(anchor)
                    seen_tasks[sample['task']] += 1
                    exposure.write(json.dumps({'optimizer_update': step+1, 'micro': micro,
                        'episode': anchor[0], 'frame': anchor[1], 'task': sample['task'],
                        'kind': data.rows[anchor[0]]['source_kind']})+'\n')
                gradients = {n: p.grad for n, p in trainable.items() if p.grad is not None}
                if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
                    raise ValueError('Actual optimizer gradients absent or nonfinite')
                lora_b = {n: int(torch.count_nonzero(g)) for n, g in gradients.items() if 'lora_B' in n}
                if not any(lora_b.values()):
                    raise ValueError('The real native LoRA branch received no gradients')
                norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 10., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if any(not torch.isfinite(p).all() for p in trainable.values()):
                    raise ValueError('Nonfinite FP32 master after actual Adam update')
                adam_values = [v for state in optimizer.state.values() for key, v in state.items()
                               if key in ('exp_avg', 'exp_avg_sq')]
                if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in adam_values):
                    raise ValueError('Adam moments must remain finite FP32')
                mean_loss = float(np.mean(losses))
                recent.append(mean_loss)
                loss_log.write(json.dumps({'optimizer_update': step, 'native_loss': mean_loss,
                    'gradient_norm_before_clip': float(norm), 'lr': optimizer.param_groups[0]['lr']})+'\n')
                if step % 10 == 0 or args.mode == 'gate':
                    exposure.flush(); loss_log.flush()
                    status(mean_native_loss_last100=float(np.mean(recent)), unique_anchors=len(seen),
                        exposure_by_action=dict(seen_tasks), elapsed_seconds=time.monotonic()-started,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
                if step == args.updates or (args.mode == 'long' and step % 2000 == 0):
                    cp = save_checkpoint(out, runtime, optimizer, data, step, rng, args, probe)
                    atomic_json(out/'latest_checkpoint.json', {'checkpoint': str(cp), 'optimizer_updates': step})
        if args.mode == 'gate':
            if args.accumulation*args.updates < 8 or set(seen_tasks) != set(data.groups):
                raise ValueError('Gate did not exercise all4 tasks and2 source branches')
            if args.family == 'xvla':
                weights = runtime.model.model.transformer.action_encoder.fc.weight.detach().reshape(30, 139, -1)[:, 40:107]
            else:
                weights = runtime.projector.fc1.weight.detach()[:, 8:]
            if not torch.count_nonzero(weights):
                raise ValueError('Appended67 columns did not learn from actual Adam updates')
            atomic_json(out/'gate.json', {'passed': True, 'family': args.family,
                'actual_optimizer_updates': step, 'unique_full_native_anchors': len(seen),
                'all4_tasks_and_complete_and_corrective_sources': True,
                'FP32_trainable_masters_and_Adam_moments': True,
                'real_native_LoRA_gradient_nonzero_layers': sum(v > 0 for v in lora_b.values()),
                'appended67_weights_changed_from_zero': True, 'checkpoint': str(cp),
                'data_audit_sha256': sha256_file(data.root/'audit.json'),
                'peak_allocated_MiB_with_actual_Adam': torch.cuda.max_memory_allocated()/2**20,
                'fresh_process_reload_still_required_before_long_fit': True,
                'model_SR_established': False})
        stage = 'completed_actual_optimizer_work'
        status('completed', checkpoint=str(cp), elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        status('failed_preserve_evidence', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    parser.add_argument('--mode', choices=('gate', 'long'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--total-updates', type=int, default=6000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--reload-proof', type=Path)
    args = parser.parse_args()
    if args.mode == 'gate' and (args.updates != 2 or args.accumulation != 4):
        parser.error('Gate must execute2 actual Adam updates with4 accumulations each')
    main(args)
