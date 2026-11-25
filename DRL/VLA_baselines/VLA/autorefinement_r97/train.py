"""New disclosed low-LR phase; unchanged native architecture/loss/actions."""
import argparse
from collections import Counter
import gc
import math
import os
from pathlib import Path
import resource
import time

import numpy as np

from .common import OUT, PARENT, RECOVERY, bindings, read, verify, atomic_json
from VLA.common.provenance import sha256_file


def learning_rate(offset, horizon):
    warm = min(1., (offset+1)/16)
    return 1e-6 + 9e-6*warm*.5*(1+math.cos(math.pi*min(offset,horizon)/horizon))


def fit(a):
    import torch
    from safetensors.torch import save_model
    from VLA.dita_transfer_r84.native import Runtime
    from VLA.dita_transfer_r84.data import sample
    from VLA.recovery_quality_r90.data import Data
    from VLA.family_transfer_r54.data import anchors
    from VLA.continuous_r86.owner import qualified

    verify(read(OUT/'QUALIFICATION.json')['code'])
    meta = qualified(a.resume)
    if not 0 < a.additional_updates <= 512 or not 4 <= a.accumulation <= 32:
        raise ValueError('Only the declared short native continuation stages')
    if a.output.exists():
        raise FileExistsError('No overwriting an earlier training attempt')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.80)
    torch.manual_seed(202697)
    bound, started = bindings(), time.monotonic()
    stats = meta['stats']
    atomic_json(a.output/'status.json', {'status':'loading_native_resume', 'pid':os.getpid(),
        'updates':meta['updates'], 'target_updates':meta['updates']+a.additional_updates})
    runtime = Runtime(stats, checkpoint=a.resume/'weights.safetensors', device='cuda')
    params = runtime.trainables
    groups = []
    for dino in (False, True):
        for decay in (False, True):
            selected = [p for n,p in params.items() if ('native.image_tokenizer.tokenizer' in n)==dino
                and (p.ndim>1 and not n.endswith('.bias'))==decay]
            if selected:
                groups.append({'params':selected, 'lr':1e-6 if dino else 1e-5,
                    'lr_scale':.1 if dino else 1., 'weight_decay':.05 if decay else 0.})
    optimizer = torch.optim.AdamW(groups, betas=(.9,.95), foreach=False)
    saved = torch.load(a.resume/'optimizer.pt', weights_only=True, map_location='cpu')
    if saved['step'] != meta['updates']:
        raise ValueError('Optimizer/metadata update count mismatch')
    optimizer.load_state_dict(saved['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'])
    start = step = int(saved['step'])
    phase_origin = a.phase_origin if a.phase_origin is not None else start
    phase_horizon = a.phase_horizon if a.phase_horizon is not None else a.additional_updates
    if not phase_origin <= start < start+a.additional_updates <= phase_origin+phase_horizon:
        raise ValueError('Resume must stay within the original locked LR phase')
    del saved
    gc.collect()
    data = Data()
    selected = anchors(data)
    counts, targets, seen = Counter(), Counter(), set()
    probes = {n:p.detach().cpu().clone() for n,p in params.items()
        if n in ('sensor74.weight','native.transformer.lm_head.weight')}
    torch.cuda.reset_peak_memory_stats()
    max_gradient = 0.

    def commit(destination, final=False):
        destination.mkdir(exist_ok=False)
        optimizer.zero_grad(set_to_none=True)
        # Predictions fork RNG internally; checkpoint resume is exact.
        predictions = [runtime.predict(sample(data,ei,fi,stats),202697+i).tolist()
            for i,(ei,fi) in enumerate(selected)]
        save_model(runtime.model, str(destination/'weights.safetensors'))
        torch.save({'optimizer':optimizer.state_dict(), 'step':step,
            'numpy_rng':rng.bit_generator.state, 'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state()}, destination/'optimizer.pt')
        atomic_json(destination/'RELOAD_ANCHORS.json', {'anchors':selected,
            'predictions':predictions, 'seed':202697, 'device':'cuda'})
        changed = {n:not torch.equal(p,params[n].detach().cpu()) for n,p in probes.items()}
        atomic_json(destination/'TRAINING.json', {'completed':True, 'family':'dita', 'device':'cuda',
            'updates':step, 'effective_batch':a.accumulation, 'stats':stats, 'counts':dict(counts),
            'code':bound, 'parent_checkpoint':str(a.resume), 'parent_updates':start,
            'phase_updates':step-start, 'phase_completed':final,
            'R97_native_DDPM_loss_architecture_and_action_interface_unchanged':True,
            'R97_optimizer_moments_and_RNG_resumed':True,
            'R97_LR_extension_NOT_original_10k_recipe':{'floor':1e-6,'peak':1e-5,
                'horizon':phase_horizon,'origin_updates':phase_origin,'warmup':16,'DINO_multiplier':.1},
            'R90_actual_sampled_source_counts':dict(data.exposure),
            'target_counts_this_stage':dict(targets), 'unique_sampled_windows_this_stage':len(seen),
            'all_native_parameters_unfrozen_not_LoRA':True, 'changed_tensors':changed,
            'sensor_gradient_L1_max':max_gradient, 'recovery_admission':str(RECOVERY/'ADMISSION.json'),
            'original_normalizers_preserved':True, 'new_model_SR_implied':False,
            'weights_sha256':sha256_file(destination/'weights.safetensors'),
            'optimizer_sha256':sha256_file(destination/'optimizer.pt'),
            'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,
            'host_peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'elapsed_s':time.monotonic()-started})
        atomic_json(a.output/'LATEST_COMMITTED.json', {'checkpoint':str(destination), 'updates':step,
            'weights_sha256':sha256_file(destination/'weights.safetensors')})

    with (a.output/'updates.jsonl').open('x') as log:
        import json
        while step < start+a.additional_updates:
            verify(bound)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step-phase_origin, phase_horizon)
            for group in optimizer.param_groups:
                group['lr'] = lr*group['lr_scale']
            values = []
            for _ in range(a.accumulation):
                ei,fi = data.draw(rng)
                output = runtime.loss(sample(data,ei,fi,stats))
                loss = output['loss']
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite native DDPM loss')
                (loss/a.accumulation).backward()
                values.append(float(loss.detach()))
                row = data.rows[ei]
                counts[row['task']] += 1
                targets[row['task']+'/'+row['pick_name']] += 1
                seen.add((int(ei),int(fi)))
                del loss,output
            gradient = params['sensor74.weight'].grad
            magnitude = float(gradient.abs().sum()) if gradient is not None else 0.
            max_gradient = max(max_gradient,magnitude)
            torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
            optimizer.step()
            step += 1
            state = {'status':'actual_native_low_LR_finetune', 'pid':os.getpid(), 'device':'cuda',
                'updates':step, 'target_updates':start+a.additional_updates, 'phase_updates':step-start,
                'effective_batch':a.accumulation, 'learning_rate':lr, 'loss':float(np.mean(values)),
                'source_counts':dict(data.exposure), 'elapsed_s':time.monotonic()-started,
                'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,
                'host_peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
            log.write(json.dumps(state)+'\n'); log.flush()
            atomic_json(a.output/'status.json',state)
            if (step-start)%128 == 0 or step == start+a.additional_updates:
                commit(a.output/f'committed_{step}',final=step==start+a.additional_updates)
    if len(counts)!=4 or max_gradient<=0 or data.exposure['near_miss_corrective']<=0:
        raise ValueError('Actual four-task/native-gradient/recovery exposure gate failed')
    cp = Path(read(a.output/'LATEST_COMMITTED.json')['checkpoint'])
    if not all(read(cp/'TRAINING.json')['changed_tensors'].values()):
        raise ValueError('Both native and privileged encoder probes must actually update')
    atomic_json(a.output/'COMPLETED.json', {'checkpoint':str(cp), 'updates':step,
        'actual_additional_updates':a.additional_updates, 'actual_device':'cuda'})
    atomic_json(a.output/'status.json', {'status':'completed_requires_exact_CUDA_reload',
        'checkpoint':str(cp), 'updates':step, 'actual_additional_updates':a.additional_updates})


def reload(a):
    from VLA.dita_transfer_r84.train import reload as original_reload
    a.device = 'cuda'
    original_reload(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='op',required=True)
    f = sub.add_parser('fit')
    f.add_argument('--output',type=Path,required=True)
    f.add_argument('--resume',type=Path,required=True)
    f.add_argument('--additional-updates',type=int,required=True)
    f.add_argument('--accumulation',type=int,default=32)
    f.add_argument('--phase-origin',type=int)
    f.add_argument('--phase-horizon',type=int)
    r = sub.add_parser('reload')
    r.add_argument('--checkpoint',type=Path,required=True)
    a = p.parse_args()
    fit(a) if a.op=='fit' else reload(a)
