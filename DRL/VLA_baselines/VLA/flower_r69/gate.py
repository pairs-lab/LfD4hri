"""Bounded native FLOWER gate: exact load, 2 Adam updates, saved predictions.

Not a benchmark; no claimed robot success-rate. Fresh-process reload is a
separate command so a same-object in-memory roundtrip cannot pass as reload.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data
from VLA.flower_r69.native import (load_foundation, adapt, native_batch, action_targets,
                                   image_tensor, canonical_actions, SOURCE, ROOT)


def prediction(model, batch, seed=202669):
    model.eval()
    with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.manual_seed(seed)
        cond = model.encode_observations(batch)
        noise = torch.randn(1, 10, 7, device='cuda')
        return model.sample_actions(noise, cond, inference=True)[0].float().cpu().numpy()


def anchors(data):
    chosen = []
    for task in sorted(data.groups):
        for kind in ('complete_original', 'pose_corrective'):
            ei = next(i for i,r in enumerate(data.rows)
                      if r['task'] == task and r['source_kind'] == kind and r['frames'] >= 10)
            chosen.append((ei, 0))
    return chosen


def bindings():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [SOURCE / 'flower_vla/agents/flower.py', SOURCE / 'flower_vla/agents/networks/transformers.py',
              ROOT/'VLA/family_transfer_r54/data.py', ROOT/'VLA/fit_feedback_r52.py']
    return {str(p): sha256_file(p) for p in paths}


def main(a):
    torch.set_num_threads(2)
    torch.manual_seed(202669)
    torch.cuda.set_per_process_memory_fraction(.79, 0)
    torch.cuda.reset_peak_memory_stats()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    atomic_json(a.output/'status.json', {'stage': 'verify_complete_TRAIN_bank', 'pid': os.getpid()})
    data = Data(horizon=10, verify_rgb_hashes=False)
    chosen = anchors(data)
    if a.mode == 'reload':
        old = json.loads((a.checkpoint/'GATE2.json').read_text())
        for p,digest in old['code'].items():
            if sha256_file(p) != digest:
                raise ValueError('Gate source changed before fresh reload: '+p)
        chosen = [tuple(x) for x in old['anchors']]
    atomic_json(a.output/'status.json', {'stage': 'strict_native_foundation_load', 'pid': os.getpid()})
    model, tokenizer, foundation = load_foundation()
    atomic_json(a.output/'STRICT_LOAD.json', foundation)
    addition = adapt(model)
    params = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if a.mode == 'reload':
        if sha256_file(a.checkpoint/'trainables.pt') != old['saved_trainables_sha256']:
            raise ValueError('Saved trainable file changed')
        saved = torch.load(a.checkpoint/'trainables.pt', map_location='cpu', weights_only=True, mmap=True)
        if set(saved) != set(params):
            raise ValueError('Exact full trainable key set required')
        with torch.no_grad():
            for key,value in saved.items():
                if value.dtype != torch.float32 or value.shape != params[key].shape:
                    raise ValueError('Reload precision/shape differs')
                params[key].copy_(value)
                if not torch.equal(params[key].detach().cpu(), value):
                    raise ValueError('Saved trainable master reload not exact')
        errors = []
        expected = np.load(a.checkpoint/'PREDICTIONS.npy', allow_pickle=False)
        for index,(ei,fi) in enumerate(chosen[::2]):
            batch = native_batch(data.at(ei,fi), tokenizer, data.stats)
            actual = prediction(model, batch)
            errors.append(float(np.max(np.abs(actual-expected[index]))))
        result = {'passed': max(errors) == 0., 'fresh_process': True, 'actual_predictions': 4,
            'prediction_max_abs_errors': errors, 'all_saved_FP32_trainables_exact': True,
            'robot_success_rate_measured': False, 'code': bindings()}
        atomic_json(a.output/'FRESH_RELOAD.json', result)
        if not result['passed']:
            raise ValueError('Fresh-process native prediction differs')
        print(json.dumps(result), flush=True)
        return
    collage = Image.new('RGB', (3*224, 4*250), '#ffffff')
    draw = ImageDraw.Draw(collage)
    for row,(ei,fi) in enumerate(chosen[::2]):
        sample = data.at(ei,fi)
        for col,key in enumerate(('front','wrist','map')):
            collage.paste(Image.fromarray(sample[key]).resize((224,224)), (col*224,row*250+24))
            draw.text((col*224+4,row*250+5), sample['task']+' / '+key, fill='black')
    collage.save(a.output/'MODEL_INPUT_224_CONTACT_SHEET.png')
    groups = [{'params':[p for n,p in params.items() if n.startswith('vlm.')], 'lr':1e-5},
              {'params':[p for n,p in params.items() if not n.startswith('vlm.')], 'lr':2e-5}]
    optimizer = torch.optim.AdamW(groups, betas=(.9,.95), weight_decay=.05, foreach=False)
    receipts, updates = [], []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for ei,fi in chosen[step*4:step*4+4]:
            sample = data.at(ei,fi)
            if sample['action_is_pad'].any():
                raise ValueError('Gate deliberately uses real 10-step targets; no terminal padding')
            batch = native_batch(sample, tokenizer, data.stats)
            model.train()
            # Keep backbone dropout disabled without disabling gradients,
            # matching the small-data X-VLA adaptation rather than freezing it.
            model.vlm.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cond = model.encode_observations(batch)
                target = torch.from_numpy(action_targets(sample['action'], data.stats))[None].to('cuda')
                loss,_ = model.rf_loss(cond, target)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite native rectified-flow loss')
            (loss/4.).backward()
            receipts.append({'episode':ei, 'frame':fi, 'task':sample['task'], 'loss':float(loss.detach()),
                'native_input':model.last_input_shapes,
                'RGB_sha256':{k:hashlib.sha256(sample[k].tobytes()).hexdigest() for k in ('front','wrist','map')},
                'instruction':sample['instruction'], 'future_label_or_teacher_phase_in_input':False})
            atomic_json(a.output/'status.json', {'stage':'actual_backward', 'pid':os.getpid(),
                'completed_optimizer_updates':step, 'micro_batches':len(receipts), 'last':receipts[-1],
                'elapsed_s':time.monotonic()-started})
            del cond, loss, batch, target
        gradient = {label:sum(float(p.grad.detach().float().abs().sum()) for n,p in params.items()
            if predicate(n) and p.grad is not None) for label,predicate in {
                'VLM_LoRA':lambda n:n.startswith('vlm.'),
                'state67_projection':lambda n:n.startswith('proprio_encoders.eef_delta.'),
                'native_flow_core':lambda n:n.startswith('dit.')}.items()}
        if not all(np.isfinite(v) and v > 0 for v in gradient.values()):
            raise ValueError('Missing/invalid actual gradients: '+str(gradient))
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
        optimizer.step()
        updates.append({'actual_optimizer_updates':step+1, 'gradient_L1':gradient, 'pre_clip_grad_norm':float(norm)})
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    predictions = []
    for ei,fi in chosen[::2]:
        batch = native_batch(data.at(ei,fi), tokenizer, data.stats)
        predictions.append(prediction(model,batch))
    np.save(a.output/'PREDICTIONS.npy', np.stack(predictions), allow_pickle=False)
    # Consumption tests hold noise/model fixed; change one causal input only.
    sample = data.at(*chosen[0]); batch = native_batch(sample,tokenizer,data.stats)
    reference = prediction(model,batch)
    effect = {}
    for role,index in (('front',0),('wrist',1),('GT_map',2)):
        changed = {**batch,'images':batch['images'].clone()}
        changed['images'][:,index].zero_()
        effect[role] = float(np.max(np.abs(prediction(model,changed)-reference)))
    changed = {**batch,'numeric':batch['numeric'].clone()}; changed['numeric'].zero_()
    effect['state67'] = float(np.max(np.abs(prediction(model,changed)-reference)))
    if not all(v > 0 and np.isfinite(v) for v in effect.values()):
        raise ValueError('A supplied observation branch has no measurable effect: '+str(effect))
    # Learned masters only, foundation is referenced rather than duplicated.
    with (a.output/'trainables.pt').open('xb') as stream:
        torch.save({n:p.detach().cpu().contiguous().clone() for n,p in params.items()},stream)
    result = {'passed':True, 'actual_optimizer_updates':2, 'micro_batches':8,
        'all4_tasks':sorted({r['task'] for r in receipts}), 'TRAIN_episodes':len(data.rows), 'TRAIN_frames':len(data),
        'anchors':chosen, 'strict_native_load':foundation, 'adaptation':addition, 'updates':updates,
        'actual_input_receipts':receipts, 'fixed_noise_input_perturbation_max_abs_effect':effect,
        'peak_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'peak_GPU_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
        'elapsed_s':time.monotonic()-started, 'saved_trainables_sha256':sha256_file(a.output/'trainables.pt'),
        'code':bindings(), 'long_training_ready':False,
        'remaining_gates':['fresh_process_reload','same_PyBullet_neural8_safety','terminal_padding_loss_audit'],
        'model_success_rate_measured':False}
    atomic_json(a.output/'GATE2.json', result)
    atomic_json(a.output/'status.json', {'stage':'completed_two_update_gate_NOT_benchmark',
        'pid':os.getpid(),'actual_optimizer_updates':2,'long_training_started':False})
    print(json.dumps({k:v for k,v in result.items() if k not in ('actual_input_receipts','code')}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('fit2','reload'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    if args.mode == 'reload' and args.checkpoint is None:
        p.error('Reload requires the exact prior gate checkpoint')
    main(args)
