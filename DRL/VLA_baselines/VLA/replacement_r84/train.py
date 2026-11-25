"""Native full trainability gate, finite subsequent fits, exact optimizer/RNG resume."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,TASKS,bindings,verify,read,utc
from .data import Data,sample
from .native import Runtime

def fit(a):
    import torch
    from safetensors.torch import save_model
    from VLA.family_transfer_r54.data import anchors
    if not read(OUT/'VLANEXT_DATA_AUDIT.json')['passed_contracts']:
        raise ValueError('CPU contract gate required')
    torch.set_num_threads(2)
    torch.manual_seed(202684)
    torch.cuda.set_per_process_memory_fraction(.90)
    a.output.mkdir(parents=True,exist_ok=False)
    stats = read(OUT/'VLANEXT_DATA_AUDIT.json')['stats']
    bound = bindings()
    verify(read(OUT/'VLANEXT_FOUNDATION.json')['files'])
    started = time.monotonic()
    atomic_json(a.output/'status.json', {'status':'loading_actual_native_full_model','pid':os.getpid(),'UTC':utc()})
    runtime = Runtime(stats,checkpoint=a.resume/'weights.safetensors' if a.resume else None)
    params = runtime.trainables
    # Official AdamW structure and LR/decay; mixed parameter/moment dtypes are
    # explicitly recorded, not falsely advertised as FP32 masters or LoRA.
    if a.optimizer == 'adamw8':
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(params.values(),lr=1e-4,weight_decay=.1,betas=(.9,.999),eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=.1, foreach=False)
    atomic_json(a.output/'PARAMETER_AUDIT.json',{
        'parameter_count':sum(p.numel() for p in runtime.model.parameters()),
        'trainable_count':sum(p.numel() for p in params.values()),
        'per_top_module':{prefix:sum(p.numel() for n,p in params.items() if n.startswith(prefix))
            for prefix in ('lmm.','action_head.','action_projector.','meta_queries')},
        'parameter_dtypes':{str(dtype):sum(p.numel() for p in params.values() if p.dtype==dtype) for dtype in {p.dtype for p in params.values()}},
        'optimizer':a.optimizer,'full_unfrozen_not_LoRA':True})
    data = Data()
    selected = anchors(data)
    rng = np.random.default_rng(202684)
    step = 0
    if a.resume:
        meta = read(a.resume/'TRAINING.json')
        verify(meta['code'])
        if not meta['completed'] or meta['stats'] != stats or meta['optimizer']!=a.optimizer:
            raise ValueError('Resume contract mismatch')
        if sha256_file(a.resume/'optimizer.pt') != meta['optimizer_sha256']:
            raise ValueError('Optimizer integrity mismatch')
        saved = torch.load(a.resume/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        step = int(saved['step'])
        del saved
    if not step < a.updates <= 20000 or a.accumulation < 1:
        raise ValueError('Finite <= native20000 update recipe required')
    counts = {t:0 for t in TASKS}
    branch_grad = []
    probe_names = ['action_projector.weight','action_head.final_layer.linear.weight']
    probe_before = {n:params[n].detach().cpu().clone() for n in probe_names}
    torch.cuda.reset_peak_memory_stats()
    with (a.output/'updates.jsonl').open('x') as log:
        while step < a.updates:
            verify(bound)
            optimizer.zero_grad(set_to_none=True)
            lr = 1e-4*min(1.,(step+1)/1000)*.5*(1+math.cos(math.pi*step/20000))
            for group in optimizer.param_groups:
                group['lr'] = lr
            losses,components = [],{}
            for micro in range(a.accumulation):
                ei,fi = selected[(step*a.accumulation+micro)%len(selected)] if a.updates==4 else data.draw(rng)
                s = sample(data,ei,fi,stats)
                output = runtime.loss(s)
                value = output['loss']
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite native full loss')
                (value/a.accumulation).backward()
                losses.append(float(value.detach()))
                for name,v in output['loss_components'].items():
                    components.setdefault(name,[]).append(float(v.detach()))
                counts[data.rows[ei]['task']] += 1
                del value,output
            grad = params['action_projector.weight'].grad
            branch_grad.append(float(grad[:,7:].abs().sum()) if grad is not None else 0.)
            torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
            optimizer.step()
            step += 1
            state = {'status':'actual_native_full_optimizer_updates','pid':os.getpid(),'UTC':utc(),
                'updates':step,'target_updates':a.updates,'effective_batch':a.accumulation,
                'loss':float(np.mean(losses)), 'native_components':{k:float(np.mean(v)) for k,v in components.items()},
                'LR':lr,'numeric67_gradient_L1':branch_grad[-1],
                'elapsed_s':time.monotonic()-started,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20}
            log.write(json.dumps(state)+'\n');log.flush()
            atomic_json(a.output/'status.json',state)
    optimizer.zero_grad(set_to_none=True)
    changed = {n:not torch.equal(v,params[n].detach().cpu()) for n,v in probe_before.items()}
    if not all(changed.values()) or max(branch_grad)<=0 or not all(counts.values()):
        raise ValueError('Full four-task/privileged gradient/weight-change qualification failed')
    save_model(runtime.model,str(a.output/'weights.safetensors'))
    torch.save({'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},a.output/'optimizer.pt')
    predictions = [runtime.predict(sample(data,ei,fi,stats),202684+i).tolist() for i,(ei,fi) in enumerate(selected)]
    atomic_json(a.output/'RELOAD_ANCHORS.json', {'anchors':selected,'seed':202684,'predictions':predictions})
    atomic_json(a.output/'TRAINING.json', {'completed':True,'UTC':utc(),'family':'vlanext_s',
        'full_native_architecture_and_trainability':True,'LoRA_or_frozen_backbone':False,
        'native20k128_recipe_reproduced':a.updates==20000 and a.accumulation==128 and a.optimizer=='adamw32',
        'updates':step,'effective_batch':a.accumulation,'native_reference_batch':128,
        'stats':stats,'code':bound,'counts':counts,'probe_weights_changed':changed,
        'privileged_numeric_gradient_L1_max':max(branch_grad),
        'parameter_count':sum(p.numel() for p in runtime.model.parameters()),
        'trainable_count':sum(p.numel() for p in params.values()),
        'parameter_dtypes':{str(dtype):sum(p.numel() for p in params.values() if p.dtype==dtype) for dtype in {p.dtype for p in params.values()}},
        'optimizer':a.optimizer,
        'optimizer_moment_dtypes':sorted({str(v.dtype) for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq','state1','state2')}),
        'optimizer_recipe':'AdamW(lr1e-4,weight_decay0.1,clip1,warmup1000,cosine20000)',
        'optimizer_precision_differs_from_official':a.optimizer!='adamw32',
        'elapsed_s':time.monotonic()-started,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,
        'weights_sha256':sha256_file(a.output/'weights.safetensors'),
        'optimizer_sha256':sha256_file(a.output/'optimizer.pt'),
        'no_closed_loop_SR_implied':True})
    atomic_json(a.output/'status.json',{'status':'native_training_completed_requires_fresh_reload','UTC':utc(),'updates':step})

def reload(a):
    import torch
    torch.set_num_threads(2)
    meta = read(a.checkpoint/'TRAINING.json')
    verify(meta['code'])
    if sha256_file(a.checkpoint/'weights.safetensors') != meta['weights_sha256']:
        raise ValueError('Native learned weight changed')
    runtime = Runtime(meta['stats'],checkpoint=a.checkpoint/'weights.safetensors')
    data = Data()
    ref = read(a.checkpoint/'RELOAD_ANCHORS.json')
    errors=[]
    for i,((ei,fi),pred) in enumerate(zip(ref['anchors'],ref['predictions'],strict=True)):
        actual=runtime.predict(sample(data,ei,fi,meta['stats']),ref['seed']+i)
        errors.append(float(abs(actual-np.asarray(pred,np.float32)).max()))
    # BF16 kernels have finite precision, deterministic same-process seed checked.
    if len(errors)!=8 or max(errors)>1e-6:
        raise ValueError('Fresh native reload mismatch: '+str(errors))
    atomic_json(a.checkpoint/'FRESH_RELOAD.json',{'passed':True,'actual_anchors':8,'max_errors':errors,'UTC':utc()})

if __name__ == '__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='operation',required=True)
    f=sub.add_parser('fit');f.add_argument('--output',type=Path,required=True)
    f.add_argument('--updates',type=int,required=True);f.add_argument('--accumulation',type=int,default=4)
    f.add_argument('--resume',type=Path)
    f.add_argument('--optimizer',choices=('adamw32','adamw8'),default='adamw32')
    r=sub.add_parser('reload');r.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();fit(a) if a.operation=='fit' else reload(a)
