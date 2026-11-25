"""Full native Dita FT, original optimizer groups, exact stage/RNG/checkpoint resume."""
import argparse
from pathlib import Path
import json
import os
import time
import math
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,ROOT,BASE,TEXT,read,bindings,verify
from .data import sample
from .native import Runtime

def fit(a):
    import torch
    from safetensors.torch import save_model
    from VLA.replacement_r84.data import Data
    from VLA.family_transfer_r54.data import anchors
    if not read(OUT/'CPU_PREFLIGHT.json')['passed']:
        raise ValueError('Strict native CPU/action/history gate missing')
    verify(read(OUT/'TEXT_FOUNDATION.json')['files'])
    torch.set_num_threads(2);torch.manual_seed(2026842)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.80)
    a.output.mkdir(parents=True,exist_ok=False)
    bound=bindings();stats=read(ROOT/'VLA/audits/replacement_r84_20261020/VLANEXT_DATA_AUDIT.json')['stats']
    atomic_json(a.output/'status.json',{'status':'loading_actual_native_Dita','device':a.device,'pid':os.getpid()})
    started=time.monotonic();r=Runtime(stats,checkpoint=a.resume/'weights.safetensors' if a.resume else None,device=a.device)
    params=r.trainables
    groups=[]
    # Same native source group rule: DINO image tokenizer LR x0.1; all other
    # parameters LR x1; no decay for bias/norm. No newly frozen action backbone.
    for dino in (False,True):
        for decay in (False,True):
            selected=[p for n,p in params.items() if ('native.image_tokenizer.tokenizer' in n)==dino
                and (p.ndim>1 and not n.endswith('.bias'))==decay]
            if selected:groups.append({'params':selected,'lr':1e-5 if dino else 1e-4,
                'lr_scale':.1 if dino else 1.,'weight_decay':.05 if decay else 0.})
    opt=torch.optim.AdamW(groups,betas=(.9,.95),foreach=False)
    rng=np.random.default_rng(2026842);step=0
    if a.resume:
        meta=read(a.resume/'TRAINING.json');verify(meta['code'])
        if meta['stats']!=stats or not meta['completed']:raise ValueError('Dita resume mismatch')
        if sha256_file(a.resume/'optimizer.pt')!=meta['optimizer_sha256']:raise ValueError('Optimizer bytes changed')
        saved=torch.load(a.resume/'optimizer.pt',weights_only=True,map_location='cpu')
        opt.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng']);step=saved['step']
        if a.device=='cuda' and saved['cuda_rng'] is not None:torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    if not step<a.updates<=10000 or a.accumulation<1:raise ValueError('Finite 10k recipe required')
    data=Data();selected=anchors(data);counts={};seen=set();target_counts={};max_grad=0.
    probes={n:p.detach().cpu().clone() for n,p in params.items() if n in
        ('sensor74.weight','native.transformer.lm_head.weight')}
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    with (a.output/'updates.jsonl').open('x') as log:
        while step<a.updates:
            verify(bound);opt.zero_grad(set_to_none=True)
            warm=min(1.,(step+1)/100)
            multiplier=warm*(.01+.99*.5*(1+math.cos(math.pi*step/10000)))
            for g in opt.param_groups:g['lr']=1e-4*g['lr_scale']*multiplier
            values=[]
            for micro in range(a.accumulation):
                ei,fi=selected[(step*a.accumulation+micro)%len(selected)] if a.updates==2 else data.draw(rng)
                s=sample(data,ei,fi,stats)
                output=r.loss(s);v=output['loss']
                if not torch.isfinite(v):raise ValueError('Nonfinite actual native diffusion loss')
                (v/a.accumulation).backward();values.append(float(v.detach()))
                task=data.rows[ei]['task'];counts[task]=counts.get(task,0)+1;seen.add((int(ei),int(fi)))
                k=task+'/'+data.rows[ei]['pick_name'];target_counts[k]=target_counts.get(k,0)+1
                del v,output
            g=params['sensor74.weight'].grad
            norm=float(g.abs().sum()) if g is not None else 0.;max_grad=max(max_grad,norm)
            torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
            opt.step();step+=1
            state={'status':'actual_full_native_Dita_optimizer_updates','pid':os.getpid(),'device':a.device,
                'updates':step,'target_updates':a.updates,'effective_batch':a.accumulation,
                'loss':float(np.mean(values)),'sensor74_gradient_L1':norm,
                'unique_sampled_windows_this_stage':len(seen),'target_strata_seen_this_stage':len(target_counts),
                'elapsed_s':time.monotonic()-started}
            log.write(json.dumps(state)+'\n');log.flush();atomic_json(a.output/'status.json',state)
    opt.zero_grad(set_to_none=True)
    changed={n:not torch.equal(v,params[n].detach().cpu()) for n,v in probes.items()}
    if not all(changed.values()) or max_grad<=0 or len(counts)!=4:
        raise ValueError('Native/sensor gradients or four-task gate failed')
    save_model(r.model,str(a.output/'weights.safetensors'))
    torch.save({'optimizer':opt.state_dict(),'step':step,'numpy_rng':rng.bit_generator.state,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if a.device=='cuda' else None},a.output/'optimizer.pt')
    predictions=[r.predict(sample(data,ei,fi,stats),2026842+i).tolist() for i,(ei,fi) in enumerate(selected)]
    atomic_json(a.output/'RELOAD_ANCHORS.json',{'anchors':selected,'predictions':predictions,'seed':2026842,'device':a.device})
    atomic_json(a.output/'TRAINING.json',{'completed':True,'family':'dita','device':a.device,
        'updates':step,'effective_batch':a.accumulation,'stats':stats,'counts':counts,
        'target_counts_this_stage':target_counts,'unique_sampled_windows_this_stage':len(seen),
        'all_native_parameters_unfrozen_not_LoRA':True,'native_parameter_count':213225984,
        'new_sensor_parameters':57600,'frozen_original_CLIP_text_parameters':read(OUT/'TEXT_FOUNDATION.json')['frozen_text_parameters'],
        'optimizer':'FP32 AdamW, betas0.9/0.95, WD0.05 excluding bias/norm, DINO LR x0.1',
        'base_LR':1e-4,'cosine_horizon':10000,'warmup_updates':100,
        'native_10k32_budget_reached':step==10000 and a.accumulation==32,
        'explicit_budget_deviation':'target-domain warmup100 and full-bank balanced sampling, not native distributed dataloader/epoch schedule',
        'native_open01_gripper_and_DDPM_epsilon_loss':True,'changed_tensors':changed,
        'sensor_gradient_L1_max':max_grad,'code':bound,'text_foundation':read(OUT/'TEXT_FOUNDATION.json'),
        'weights_sha256':sha256_file(a.output/'weights.safetensors'),
        'optimizer_sha256':sha256_file(a.output/'optimizer.pt'),
        'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else None,
        'elapsed_s':time.monotonic()-started,'actual_robot_SR_implied':False})
    atomic_json(a.output/'status.json',{'status':'training_completed_requires_fresh_reload','updates':step,'device':a.device})

def reload(a):
    import torch
    from VLA.replacement_r84.data import Data
    torch.set_num_threads(2);meta=read(a.checkpoint/'TRAINING.json');verify(meta['code'])
    if sha256_file(a.checkpoint/'weights.safetensors')!=meta['weights_sha256']:raise ValueError('Native learned bytes changed')
    r=Runtime(meta['stats'],checkpoint=a.checkpoint/'weights.safetensors',device=a.device)
    data=Data();ref=read(a.checkpoint/'RELOAD_ANCHORS.json');errors=[]
    for i,((ei,fi),p) in enumerate(zip(ref['anchors'],ref['predictions'],strict=True)):
        actual=r.predict(sample(data,ei,fi,meta['stats']),ref['seed']+i)
        errors.append(float(abs(actual-np.asarray(p,np.float32)).max()))
    if len(errors)!=8 or max(errors)>1e-6:raise ValueError('Fresh same-device native reload mismatch '+str(errors))
    atomic_json(a.checkpoint/('FRESH_RELOAD_'+a.device+'.json'),{'passed':True,'device':a.device,'actual_anchors':8,'max_errors':errors})

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='op',required=True)
    f=sub.add_parser('fit');f.add_argument('--output',type=Path,required=True);f.add_argument('--updates',type=int,required=True)
    f.add_argument('--accumulation',type=int,default=32);f.add_argument('--resume',type=Path)
    f.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    r=sub.add_parser('reload');r.add_argument('--checkpoint',type=Path,required=True);r.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args();fit(a) if a.op=='fit' else reload(a)
