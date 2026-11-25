"""Finite native full-model continuation with complete atomic resume points."""
import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import time
import numpy as np

from .common import OUT,PARENT,bindings,read,verify,atomic_json,utc,learning_rate
from VLA.common.provenance import sha256_file


def fit(a):
    import torch
    from safetensors.torch import save_model
    from bitsandbytes.optim import AdamW8bit
    from VLA.replacement_r84.native import Runtime
    from VLA.replacement_r84.data import sample
    from VLA.family_transfer_r54.data import anchors
    from VLA.continuous_r86.owner import qualified
    from .data import Data

    verify(read(OUT/'QUALIFICATION.json')['code'])
    if not read(OUT/'TRAIN_DATA_AUDIT.json')['passed']:
        raise ValueError('Actual TRAIN audit required')
    meta=qualified(a.resume)
    if meta['family']!='vlanext_s' or meta['optimizer']!='adamw8':raise ValueError('Wrong native optimizer family')
    if not 1<=a.additional_updates<=512 or a.accumulation not in (4,32):raise ValueError('Finite declared phase only')
    if a.output.exists():raise FileExistsError('Do not overwrite prior optimizer attempt')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.88)
    torch.manual_seed(202699)
    bound=bindings();stats=meta['stats'];started=time.monotonic()
    atomic_json(a.output/'status.json',{'status':'loading_native_full_resume','pid':os.getpid(),
        'updates':meta['updates'],'target_updates':meta['updates']+a.additional_updates,'UTC':utc()})
    runtime=Runtime(stats,checkpoint=a.resume/'weights.safetensors');params=runtime.trainables
    optimizer=AdamW8bit(params.values(),lr=2e-5,weight_decay=.1,betas=(.9,.999),eps=1e-8)
    saved=torch.load(a.resume/'optimizer.pt',map_location='cpu',weights_only=True)
    if saved['step']!=meta['updates']:raise ValueError('Optimizer/metadata step mismatch')
    optimizer.load_state_dict(saved['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    start=step=int(saved['step']);del saved;gc.collect()
    origin=a.phase_origin if a.phase_origin is not None else start
    horizon=a.phase_horizon if a.phase_horizon is not None else a.additional_updates
    if not origin<=start<start+a.additional_updates<=origin+horizon:raise ValueError('Locked LR horizon changed')
    data=Data();selected=anchors(data);counts=Counter();targets=Counter();seen=set()
    probes={n:params[n].detach().cpu().clone() for n in (
        'action_projector.weight','action_head.final_layer.linear.weight')}
    max_grad=0.;torch.cuda.reset_peak_memory_stats()

    def commit(final=False):
        destination=a.output/f'committed_{step}';destination.mkdir(exist_ok=False)
        optimizer.zero_grad(set_to_none=True)
        predictions=[runtime.predict(sample(data,ei,fi,stats),202699+i).tolist()
            for i,(ei,fi) in enumerate(selected)]
        save_model(runtime.model,str(destination/'weights.safetensors'))
        torch.save({'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},destination/'optimizer.pt')
        atomic_json(destination/'RELOAD_ANCHORS.json',{'anchors':selected,'seed':202699,'predictions':predictions})
        changed={n:not torch.equal(v,params[n].detach().cpu()) for n,v in probes.items()}
        atomic_json(destination/'TRAINING.json',{'completed':True,'UTC':utc(),'family':'vlanext_s',
            'updates':step,'effective_batch':a.accumulation,'stats':stats,'optimizer':'adamw8','code':bound,
            'parent_checkpoint':str(a.resume),'parent_updates':start,'phase_updates':step-start,
            'phase_completed':final,'R99_optimizer_moments_and_RNG_resumed':True,
            'full_native_architecture_and_trainability':True,'LoRA_or_frozen_backbone':False,
            'original_normalizers_preserved':True,'native20k128_recipe_reproduced':False,
            'R99_LR_extension':{'peak':2e-5,'floor':1e-6,'origin':origin,'horizon':horizon,'warmup':16},
            'native_flow_plus_DCT_loss_unchanged':True,'counts':dict(counts),'targets':dict(targets),
            'unique_windows':len(seen),'probe_weights_changed':changed,'numeric67_gradient_L1_max':max_grad,
            'R90_actual_sampled_source_counts':dict(data.exposure),'R99_rehearsal_counts':dict(data.stage_exposure),
            'parameter_count':sum(p.numel() for p in runtime.model.parameters()),
            'trainable_count':sum(p.numel() for p in params.values()),
            'parameter_dtypes':{str(dtype):sum(p.numel() for p in params.values() if p.dtype==dtype)
                for dtype in {p.dtype for p in params.values()}},
            'weights_sha256':sha256_file(destination/'weights.safetensors'),
            'optimizer_sha256':sha256_file(destination/'optimizer.pt'),
            'elapsed_s':time.monotonic()-started,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,
            'no_closed_loop_SR_implied':True})
        atomic_json(a.output/'LATEST_COMMITTED.json',{'checkpoint':str(destination),'updates':step})

    with (a.output/'updates.jsonl').open('x') as stream:
        import json
        while step<start+a.additional_updates:
            verify(bound);optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(step-origin,horizon)
            for group in optimizer.param_groups:group['lr']=lr
            values=[];components={}
            for _ in range(a.accumulation):
                ei,fi=data.draw(rng);s=sample(data,ei,fi,stats);output=runtime.loss(s);loss=output['loss']
                if not torch.isfinite(loss):raise ValueError('Nonfinite native flow loss')
                (loss/a.accumulation).backward();values.append(float(loss.detach()))
                for name,v in output['loss_components'].items():components.setdefault(name,[]).append(float(v.detach()))
                counts[data.rows[ei]['task']]+=1;targets[data.rows[ei]['task']+'/'+data.rows[ei]['pick_name']]+=1
                seen.add((ei,fi));del loss,output
            gradient=params['action_projector.weight'].grad
            max_grad=max(max_grad,float(gradient[:,7:].abs().sum()) if gradient is not None else 0.)
            torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            state={'status':'actual_native_R99_finetune','UTC':utc(),'pid':os.getpid(),
                'updates':step,'target_updates':start+a.additional_updates,'phase_updates':step-start,
                'effective_batch':a.accumulation,'loss':float(np.mean(values)),
                'native_components':{k:float(np.mean(v)) for k,v in components.items()},
                'LR':lr,'elapsed_s':time.monotonic()-started,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20}
            stream.write(json.dumps(state)+'\n');stream.flush();atomic_json(a.output/'status.json',state)
            if (step-origin)%128==0 or step==start+a.additional_updates:commit(step==origin+horizon)
    cp=Path(read(a.output/'LATEST_COMMITTED.json')['checkpoint'])
    if len(counts)!=4 or max_grad<=0 or not all(read(cp/'TRAINING.json')['probe_weights_changed'].values()):
        raise ValueError('Actual four-task gradient/weight-update gate failed')
    atomic_json(a.output/'COMPLETED.json',{'checkpoint':str(cp),'updates':step,'actual_additional_updates':step-start})
    atomic_json(a.output/'status.json',{'status':'completed_requires_exact_CUDA_reload','checkpoint':str(cp),'updates':step})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--additional-updates',type=int,required=True)
    p.add_argument('--accumulation',type=int,default=32);p.add_argument('--phase-origin',type=int)
    p.add_argument('--phase-horizon',type=int);fit(p.parse_args())
