"""Finite native flow fine-tuning with exact saved masters/Adam and receipts."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import save_file
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT, BASE, bindings, read, verify
from .data import Data, fit_stats, targets
from .native import Runtime


def anchors(data):
    result=[]
    for task in ('touch','pick','move','pick_place'):
        ei=next(i for i,r in enumerate(data.rows) if r['task']==task and r['source_kind']=='complete_original')
        result.append((ei,0))
    return result


def save_snapshot(runtime, optimizer, data, rng, output, plan, step, counts, losses, started):
    values={n:p.detach().cpu().contiguous() for n,p in runtime.trainable.items()}
    save_file(values,str(output/'learned.safetensors.pending'))
    os.replace(output/'learned.safetensors.pending',output/'learned.safetensors')
    with (output/'optimizer.pt.pending').open('wb') as handle:
        torch.save({'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
            'actual_optimizer_updates':step},handle)
    os.replace(output/'optimizer.pt.pending',output/'optimizer.pt')
    result={**plan,'completed':step==plan['target_optimizer_updates'],
        'actual_optimizer_updates':step,'learned_sha256':sha256_file(output/'learned.safetensors'),
        'optimizer_sha256':sha256_file(output/'optimizer.pt'),
        'loss_first16':float(np.mean(losses[:16])) if losses else None,
        'loss_last16':float(np.mean(losses[-16:])) if losses else None,
        'sampler_counts':dict(counts),'elapsed_s':time.monotonic()-started,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'robot_SR_measured':False}
    atomic_json(output/'TRAINING.json',result)
    return result


def main(a):
    torch.set_num_threads(2);torch.manual_seed(202680)
    torch.cuda.set_per_process_memory_fraction(.82)
    if not read(OUT/'DOWNLOAD.json')['passed']:raise ValueError('Verified download required')
    if sha256_file(BASE/'mp_rank_00_model_states.pt')!='1bc3abd53046648ef72970f338753fadf5acf3f444117a7e3b36e1ff93f71ef0':
        raise ValueError('Pinned official action checkpoint differs')
    a.output.mkdir(parents=True,exist_ok=False)
    code=bindings();verify(code)
    atomic_json(a.output/'status.json',{'stage':'auditing_complete_TRAIN_and_loading_native_model','pid':os.getpid()})
    data=Data();stats=fit_stats(data)
    if a.resume and read(a.resume/'stats.json')!=stats:raise ValueError('TRAIN normalization changed on resume')
    atomic_json(a.output/'stats.json',stats)
    runtime=Runtime(stats,a.resume)
    params=runtime.trainable
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in params.items() if n.startswith('embedder.')],'lr':1e-5,'peak':1e-5},
        {'params':[p for n,p in params.items() if not n.startswith('embedder.')],'lr':2e-5,'peak':2e-5},
    ],betas=(.9,.95),weight_decay=.001,foreach=False)
    rng=np.random.default_rng(202680);start=0
    if a.resume:
        parent=read(a.resume/'TRAINING.json');verify(parent['code'])
        if sha256_file(a.resume/'learned.safetensors')!=parent['learned_sha256']:raise ValueError('Saved weights changed')
        saved=torch.load(a.resume/'optimizer.pt',weights_only=True,map_location='cpu',mmap=True)
        optimizer.load_state_dict(saved['optimizer']);start=int(saved['actual_optimizer_updates'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    if start>=a.updates:raise ValueError('Target must extend the source actual optimizer count')
    plan={'model':'Evo-1_LIBERO_native_privileged_PEFT_R80','target_optimizer_updates':a.updates,
        'start_optimizer_updates':start,'schedule_updates':a.schedule_updates,'alignment_until':a.alignment_until,
        'accumulation':a.accumulation,'microbatch':1,'TRAIN_episodes':len(data.rows),'TRAIN_frames':len(data),
        'native_horizon':50,'active_action_channels':7,'normalization':'TRAIN bounds OPEN01',
        'native_state_dim':24,'state_dim_with_privileged':91,
        'temporal_and_channel_mask':True,'base':runtime.base,'adaptation':runtime.adaptation,
        'source':str(a.resume) if a.resume else None,'code':code,
        'DEV_confirmation_final_used_in_gradients':False,'completed':False}
    atomic_json(a.output/'PLAN.json',plan)
    counts=Counter();losses=[];started=time.monotonic()
    first={n:p.detach().cpu().clone() for n,p in params.items()} if a.updates==2 else None
    with (a.output/'updates.jsonl').open('x') as journal:
        for step in range(start,a.updates):
            verify(code);optimizer.zero_grad(set_to_none=True)
            joint=step>=a.alignment_until
            runtime.learning_mode(joint=joint)
            warm=min(1.,(step+1)/64)
            cosine=.1+.9*.5*(1+math.cos(math.pi*min(step,a.schedule_updates)/a.schedule_updates))
            for group in optimizer.param_groups:group['lr']=group['peak']*warm*cosine
            micro=[];feedback_grad=0.;vlm_grad=0.
            for _ in range(a.accumulation):
                ei,fi=data.draw(rng);sample=data.at(ei,fi)
                value=runtime.loss(sample)
                (value/a.accumulation).backward();micro.append(float(value.detach()))
                counts[sample['task']+'/'+data.last_sampling_mode]+=1
                del value
            gradient=runtime.model.action_head.state_encoder.fc1.linear.weight.grad
            if gradient is not None:feedback_grad=float(gradient[:,24:].abs().sum())
            for name,p in params.items():
                if name.startswith('embedder.') and p.grad is not None:vlm_grad+=float(p.grad.abs().sum())
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(np.mean(micro)))
            row={'actual_optimizer_updates':step+1,'target_updates':a.updates,'loss':losses[-1],
                'phase':'joint_PEFT' if joint else 'native_head_alignment',
                'gradient_norm':float(norm),'numeric67_projection_gradient_L1':feedback_grad,
                'VLM_LoRA_gradient_L1':vlm_grad,'elapsed_s':time.monotonic()-started,
                'pid':os.getpid(),'native_input':runtime.last_input}
            journal.write(json.dumps(row)+'\n');journal.flush()
            atomic_json(a.output/'status.json',{'stage':'running_actual_native_finetune',**row})
            print(json.dumps({k:row[k] for k in ('actual_optimizer_updates','loss','phase','elapsed_s')}),flush=True)
            if (step+1)%256==0:save_snapshot(runtime,optimizer,data,rng,a.output,plan,step+1,counts,losses,started)
    optimizer.zero_grad(set_to_none=True)
    result=save_snapshot(runtime,optimizer,data,rng,a.output,plan,a.updates,counts,losses,started)
    selected=anchors(data)
    runtime.capture_pixels=True
    predictions=[runtime.predict(data.at(ei,fi),202680+i) for i,(ei,fi) in enumerate(selected)]
    np.save(a.output/'PREDICTIONS.npy',np.stack(predictions),allow_pickle=False)
    atomic_json(a.output/'PREDICTION_ANCHORS.json',selected)
    from PIL import Image
    pixels=runtime.last_pixels
    if pixels is None:raise ValueError('Actual policy pixel audit missing')
    decoded=((pixels*torch.tensor([.229,.224,.225]).view(1,3,1,1)+
        torch.tensor([.485,.456,.406]).view(1,3,1,1))*255).round().clamp(0,255).byte()
    canvas=Image.new('RGB',(3*448,448))
    for i,pix in enumerate(decoded):canvas.paste(Image.fromarray(pix.permute(1,2,0).numpy()),(448*i,0))
    canvas.save(a.output/'ACTUAL_NATIVE_INPUT.png')
    if first is not None:
        changed=[n for n,p in params.items() if not torch.equal(first[n],p.detach().cpu())]
        tail=runtime.model.action_head.state_encoder.fc1.linear.weight[:,24:]
        opt_float=all(v.dtype==torch.float32 for s in optimizer.state.values() for k,v in s.items()
            if torch.is_tensor(v) and k in ('exp_avg','exp_avg_sq'))
        gate={'passed':bool(changed) and bool(tail.count_nonzero()) and opt_float,
              'changed_parameter_tensors':len(changed),'changed_VLM_LoRA_tensors':sum(n.startswith('embedder.') for n in changed),
              'numeric67_columns_learned':int(tail.count_nonzero()),'Adam_masters_FP32':opt_float,
              'base':runtime.base,'adaptation':runtime.adaptation,
              'actual_native_input':runtime.last_input,'actual_optimizer_updates':2,
              'peak_allocated_MiB':result['peak_allocated_MiB'],'peak_reserved_MiB':result['peak_reserved_MiB'],
              'robot_SR_measured':False}
        atomic_json(a.output/'GATE2.json',gate)
        if not gate['passed']:raise ValueError('Actual native optimizer gate failed')
    atomic_json(a.output/'status.json',{'stage':'completed_requires_fresh_reload','actual_optimizer_updates':a.updates,
        'elapsed_s':time.monotonic()-started,'pid':os.getpid()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--accumulation',type=int,default=16);p.add_argument('--alignment-until',type=int,default=128)
    p.add_argument('--schedule-updates',type=int,default=4096)
    main(p.parse_args())
