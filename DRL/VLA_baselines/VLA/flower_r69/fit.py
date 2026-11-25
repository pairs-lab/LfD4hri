"""Resource-bounded FLOWER fine-tuning after actual load/reload/robot gates.

This trainer never reads final800 outcomes. Calibrate before a long run; the
full official 30K-step/full-VLM recipe is NOT claimed for this LoRA variant.
"""
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data
from VLA.fit_feedback_r52 import SAMPLER
from VLA.flower_r69.native import load_foundation, adapt, native_batch, action_targets, canonical_actions
from VLA.flower_r69.loss import loss as masked_flow_loss
from VLA.flower_r69.gate import bindings, prediction, anchors


def read(p): return json.loads(Path(p).read_text())


def save(path, values):
    tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('wb') as stream:
        torch.save(values,stream)
    os.replace(tmp,path)


def main(a):
    if not 1<=a.accumulation<=32 or not 1<=a.updates<=4096:
        raise ValueError('Bounded explicit update count / accumulation required')
    if a.updates>128 and a.resume is None:
        raise ValueError('Long fit must resume a completed calibration, not bypass it')
    init=read(a.initial/'GATE2.json');fresh=read(a.reload_proof);robot=read(a.robot_proof)
    if not (init['passed'] and fresh['passed'] and fresh['fresh_process'] and robot['passed']
            and robot['actual_neural_controls']==8):
        raise ValueError('Own three completed actual gates are required')
    # Check core data/model code. A later test-only edit cannot silently change
    # the native forward, but input/model/label changes require rerunning gates.
    for path,digest in init['code'].items():
        if sha256_file(path)!=digest:
            raise ValueError('Gate binding changed; rerun qualification before fit: '+path)
    if robot['metadata']['checkpoint']!=str(a.initial.resolve()):
        raise ValueError('Robot gate used a different learned checkpoint')
    if sha256_file(a.initial/'trainables.pt')!=init['saved_trainables_sha256']:
        raise ValueError('Initial learned checkpoint changed')
    torch.set_num_threads(2);torch.manual_seed(202669)
    torch.cuda.set_per_process_memory_fraction(.79)
    a.output.mkdir(parents=True,exist_ok=False)
    code=bindings();started=time.monotonic();rng=np.random.default_rng(202669)
    data=Data(horizon=10)
    model,tokenizer,base=load_foundation();addition=adapt(model)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    source=a.initial if a.resume is None else a.resume
    if a.resume is not None:
        prior=read(source/'TRAINING.json')
        if not prior['completed'] or prior['actual_new_optimizer_updates']<128:
            raise ValueError('Long fit requires a completed128-update calibration')
        if prior['accumulation']!=a.accumulation:
            raise ValueError('Accumulation/protocol cannot silently change at resume')
        for path,digest in prior['code'].items():
            if sha256_file(path)!=digest:
                raise ValueError('Training code changed at resume: '+path)
        if sha256_file(source/'trainables.pt')!=prior['trainables_sha256']:
            raise ValueError('Calibration master file changed')
    learned=torch.load(source/'trainables.pt',weights_only=True,mmap=True,map_location='cpu')
    if set(learned)!=set(params):raise ValueError('Exact learned key set required')
    with torch.no_grad():
        for name,p in params.items():
            value=learned[name]
            if value.shape!=p.shape or value.dtype!=torch.float32:raise ValueError('Invalid saved master')
            p.copy_(value)
    del learned
    groups=[{'params':[p for n,p in params.items() if n.startswith('vlm.')],'lr':1e-5},
            {'params':[p for n,p in params.items() if not n.startswith('vlm.')],'lr':2e-5}]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=.05,foreach=False)
    begin=0;counts=Counter()
    if a.resume is not None:
        opt=torch.load(source/'optimizer.pt',weights_only=True,map_location='cpu')
        optimizer.load_state_dict(opt['optimizer']);rng.bit_generator.state=opt['numpy_rng']
        torch.set_rng_state(opt['torch_cpu_rng']);torch.cuda.set_rng_state(opt['torch_cuda_rng'])
        begin=int(opt['actual_new_optimizer_updates']);counts.update(opt['sampler_counts']);del opt
    if begin>=a.updates:raise ValueError('No new updates requested')
    plan={'model':'FLOWER_released12_block_privileged_LoRA8', 'source':str(source.resolve()),
        'source_gate_optimizer_updates':2,'target_new_updates':a.updates,'accumulation':a.accumulation,
        'total_schedule_updates':4096,'warmup_updates':128,'base':base,'adaptation':addition,
        'TRAIN_episodes':len(data.rows),'TRAIN_frames':len(data),'sampler':SAMPLER,
        'images':'front,wrist,GT_map224_CLIP_same_train_and_inference','numeric':'complete_causal67',
        'loss':'native_uniform_EEF7_RF_plus_explicit_temporal_pad_mask','paper_recipe_full_reproduction':False,
        'final800_outcomes_read':False,'code':code}
    atomic_json(a.output/'PLAN.json',plan)
    atomic_json(a.output/'stats.json',data.stats)
    journal=(a.output/'updates.jsonl').open('x');losses=[]
    def progress(stage,step,**kw):
        atomic_json(a.output/'status.json',{'stage':stage,'pid':os.getpid(),
            'actual_new_optimizer_updates':step,'target_new_updates':a.updates,
            'source_gate_optimizer_updates':2,'accumulation':a.accumulation,
            'sampler_counts':dict(counts),'elapsed_s':time.monotonic()-started,**kw})
    def snapshot(step):
        for path,digest in code.items():
            if sha256_file(path)!=digest:raise ValueError('Training source changed mid-run')
        save(a.output/'trainables.pt',{n:p.detach().cpu().contiguous().clone() for n,p in params.items()})
        save(a.output/'optimizer.pt',{'optimizer':optimizer.state_dict(),
            'actual_new_optimizer_updates':step,'numpy_rng':rng.bit_generator.state,
            'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state(),
            'sampler_counts':dict(counts)})
    try:
        progress('native_masked_loss_calibration_or_finetune',begin)
        for step in range(begin,a.updates):
            if (a.output/'USER_STOP_REQUESTED.json').exists():
                snapshot(step);progress('user_stop_requested_preserve_recovery',step);return
            warm=min(1.,(step+1)/128.)
            decay=.5+.5*math.cos(math.pi*max(0,step-128)/(4096-128))
            for group,peak in zip(optimizer.param_groups,(1e-5,2e-5)):
                group['lr']=peak*warm*(.5+.5*decay)
            optimizer.zero_grad(set_to_none=True);micros=[]
            for _ in range(a.accumulation):
                ei,fi=data.draw(rng);sample=data.at(ei,fi)
                batch=native_batch(sample,tokenizer,data.stats)
                target=torch.from_numpy(action_targets(sample['action'],data.stats))[None].to('cuda')
                pad=torch.from_numpy(sample['action_is_pad'])[None].to('cuda')
                model.train();model.vlm.eval()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cond=model.encode_observations(batch)
                    loss=masked_flow_loss(model,cond,target,pad)
                if not torch.isfinite(loss):raise ValueError('Nonfinite native loss; stop before long run')
                (loss/a.accumulation).backward();micros.append(float(loss.detach()))
                counts['task_'+sample['task']]+=1
                counts['kind_'+data.rows[ei]['source_kind']]+=1
                counts['real_target_steps']+=int((~sample['action_is_pad']).sum())
                counts['masked_nonexistent_tail_steps']+=int(sample['action_is_pad'].sum())
                del batch,cond,target,pad,loss
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(np.mean(micros)))
            row={'actual_new_optimizer_updates':step+1,'loss':losses[-1],
                'pre_clip_gradient_norm':float(norm),'LRs':[g['lr'] for g in optimizer.param_groups]}
            journal.write(json.dumps(row)+'\n');journal.flush()
            if (step+1)%8==0 or step==begin:
                progress('native_masked_loss_calibration_or_finetune',step+1,
                    mean_last32_RF_loss=float(np.mean(losses[-32:])),last=row)
            if (step+1)%512==0:snapshot(step+1)
        optimizer.zero_grad(set_to_none=True);snapshot(a.updates)
        # Predictions for independent fresh-process reload, not success-rate.
        selected=anchors(data)[::2];pred=[];diagnostics=[]
        for ei,fi in selected:
            sample=data.at(ei,fi);p=prediction(model,native_batch(sample,tokenizer,data.stats))
            pred.append(p);decoded=canonical_actions(p,data.stats);truth=sample['action']
            valid=~sample['action_is_pad']
            diagnostics.append({'task':sample['task'],'TRAIN_anchor_NOT_validation':True,
                'mean_XY_absolute_error_m':float(np.abs(decoded[valid,:2]-truth[valid,:2]).mean()),
                'mean_Z_absolute_error_m':float(np.abs(decoded[valid,2]-truth[valid,2]).mean()),
                'gripper_sign_accuracy':float(np.mean((decoded[valid,6]>.5)==(truth[valid,6]>.5)))})
        np.save(a.output/'PREDICTIONS.npy',np.stack(pred),allow_pickle=False)
        result={**plan,'completed':True,'actual_new_optimizer_updates':a.updates,
            'trainables_sha256':sha256_file(a.output/'trainables.pt'),
            'optimizer_sha256':sha256_file(a.output/'optimizer.pt'),
            'accumulation':a.accumulation,'sampler_counts':dict(counts),
            'loss_first32':float(np.mean(losses[:32])),'loss_last32':float(np.mean(losses[-32:])),
            'TRAIN_prediction_diagnostics':diagnostics,'anchors':selected,
            'robot_success_rate_measured':False,'elapsed_s':time.monotonic()-started,
            'peak_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
        atomic_json(a.output/'TRAINING.json',result);progress('completed_requires_fresh_reload_and_development_rollouts',a.updates)
        print(json.dumps({'completed_new_updates':a.updates,'loss_first32':result['loss_first32'],
            'loss_last32':result['loss_last32'],'measured_robot_SR':False}),flush=True)
    except Exception as e:
        progress('integration_error_NOT_model_success_rate',step if 'step' in locals() else begin,error=str(e));raise
    finally:
        journal.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--reload-proof',type=Path,required=True)
    p.add_argument('--robot-proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=128);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--resume',type=Path)
    main(p.parse_args())
