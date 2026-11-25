"""Actual optimizer gates -> exact reload -> native calibration/full fit."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.intern_r72.common import BASE,OUT,read,verify
from VLA.intern_r72.runtime_binding import bindings
from VLA.intern_r72.data import Data,Inputs,targets
from VLA.intern_r72.native import foundation,loss,prediction
from VLA.intern_r72.adapters import adapt
from VLA.target_balance_r74.sampler import draw as target_first_draw,RECIPE as TARGET_SAMPLER


def save(path,obj):
    temp=path.with_suffix(path.suffix+'.pending')
    with temp.open('wb') as f:torch.save(obj,f)
    os.replace(temp,path)


def examples(data):
    return [(next(i for i,r in enumerate(data.rows) if r['task']==t and r['source_kind']==kind),0)
            for kind in ('complete_original','pose_corrective') for t in ('touch','pick','move','pick_place')]


def main(a):
    if a.mode=='gate' and (a.updates!=2 or a.resume):raise ValueError('Gate must be2 actual fresh updates')
    if a.mode=='fit' and (a.resume is None or not 2<a.updates<=2048):raise ValueError('Bounded fit requires completed source gate/calibration')
    cpu=read(OUT/'cpu_preflight/CPU_PREFLIGHT.json');verify(cpu['code'])
    if not cpu['passed_input_label_and_native_weight_loading']:raise ValueError('CPU qualification incomplete')
    if sha256_file(BASE/'model.safetensors')!=cpu['foundation_sha256']:raise ValueError('Native foundation changed')
    torch.set_num_threads(2);torch.manual_seed(202672);torch.cuda.set_per_process_memory_fraction(.91)
    torch.cuda.reset_peak_memory_stats();a.output.mkdir(parents=True,exist_ok=False)
    code=bindings();started=time.monotonic();stage='load_complete_TRAIN_and_model';step=0
    def status(**kw):atomic_json(a.output/'status.json',{'stage':stage,'pid':os.getpid(),
        'completed_optimizer_updates':step,'target_updates':a.updates,'elapsed_s':time.monotonic()-started,**kw})
    try:
        status();data=Data();stats={**data.stats,'intern_action':data.intern_action_stats};inputs=Inputs(stats)
        model,base=foundation();addition=adapt(model,a.adaptation)
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        groups=[{'params':[p for n,p in params.items() if '.qwen3_5.' in n],'lr':1e-5},
                {'params':[p for n,p in params.items() if '.qwen3_5.' not in n],'lr':2e-5}]
        optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=.01,foreach=False)
        rng=np.random.default_rng(202672);counts=Counter();losses=[]
        if a.resume:
            old=read(a.resume/'TRAINING.json');verify(old['code'])
            if not old['completed'] or old['adaptation']['mode']!=a.adaptation:raise ValueError('Completed same-variant gate/resume required')
            if sha256_file(a.resume/'trainables.pt')!=old['trainables_sha256']:raise ValueError('Resume masters changed')
            saved=torch.load(a.resume/'trainables.pt',weights_only=True,mmap=True,map_location='cpu')
            if set(saved)!=set(params):raise ValueError('Resume keys differ')
            with torch.no_grad():
                for n,p in params.items():
                    if saved[n].dtype!=torch.float32 or saved[n].shape!=p.shape:raise ValueError('Resume shape/dtype differs')
                    p.copy_(saved[n])
            del saved
            opt=torch.load(a.resume/'optimizer.pt',weights_only=True,map_location='cpu')
            optimizer.load_state_dict(opt['optimizer']);rng.bit_generator.state=opt['numpy_rng']
            torch.set_rng_state(opt['torch_cpu_rng']);torch.cuda.set_rng_state(opt['torch_cuda_rng'])
            step=int(opt['step']);counts.update(opt['counts']);del opt
            if step>=a.updates:raise ValueError('No additional updates requested')
        chosen=examples(data);receipts=[];gradients=[]
        budget=read(OUT/'TOKEN_BUDGET_AUDIT.json')
        if not budget['passed']:raise ValueError('All-episode token audit incomplete')
        if a.mode=='gate':
            worst=budget['worst']
            slot=('touch','pick','move','pick_place').index(worst['task'])
            chosen[slot]=(worst['episode'],worst['frame'])
        plan={'model':'InternVLA-A1.5_native_MoT_flow_privileged67_LoRA', 'base':base,'adaptation':addition,
            'source':None if a.resume is None else str(a.resume.resolve()),'target_updates':a.updates,
            'accumulation':a.accumulation,'TRAIN_episodes':len(data.rows),'TRAIN_frames':len(data),
            'native_horizon50':True,'effective_labels7_native_head32_masked':True,
            'full_paper_recipe_NOT_reproduced':True,'no_WAN_aux_training':True,'sampler':TARGET_SAMPLER,
            'variable_real_horizon_FAST_tail_supervision':True,'no_final800_used':True,'code':code}
        atomic_json(a.output/'PLAN.json',plan);atomic_json(a.output/'stats.json',stats)
        def snapshot():
            verify(code)
            save(a.output/'trainables.pt',{n:p.detach().cpu().contiguous().clone() for n,p in params.items()})
            save(a.output/'optimizer.pt',{'optimizer':optimizer.state_dict(),'step':step,'counts':dict(counts),
                'numpy_rng':rng.bit_generator.state,'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state()})
        stage='actual_native_optimizer_updates';status()
        with (a.output/'updates.jsonl').open('x') as journal:
            while step<a.updates:
                verify(code);optimizer.zero_grad(set_to_none=True);micro=[]
                warm=min(1.,(step+1)/128.)
                decay=.5+.5*math.cos(math.pi*max(0,step-128)/(2048-128))
                for g,peak in zip(optimizer.param_groups,(1e-5,2e-5)):g['lr']=peak*warm*(.5+.5*decay)
                for j in range(a.accumulation):
                    ei,fi=chosen[step*4+j] if a.mode=='gate' else target_first_draw(data,rng)
                    sample=data.at(ei,fi);batch=inputs(sample,training=True,device='cuda')
                    model.train();model.qwen3_5_with_expert.qwen3_5.eval()
                    value,metrics=loss(model,batch,sample,stats)
                    if not torch.isfinite(value):raise ValueError('Nonfinite actual native loss')
                    (value/a.accumulation).backward();micro.append(float(value.detach()))
                    counts[sample['task']+'/'+data.last_sampling_mode if data.last_sampling_mode else sample['task']]+=1
                    if a.mode=='gate':receipts.append({'episode':ei,'frame':fi,'task':sample['task'],
                        **inputs.last_receipt,**metrics,'actual_loss':float(value.detach())})
                    del batch,value
                branch_grad={key:sum(float(p.grad.detach().abs().sum()) for n,p in params.items()
                    if pred(n) and p.grad is not None) for key,pred in {
                        'VLM_LoRA':lambda n:'.qwen3_5.' in n,
                        'state67':lambda n:n.startswith('state_proj.'),
                        'action_expert':lambda n:'.action_expert.' in n}.items()} if a.mode=='gate' else {}
                if a.mode=='gate' and not all(np.isfinite(v) and v>0 for v in branch_grad.values()):
                    raise ValueError('Missing actual native observation/expert gradients: '+str(branch_grad))
                gradients.append(branch_grad)
                norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
                optimizer.step();step+=1;losses.append(float(np.mean(micro)))
                row={'step':step,'loss':losses[-1],'pre_clip_grad_norm':float(norm),
                    'LRs':[g['lr'] for g in optimizer.param_groups],'elapsed_s':time.monotonic()-started}
                journal.write(json.dumps(row)+'\n');journal.flush()
                if step%4==0 or a.mode=='gate':status(last=row,mean_last32=float(np.mean(losses[-32:])))
                if step%256==0:snapshot()
        optimizer.zero_grad(set_to_none=True);snapshot()
        stage='saved_master_prediction_and_leakage_checks';status()
        predictions=[]
        for ei,fi in chosen[:4]:
            sample=data.at(ei,fi);batch=inputs(sample,training=False,device='cuda')
            predictions.append(prediction(model,batch))
        np.save(a.output/'PREDICTIONS.npy',np.stack(predictions),allow_pickle=False)
        leakage=None
        if a.mode=='gate':
            sample=data.at(*chosen[0]);batch=inputs(sample,training=True,device='cuda')
            model.eval();noise=torch.zeros((1,50,32),device='cuda');time_arg=torch.full((1,),.5,device='cuda')
            action=torch.from_numpy(targets(sample['action'],sample['action_is_pad'],stats['intern_action']))[None].cuda()
            changed={**batch,'lang_tokens':batch['lang_tokens'].clone()}
            ids=(changed['lang_tokens']>=248077)&(changed['lang_tokens']<=250124)
            changed['lang_tokens'][ids]=248077+(changed['lang_tokens'][ids]-248077+1)%2048
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                ref=model.forward(**batch,actions=action,noise=noise,time=time_arg)[0]
                alt=model.forward(**changed,actions=action,noise=noise,time=time_arg)[0]
            leakage=float((ref[:,:,:7]-alt[:,:,:7]).abs().max())
            if leakage>1e-4:raise ValueError('Action branch reads teacher FAST tokens: '+str(leakage))
        verify(code)
        result={**plan,'completed':True,'completed_optimizer_updates':step,'trainables_sha256':sha256_file(a.output/'trainables.pt'),
            'optimizer_sha256':sha256_file(a.output/'optimizer.pt'),'loss_first32':float(np.mean(losses[:32])),
            'loss_last32':float(np.mean(losses[-32:])),'anchors':chosen[:4],
            'peak_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'elapsed_s':time.monotonic()-started,'actual_input_receipts':receipts,'gradient_receipts':gradients if a.mode=='gate' else [],
            'teacher_FAST_perturbation_action_error_max_abs':leakage,'robot_SR_measured':False}
        if a.mode=='gate':result['actual_worst_audited_training_token_window_in_gate']=budget['worst']
        atomic_json(a.output/'TRAINING.json',result);stage='completed_requires_exact_fresh_reload';status()
    except Exception as error:
        status(error=str(error),integration_error_NOT_model_SR=True);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('gate','fit'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--adaptation',choices=('expert_full','expert_lora'),default='expert_full')
    p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.mode=='gate' and a.accumulation!=4:p.error('Gate is exactly8 micros across4tasks')
    main(a)
