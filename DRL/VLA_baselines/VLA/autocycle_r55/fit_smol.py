"""Same native Smol flow, extended exact Adam/RNG accounting, no policy edits."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import os
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import checkpoint_contract,SMOL
from VLA.family_transfer_r54.data import Data,anchors
from .common import DATA,hashes,read
from .sampling import draw,contract,audit


def main(a):
    import torch
    from safetensors import safe_open
    from VLA.family_transfer_r54.smol import Runtime,statistics
    cfg=checkpoint_contract(a.resume);before=cfg['added_optimizer_updates']
    if cfg['model']!='smolvla' or not before<a.updates<=before+2000:raise ValueError('Finite own Smol continuation required')
    if a.mode=='gate' and a.updates-before!=2:raise ValueError('Gate must emit exactly2 NEW updates')
    if a.mode=='long':
        q=read(a.reload_proof)
        if (not q.get('passed') or q['checkpoint']!=str(a.resume.resolve())
            or q['weights_sha256']!=sha256_file(a.resume/'model.safetensors')
            or cfg.get('r55_new_stage_updates')!=2):raise ValueError('Own exact new2/fresh-reload proof required')
    a.output.mkdir(parents=True,exist_ok=False)
    data=Data(a.data)
    if statistics(data)!=cfg['stats']:raise ValueError('A continuation cannot silently change native normalizers')
    atomic_json(a.output/'sampling_audit.json',audit(data,a.sampling_profile))
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(13500*2**20/torch.cuda.get_device_properties(0).total_memory)
    runtime=Runtime(a.resume,cfg['stats']);params=[p for p in runtime.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    saved=torch.load(a.resume/'r54_training_state.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(saved['optimizer']);rng=np.random.default_rng();rng.bit_generator.state=saved['sampler_rng']
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    gates=anchors(data);exposure=Counter();seen=set();stage='actual_native_optimizer_continuation'
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
            'model':'smolvla','updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})
    try:
        for update in range(before+1,a.updates+1):
            runtime.model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(4):
                pos=gates[4*(update-before-1)+micro] if a.mode=='gate' else draw(data,rng,a.sampling_profile)
                sample=data.at(*pos);seen.add(sample['task']);exposure[sample['task']]+=1
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=runtime.loss(sample)
                if not torch.isfinite(loss):raise ValueError('Nonfinite native flow loss')
                (loss/4).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);optimizer.step()
            if update%10==0 or update==a.updates:status(actual_optimizer_updates=update,
                newly_emitted_optimizer_updates=update-before,loss=float(np.mean(losses)),gradient_norm=float(norm),
                actual_task_microbatch_exposure=dict(exposure))
        counters={int(s['step']) for s in optimizer.state.values() if 'step' in s}
        if counters!={a.updates}:raise ValueError('Actual Adam counters differ from claimed cumulative updates')
        if a.mode=='gate' and len(seen)!=4:raise ValueError('Native GPU gate did not train all4 tasks')
        if any(v.dtype!=torch.float32 or not torch.isfinite(v).all()
               for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Nonfinite or non-FP32 actual Adam moments')
        cp=a.output/f'step-{a.updates:06d}';runtime.model.save_pretrained(cp)
        names=[n for n,p in runtime.model.named_parameters() if p.requires_grad]
        with safe_open(cp/'model.safetensors',framework='pt',device='cpu') as source:
            for name in names:
                if source.get_tensor(name).dtype!=torch.float32:raise ValueError('Saved learned native tensor downcast')
        torch.save({'optimizer':optimizer.state_dict(),'sampler_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},cp/'r54_training_state.pt')
        atomic_json(cp/'r54_contract.json',{**cfg,'parent':str(a.resume.resolve()),'code':hashes(),
            'added_optimizer_updates':a.updates,'r55_cumulative_optimizer_updates':a.updates,
            'r55_new_stage_updates':a.updates-before,'r55_sampler':contract(a.sampling_profile),
            'r55_parent_training_state_sha256':sha256_file(a.resume/'r54_training_state.pt'),
            'r55_native_Adam_and_CPU_CUDA_sampler_RNG_resumed':True,
            'actual_Adam_counters':sorted(counters),'success_rate_established':False})
        atomic_json(cp/'fp32_master_contract.json',{**read(SMOL/'fp32_master_contract.json'),
            'trainable_parameter_names':names,'saved_weights_and_adam_moments_fp32':True})
        expected=[runtime.predict(data.at(ei,fi),202655).tolist() for ei,fi in gates]
        atomic_json(cp/'reload_expected.json',{'anchors':gates,'predictions':expected,'seed':202655})
        atomic_json(a.output/'latest_checkpoint.json',{'checkpoint':str(cp.resolve())})
        status('completed',actual_optimizer_updates=a.updates,newly_emitted_optimizer_updates=a.updates-before)
    except Exception as e:status('failed_preserve_evidence_NOT_zero_SR',error=f'{type(e).__name__}: {e}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','long'),required=True)
    for name in ('resume','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--updates',type=int,required=True)
    p.add_argument('--reload-proof',type=Path);p.add_argument('--sampling-profile',choices=('acquisition','transport'),required=True)
    main(p.parse_args())
