"""Native SmolVLA AdamW with explicit actual-update / FP32 / reload gates."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import DATA, SMOL, SCHEMA, hashes, read, checkpoint_contract, verify
from .data import Data, anchors, SAMPLER


def main(a):
    import torch
    from safetensors import safe_open
    from .smol import Runtime, statistics
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(13500*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(202654); np.random.seed(202654)
    data = Data(a.data); stat = statistics(data)
    parent = a.resume.resolve() if a.resume else SMOL
    origin = 0
    if a.mode=='long':
        if a.resume is None or a.reload_proof is None: raise ValueError('Own actual gate2 and fresh reload required')
        cfg=checkpoint_contract(parent); q=read(a.reload_proof)
        if (cfg['model']!='smolvla' or cfg['added_optimizer_updates']!=2 or not q.get('passed')
            or q.get('checkpoint')!=str(parent) or q.get('weights_sha256')!=sha256_file(parent/'model.safetensors')):
            raise ValueError('Actual matching two-update qualification required')
        origin=cfg['added_optimizer_updates']
        if cfg['stats']!=stat: raise ValueError('Warm normalization changed after native gate')
    elif a.updates!=2 or a.resume is not None: raise ValueError('Gate is exactly two actual NEW updates')
    if not origin < a.updates <= 2500: raise ValueError('Explicit finite added-update budget')
    runtime=Runtime(parent,stat)
    parameters=[p for p in runtime.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    rng=np.random.default_rng(202654)
    if origin:
        saved=torch.load(parent/'r54_training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    first={n:p.detach().cpu().clone() for n,p in runtime.model.named_parameters() if p.requires_grad} if a.mode=='gate' else None
    stage='actual_native_flow_fitting'
    def status(state='running',**extra):
        atomic_json(a.output/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
            'model':'smolvla','updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    try:
        admission_anchors=anchors(data)
        seen=set()
        for update in range(origin+1,a.updates+1):
            runtime.model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(4):
                position=admission_anchors[4*(update-1)+micro] if a.mode=='gate' else data.draw(rng)
                sample=data.at(*position);seen.add(sample['task'])
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=runtime.loss(sample)
                if not torch.isfinite(loss):raise ValueError('Native flow loss is nonfinite')
                (loss/4).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            optimizer.step()
            if update%10==0 or update==a.updates:
                status(added_optimizer_updates=update,actual_microbatches=4*update,loss=float(np.mean(losses)),gradient_norm=float(norm))
        steps={int(s['step']) for s in optimizer.state.values() if 'step' in s}
        if a.mode=='gate' and len(seen)!=4:raise ValueError('Actual two-update gate did not supervise all four tasks')
        if steps!={a.updates}:raise ValueError('Actual saved Adam counters differ from claimed updates')
        if any(v.dtype!=torch.float32 for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Actual saved Adam moments are not FP32')
        changed=sum(not torch.equal(first[n],p.detach().cpu()) for n,p in runtime.model.named_parameters() if n in first) if first else None
        if first is not None and not changed:raise ValueError('Actual native gate did not change any weights')
        cp=a.output/f'step-{a.updates:06d}';runtime.model.save_pretrained(cp)
        trainable=[n for n,p in runtime.model.named_parameters() if p.requires_grad]
        with safe_open(cp/'model.safetensors',framework='pt',device='cpu') as saved:
            for name in trainable:
                if saved.get_tensor(name).dtype!=torch.float32:raise ValueError('Learned native value downcast on save')
        torch.save({'optimizer':optimizer.state_dict(),'sampler_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},cp/'r54_training_state.pt')
        atomic_json(cp/'r54_contract.json',{'schema':SCHEMA,'model':'smolvla','data':str(a.data.resolve()),
            'data_audit_sha256':sha256_file(a.data/'audit.json'),'stats':stat,'code':hashes(),
            'native_VLA_actions_only':True,'observation':'two_RGB_native67_R52_bounded5_plus_R53_GT_image',
            'native_horizon':5,'added_optimizer_updates':a.updates,'optimizer_resumed_from_gate':bool(origin),
            'parent':str(parent),'sampler':SAMPLER,'changed_trainable_tensors_gate':changed,
            'actual_Adam_counters':sorted(steps),'success_rate_established':False})
        atomic_json(cp/'fp32_master_contract.json',{**read(SMOL/'fp32_master_contract.json'),
            'trainable_parameter_names':trainable,'saved_weights_and_adam_moments_fp32':True})
        predictions=[]
        for ei,fi in anchors(data): predictions.append(runtime.predict(data.at(ei,fi),seed=202654).tolist())
        atomic_json(cp/'reload_expected.json',{'anchors':anchors(data),'predictions':predictions,'seed':202654})
        atomic_json(a.output/'latest_checkpoint.json',{'checkpoint':str(cp.resolve())})
        stage='completed_native_flow_fit_REQUIRES_fresh_reload';status('completed',added_optimizer_updates=a.updates)
    except Exception as e:status('failed_preserve_evidence_NOT_zero_SR',error=f'{type(e).__name__}: {e}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('gate','long'),required=True);p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--reload-proof',type=Path)
    main(p.parse_args())
