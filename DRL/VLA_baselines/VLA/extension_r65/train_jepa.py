"""Two real Adam updates, strict fresh reload, then finite native continuation."""
import argparse
import math
import os
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data, anchors
from .common import DATA, SCHEMA, OUT, BASE, sources, verify, read, utc, native_contract


def main(a):
    import torch
    from safetensors.torch import save_file
    from .jepa import Runtime, stats
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(13500*2**20/torch.cuda.get_device_properties(0).total_memory)
    input_proof=read(a.input_proof)
    if (not input_proof.get('passed') or input_proof['model']!='vla_jepa'
        or input_proof['data_audit_sha256']!=sha256_file(DATA/'audit.json')):
        raise ValueError('Own current native CPU input audit required')
    verify(input_proof['code'])
    origin = 0; data = Data(DATA,horizon=7); stat = stats(data)
    if a.resume:
        c = native_contract(a.resume); origin = c['actual_optimizer_updates']
        q = read(a.reload_proof)
        if origin != 2 or not q.get('passed') or q['checkpoint'] != str(a.resume.resolve()) or c['stats'] != stat:
            raise ValueError('Own two-update/fresh-reload proof required before continuation')
        robot=read(a.robot_proof)
        if not robot.get('passed') or robot.get('actual_native_controls')!=8:
            raise ValueError('Own actual neural8 simulator integration required')
    elif a.updates != 2:
        raise ValueError('Require actual two-update GPU gate first')
    if not origin < a.updates <= 2500:
        raise ValueError('Finite 2500 TOTAL added native updates')
    runtime = Runtime(stat,a.resume)
    params = list(runtime.trainable.values())
    opt = torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    rng = np.random.default_rng(202665)
    torch.manual_seed(202665)
    if origin:
        saved = torch.load(a.resume/'training_state.pt',map_location='cpu',weights_only=False)
        opt.load_state_dict(saved['optimizer']); rng.bit_generator.state = saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    first = {n:p.detach().cpu().clone() for n,p in runtime.trainable.items()} if not origin else None
    gate = anchors(data); seen=set()
    actual_update=origin
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'pid':os.getpid(),'model':'vla_jepa',
            'actual_optimizer_updates':actual_update,'UTC':utc(),**kw})
    try:
        for update in range(origin+1,a.updates+1):
            lr=1e-4*(.1+.9*min(update/100,1.)) if update<100 else 1e-5+9e-5*.5*(1+math.cos(math.pi*(update-100)/2400))
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True); losses=[]
            for micro in range(4):
                ei,fi = gate[4*(update-1)+micro] if not origin else data.draw(rng)
                sample = data.at(ei,fi); seen.add(sample['task'])
                # Native Qwen owns its bf16 autocast. The native DiT boundary
                # remains FP32 during both flow training and native prediction.
                loss=runtime.loss(sample)
                if not torch.isfinite(loss): raise ValueError('Native JEPA flow loss nonfinite')
                (loss/4).backward(); losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True); opt.step()
            actual_update=update
            if update%10==0 or update==a.updates:
                status(actual_optimizer_updates=update,actual_microbatches=4*update,
                    loss=float(np.mean(losses)),gradient_norm=float(norm),
                    GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
        counters={int(s['step']) for s in opt.state.values() if 'step' in s}
        if counters != {a.updates} or (not origin and len(seen)!=4):
            raise ValueError('Actual Adam counters/four-task gate mismatch')
        if any(v.dtype != torch.float32 for s in opt.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Native learned Adam moments not FP32')
        changed=sum(not torch.equal(first[n],p.detach().cpu()) for n,p in runtime.trainable.items()) if first else None
        if first is not None and changed==0: raise ValueError('Two actual optimizer updates changed no parameters')
        cp=a.output/f'step-{a.updates:06d}';cp.mkdir()
        save_file({n:p.detach().cpu().contiguous() for n,p in runtime.trainable.items()},str(cp/'learned.safetensors'))
        torch.save({'optimizer':opt.state_dict(),'sampler_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},cp/'training_state.pt')
        contract={'schema':SCHEMA,'model':'vla_jepa','actual_optimizer_updates':a.updates,
            'actual_neural_VLA_actions_only':True,'stats':stat,'code':sources(),
            'base':str(BASE),'base_weights_sha256':sha256_file(BASE/'model.safetensors'),
            'data_audit_sha256':sha256_file(DATA/'audit.json'),'native_horizon':7,
            'native_load_receipt':runtime.load_receipt,'actual_Adam_counters':sorted(counters),
            'FP32_learned_values_and_moments':True,'learned_sha256':sha256_file(cp/'learned.safetensors'),
            'optimizer_and_rng_resumed':bool(origin),'changed_tensors_in_gate':changed,
            'adaptation':'frozen pretrained Qwen; native DiT+state67 FP32; no world auxiliary loss',
            'action':'canonical world EEF delta7; no gripper flip, relative transform, Oracle or geometry repair'}
        atomic_json(cp/'r65_contract.json',contract)
        predictions=[runtime.predict(data.at(ei,fi),202665).tolist() for ei,fi in gate]
        atomic_json(cp/'reload_expected.json',{'seed':202665,'anchors':gate,'predictions':predictions})
        atomic_json(a.output/'latest_checkpoint.json',{'checkpoint':str(cp.resolve())})
        status('completed_REQUIRES_fresh_reload',actual_optimizer_updates=a.updates)
    except Exception as e:
        status('integration_error_NOT_model_zero_SR',error=f'{type(e).__name__}: {e}')
        raise


def reload(a):
    import torch
    from .jepa import Runtime
    c=native_contract(a.checkpoint); verify({str(BASE/'model.safetensors'):c['base_weights_sha256'],
        str(a.checkpoint/'learned.safetensors'):c['learned_sha256']})
    runtime=Runtime(c['stats'],a.checkpoint); data=Data(DATA,horizon=7); expected=read(a.checkpoint/'reload_expected.json'); errors=[]
    for (ei,fi),prediction in zip(expected['anchors'],expected['predictions'],strict=True):
        x=data.at(ei,fi);one=runtime.predict(x,expected['seed']);two=runtime.predict(x,expected['seed'])
        errors.extend([float(abs(one-two).max()),float(abs(one-np.asarray(prediction,np.float32)).max())])
    if len(errors)!=16 or max(errors)>1e-6:raise ValueError('Fresh reload/repeated native prediction changed')
    atomic_json(a.output,{'passed':True,'model':'vla_jepa','checkpoint':str(a.checkpoint.resolve()),
        'actual_optimizer_updates':c['actual_optimizer_updates'],'native_predictions':8,'repeat_predictions':8,
        'all_saved_learned_values_exact':True,'max_prediction_error':max(errors),
        'model_success_rate_established':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fit','reload'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--reload-proof',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--input-proof',type=Path);p.add_argument('--robot-proof',type=Path)
    a=p.parse_args();main(a) if a.mode=='fit' else reload(a)
