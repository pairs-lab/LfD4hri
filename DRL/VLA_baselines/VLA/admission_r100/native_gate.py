"""Full authentic native optimizer and independent reload, never a toy proxy."""
import argparse
from pathlib import Path
import os
import time
import numpy as np

from .common import OUT,ROOT,ASSETS,read,atomic_json,utc,bindings
from VLA.common.provenance import sha256_file


def family_runtime(family,stats,checkpoint=None):
    if family=='seer':
        from .seer_native import Runtime
        return Runtime(stats,checkpoint=checkpoint)
    if family=='gr00t_n15':
        from .groot_native import Runtime
        return Runtime(stats,checkpoint=checkpoint)
    if family=='falcon_esm':
        from .falcon_native import Runtime
        return Runtime(stats,checkpoint=checkpoint)
    raise ValueError('Unknown authentic family')


def family_data(family):
    if family=='seer':
        from .seer_data import Data
    elif family=='gr00t_n15':
        from .groot_native import Data
    else:
        from .falcon_native import Data
    return Data()


def example(data,family,ei,fi):return data.native_sample(ei,fi) if family=='seer' else data.at(ei,fi)


def source_bindings(family):
    runtime_file={'seer':'seer_native.py','gr00t_n15':'groot_native.py','falcon_esm':'falcon_native.py'}[family]
    names=['common.py','native_gate.py','checkpoint_io.py',runtime_file]
    if family=='seer':names.append('seer_data.py')
    if family=='falcon_esm':names=['common.py','native_gate.py','checkpoint_io.py','falcon_native.py','falcon_depth.py']
    selected={str(Path(__file__).parent/name):sha256_file(Path(__file__).parent/name) for name in names}
    if family=='seer':
        folder=ROOT/'VLA/third_party/seer/models'
        selected.update({str(p):sha256_file(p) for p in folder.glob('*.py')})
    return selected


def main(a):
    import torch
    from .checkpoint_io import save_adaptation
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.manual_seed(2026100);root=OUT/a.family;root.mkdir(exist_ok=True)
    stats=read(root/'DATA_STATS.json');data=family_data(a.family)
    anchors=[]
    for task in ('touch','pick','move','pick_place'):
        ei=next(i for i,row in enumerate(data.rows) if row['task']==task and row['source_kind']=='complete_original')
        for fi in (0,min(data.rows[ei]['frames']-1,data.rows[ei]['frames']//2)):anchors.append((ei,fi))
    if a.op=='reload':
        meta=read(a.checkpoint/'TRAINING.json')
        if sha256_file(a.checkpoint/'weights.safetensors')!=meta['weights_sha256']:raise ValueError('Native gate weights changed')
        runtime=family_runtime(a.family,stats,checkpoint=a.checkpoint)
        refs=read(a.checkpoint/'RELOAD_ANCHORS.json');errors=[]
        for i,((ei,fi),wanted) in enumerate(zip(refs['anchors'],refs['predictions'],strict=True)):
            actual=runtime.predict(example(data,a.family,ei,fi),2026100+i)
            errors.append(float(np.max(abs(actual-np.asarray(wanted,np.float32)))))
        if len(errors)!=8 or max(errors)>1e-6:raise ValueError('Full native fresh CUDA reload mismatch: '+str(errors))
        atomic_json(a.checkpoint/'FRESH_RELOAD.json',{'passed':True,'UTC':utc(),'anchors':8,'max_errors':errors,
            'actual_full_native_model':True,'memory_allocated_MiB':torch.cuda.memory_allocated()/2**20})
        print({'passed':True,'errors':errors},flush=True);return
    if a.output.exists():raise FileExistsError('No overwriting a native gate attempt')
    a.output.mkdir(parents=True);started=time.monotonic()
    atomic_json(a.output/'status.json',{'UTC':utc(),'status':'loading_full_authentic_native_model',
        'family':a.family,'pid':os.getpid(),'memory_fraction':a.memory_fraction})
    runtime=family_runtime(a.family,stats);params=runtime.trainables
    optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=1e-4,foreach=False)
    names=[n for n in params if ('privileged67.2.weight' in n or 'arm_action_decoder.0.weight' in n
        or 'image_decoder_pred.weight' in n or 'state67.2.weight' in n or 'slot_W' in n
        or 'act_head.actions.mlp.6.weight' in n)]
    if len(names)<2:raise ValueError('Native and privileged gradient probes missing')
    probes={n:params[n].detach().cpu().clone() for n in names};gradients={n:[] for n in names}
    counts={task:0 for task in ('touch','pick','move','pick_place')};losses=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);parts=[]
        for task_index in range(4):
            ei,fi=anchors[2*task_index+step]
            out=runtime.loss(example(data,a.family,ei,fi));loss=out['loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite full native loss')
            (loss/4).backward();parts.append(float(loss.detach()));counts[data.rows[ei]['task']]+=1
        for n in names:
            grad=params[n].grad
            gradients[n].append(float(grad.abs().sum()) if grad is not None else 0.)
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);optimizer.step()
        losses.append(parts)
        atomic_json(a.output/'status.json',{'UTC':utc(),'status':'actual_native_optimizer_updates',
            'pid':os.getpid(),'updates':step+1,'losses':parts,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20})
    changed={n:not torch.equal(p,params[n].detach().cpu()) for n,p in probes.items()}
    if not all(changed.values()) or any(max(g)<=0 for g in gradients.values()):raise ValueError('Native/state/image gradients or actual weight changes failed')
    optimizer.zero_grad(set_to_none=True)
    predictions=[runtime.predict(example(data,a.family,ei,fi),2026100+i).tolist() for i,(ei,fi) in enumerate(anchors)]
    save_adaptation(runtime.model,a.output)
    torch.save({'optimizer':optimizer.state_dict(),'step':2,'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.default_rng(2026100).bit_generator.state},a.output/'optimizer.pt')
    atomic_json(a.output/'RELOAD_ANCHORS.json',{'anchors':anchors,'predictions':predictions})
    atomic_json(a.output/'TRAINING.json',{'completed':True,'UTC':utc(),'family':a.family,'updates':2,
        'effective_batch':4,'stats':stats,'code':source_bindings(a.family),'actual_full_native_model':True,
        'trainable_parameters':sum(p.numel() for p in params.values()),
        'total_parameters':sum(p.numel() for p in runtime.model.parameters()),'losses':losses,
        'retained_pretrained_projector_buffer_elements':sum(v.numel() for n,v in runtime.model.named_buffers() if n.endswith(('base_W','base_b'))),
        'checkpoint_format':'all trained native/privileged/LoRA parameters; authentic frozen bases referenced separately',
        'gradient_L1':gradients,'actual_probe_weights_changed':changed,'task_exposure':counts,
        'weights_sha256':sha256_file(a.output/'weights.safetensors'),'optimizer_sha256':sha256_file(a.output/'optimizer.pt'),
        'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,'elapsed_s':time.monotonic()-started,
        'native_load_proof':runtime.load_proof,'closed_loop_SR_NOT_implied':True})
    atomic_json(a.output/'status.json',{'UTC':utc(),'status':'native_two_update_gate_completed_requires_fresh_reload','updates':2})
    print({'completed_native_optimizer_updates':2,'changed':changed,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('seer','falcon_esm','gr00t_n15'),required=True)
    p.add_argument('--op',choices=('gate','reload'),required=True);p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--memory-fraction',type=float,default=.37)
    main(p.parse_args())
