"""Finite genuine-native training, exact optimizer/RNG resumes and atomic commits."""
import argparse,gc,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
from .common import OUT,read,verify,atomic_json,utc
from .recipe import RECIPES,learning_rate
from .native_gate import family_runtime,family_data,example,source_bindings
from .checkpoint_io import save_adaptation
from VLA.common.provenance import sha256_file


def qualified(path,family):
    path=Path(path);meta=read(path/'TRAINING.json');verify(meta['code'])
    if not meta['completed'] or meta['family']!=family or not meta['actual_full_native_model']:
        raise ValueError('Wrong/incomplete actual native training record')
    for name in ('weights','optimizer'):
        file=path/('weights.safetensors' if name=='weights' else 'optimizer.pt')
        if sha256_file(file)!=meta[name+'_sha256']:raise ValueError('Checkpoint '+name+' changed')
    if meta['stats']!=read(OUT/family/'DATA_STATS.json'):raise ValueError('Frozen native TRAIN statistics changed')
    return meta


def main(a):
    import torch
    recipe=RECIPES[a.family];parent=qualified(a.resume,a.family)
    if not 1<=a.additional_updates<=recipe['initial_updates']:raise ValueError('Unbounded adaptation phase')
    if parent['updates']+a.additional_updates>2+recipe['maximum_updates']:raise ValueError('Total adaptation budget exceeded')
    if a.output.exists():raise FileExistsError('Retain existing training attempt, resume a committed checkpoint')
    a.output.mkdir(parents=True);bound={**source_bindings(a.family),
        **{str(Path(__file__).parent/n):sha256_file(Path(__file__).parent/n) for n in ('train.py','recipe.py')}}
    if not read(OUT/a.family/'DATA_AUDIT.json')['passed']:raise ValueError('Native data audit required')
    started=time.monotonic();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    atomic_json(a.output/'status.json',{'UTC':utc(),'status':'loading_authentic_native_resume','pid':os.getpid(),
        'updates':parent['updates'],'target_updates':parent['updates']+a.additional_updates})
    runtime=family_runtime(a.family,parent['stats'],a.resume);params=runtime.trainables
    optimizer=torch.optim.AdamW(params.values(),lr=recipe['lr'],weight_decay=recipe['weight_decay'],foreach=False)
    saved=torch.load(a.resume/'optimizer.pt',map_location='cpu',weights_only=True)
    if saved['step']!=parent['updates']:raise ValueError('Native optimizer step differs from metadata')
    optimizer.load_state_dict(saved['optimizer']);rng=np.random.default_rng();rng.bit_generator.state=saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    start=step=int(saved['step']);del saved;gc.collect()
    data=family_data(a.family);counts=Counter();targets=Counter();seen=set()
    refs=read(a.resume/'RELOAD_ANCHORS.json')['anchors']
    # Preserve both privileged and native-head probes, including Seer foresight.
    probe_names=list(parent.get('actual_probe_weights_changed',parent.get('probe_weights_changed',{})))
    probes={n:params[n].detach().cpu().clone() for n in probe_names}
    if len(probes)<2:raise ValueError('Native/privileged gradient gates absent from parent')
    gradients={n:0. for n in probes};torch.cuda.reset_peak_memory_stats()

    def commit(final=False):
        destination=a.output/f'committed_{step}';tmp=a.output/f'committing_{step}_{os.getpid()}'
        if destination.exists():raise FileExistsError('Cannot overwrite an immutable native commit')
        tmp.mkdir(exist_ok=False);optimizer.zero_grad(set_to_none=True)
        # Inference uses a forked RNG; checkpoint reference prediction cannot
        # change the subsequent training stream or draw future observations.
        predictions=[runtime.predict(example(data,a.family,ei,fi),2026100+i).tolist() for i,(ei,fi) in enumerate(refs)]
        save_adaptation(runtime.model,tmp)
        torch.save({'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},tmp/'optimizer.pt')
        atomic_json(tmp/'RELOAD_ANCHORS.json',{'anchors':refs,'predictions':predictions,'query_seed_base':2026100})
        changed={n:not torch.equal(v,params[n].detach().cpu()) for n,v in probes.items()}
        atomic_json(tmp/'TRAINING.json',{'completed':True,'UTC':utc(),'family':a.family,'updates':step,
            'actual_full_native_model':True,'stats':parent['stats'],'code':bound,'parent_checkpoint':str(a.resume),
            'phase_completed':final,'optimizer_moments_and_all_RNG_resumed':True,'recipe':recipe,
            'paper_native_losses_retained':True,'exact_original_paper_recipe_reproduced':False,
            'effective_batch':recipe['effective_batch'],'microbatch':1,'source_counts':dict(data.exposure),
            'rehearsal_counts':dict(data.stage_exposure),'task_exposure':dict(counts),'target_exposure':dict(targets),
            'unique_windows_this_phase':len(seen),'probe_weights_changed':changed,'gradient_L1_max':gradients,
            'trainable_parameters':sum(p.numel() for p in params.values()),
            'total_parameters':sum(p.numel() for p in runtime.model.parameters()),
            'retained_pretrained_projector_buffer_elements':sum(v.numel() for n,v in runtime.model.named_buffers() if n.endswith(('base_W','base_b'))),
            'native_load_proof':runtime.load_proof,'checkpoint_format':'all trained parameters; authentic frozen foundation referenced',
            'weights_sha256':sha256_file(tmp/'weights.safetensors'),'optimizer_sha256':sha256_file(tmp/'optimizer.pt'),
            'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,'elapsed_s':time.monotonic()-started,
            'closed_loop_SR_NOT_implied':True})
        # Files fully generated before the directory is published. Partial
        # directories remain diagnostics and are never eligible for resume.
        os.rename(tmp,destination)
        atomic_json(a.output/'LATEST_COMMITTED.json',{'checkpoint':str(destination),'updates':step})
        return destination

    with (a.output/'updates.jsonl').open('x') as stream:
        while step<start+a.additional_updates:
            verify(bound);optimizer.zero_grad(set_to_none=True);lr=learning_rate(step-2,recipe)
            for group in optimizer.param_groups:group.update(lr=lr,weight_decay=recipe['weight_decay'])
            values=[];components={}
            for _ in range(recipe['effective_batch']):
                ei,fi=data.draw(rng);result=runtime.loss(example(data,a.family,ei,fi));loss=result['loss']
                if not torch.isfinite(loss):raise ValueError('Nonfinite native training loss')
                (loss/recipe['effective_batch']).backward();values.append(float(loss.detach()))
                for n,v in result['loss_components'].items():components.setdefault(n,[]).append(float(v.detach()))
                counts[data.rows[ei]['task']]+=1;targets[data.rows[ei]['task']+'/'+data.rows[ei]['pick_name']]+=1
                seen.add((ei,fi));del loss,result
            for n in probes:
                g=params[n].grad;gradients[n]=max(gradients[n],float(g.abs().sum()) if g is not None else 0.)
            torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);optimizer.step();step+=1
            status={'UTC':utc(),'status':'actual_native_finetune','family':a.family,'pid':os.getpid(),
                'updates':step,'target_updates':start+a.additional_updates,'LR':lr,'loss':float(np.mean(values)),
                'native_components':{n:float(np.mean(v)) for n,v in components.items()},
                'effective_batch':recipe['effective_batch'],'elapsed_s':time.monotonic()-started,
                'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20}
            stream.write(json.dumps(status)+'\n');stream.flush();atomic_json(a.output/'status.json',status)
            if (step-2)%128==0 or step==start+a.additional_updates:commit(step==start+a.additional_updates)
    cp=Path(read(a.output/'LATEST_COMMITTED.json')['checkpoint']);meta=read(cp/'TRAINING.json')
    if len(counts)!=4 or not all(meta['probe_weights_changed'].values()) or any(v<=0 for v in gradients.values()):
        raise ValueError('Actual native four-task/privileged/probe training gate failed')
    atomic_json(a.output/'COMPLETED.json',{'UTC':utc(),'checkpoint':str(cp),'updates':step})
    atomic_json(a.output/'status.json',{'status':'completed_requires_fresh_reload','checkpoint':str(cp),'updates':step})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=tuple(RECIPES),required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--additional-updates',type=int,required=True);p.add_argument('--memory-fraction',type=float,default=.86)
    main(p.parse_args())
