"""Actual Orbax FP32 masters/moments/counters, not log/dirname estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def restore_selected(checkpointer,path,item):
    import jax
    import orbax.checkpoint as ocp
    # This PINNED Orbax uses transformations for a subtree restore; the
    # newer partial_restore keyword is not available in this environment.
    return checkpointer.restore(path,args=ocp.args.PyTreeRestore(item=item,
        restore_args=jax.tree.map(lambda _:ocp.ArrayRestoreArgs(restore_type=np.ndarray),item),transforms={}))


def validate(experiment,updates,output):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict,unflatten_dict
    if output.exists():raise FileExistsError('Keep each precision/value audit')
    contract_path=experiment.parent/(experiment.name+'.scene32_input_contract.json')
    master_path=experiment.parent/(experiment.name+'.scene32_master_init.json')
    contract=json.loads(contract_path.read_text());initial=json.loads(master_path.read_text())
    if contract['requested_updates']!=updates or contract['gradient_accumulation']!=4:raise ValueError('Actual recipe budget differs')
    candidates=sorted([p for p in experiment.iterdir() if p.name.isdigit()],key=lambda p:int(p.name))
    if not candidates:raise ValueError('No actual saved checkpoint')
    cp=candidates[-1].resolve();changed=0;values=0;weights=[]
    with ocp.PyTreeCheckpointer() as reader:
        sm=reader.metadata(cp/'train_state')
        floats=[v for v in jax.tree.leaves(sm) if hasattr(v,'dtype') and np.issubdtype(v.dtype,np.floating)]
        if not floats or any(v.dtype!=np.float32 for v in floats):raise ValueError('Saved optimizer floating state is not FP32')
        counters=restore_selected(reader,cp/'train_state',{'step':sm['step'],
            'opt_state':{k:sm['opt_state'][k] for k in ('gradient_step','mini_step')}})
        counters=jax.tree.map(lambda v:int(np.asarray(v).item()),counters)
        if (counters['step']!=4*updates or counters['opt_state']['gradient_step']!=updates
            or counters['opt_state']['mini_step']!=0):raise ValueError('Saved optimizer/accumulation counters differ')
        wm=reader.metadata(cp/'params');flat=flatten_dict(wm['params']);request={}
        for row in initial['trainable_tensors']:
            key=tuple(row['path']);saved_key=key+('value',) if key+('value',) in flat else key
            info=flat[saved_key]
            if info.dtype!=np.float32 or list(info.shape)!=row['shape']:raise ValueError('A saved learned master was downcast/reshaped')
            request[('params',*saved_key)]=info
        restored=restore_selected(reader,cp/'params',unflatten_dict(request));restored=flatten_dict(restored['params'])
        for row in initial['trainable_tensors']:
            key=tuple(row['path']);key=key+('value',) if key+('value',) in restored else key
            actual=np.asarray(restored[key]);digest=hashlib.sha256(actual.tobytes()).hexdigest()
            changed+=digest!=row['initial_sha256'];values+=actual.size
            weights.append({**row,'saved_sha256':digest})
    if changed==0:raise ValueError('Actual optimizer did not change any trainable parameter')
    proof={'passed':True,'model':contract['model'],'state_schema':contract['state_schema'],'checkpoint':str(cp),
        'actual_optimizer_updates':updates,'actual_microsteps':counters['step'],'saved_counters':counters,
        'saved_masters_and_moments_FP32':True,'trainable_values':values,'changed_trainable_tensors':changed,
        'trainable_tensors':weights,'input_contract':str(contract_path.resolve()),'input_contract_sha256':sha256_file(contract_path),
        'initial_master_proof_sha256':sha256_file(master_path),'neural_success_certified':False}
    atomic_json(output,proof);atomic_json(cp/'scene32_checkpoint_contract.json',
        {**contract,'actual_precision_proof':str(output.resolve()),'precision_proof_sha256':sha256_file(output)})
    source=Path(contract['source'])
    atomic_json(cp/'vla_checkpoint_manifest.json',{'source':str(source),
        'source_manifest_sha256':sha256_file(source/'dataset_manifest.json'),'state_schema':contract['state_schema'],
        'actual_optimizer_updates':updates,'privileged_inputs':True,'same_robot_objects_profile':True})
    return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.experiment.resolve(),a.updates,a.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!='trainable_tensors'}))


if __name__=='__main__':main()
