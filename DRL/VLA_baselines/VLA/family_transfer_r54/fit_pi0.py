"""Real official OpenPI LoRA fit; source-local sampler and third view only."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import DATA, SCHEMA, read, hashes, verify
from .pi0 import PiData,statistics,build_config


def main(a):
    import jax
    if jax.default_backend()!='gpu':raise ValueError('Actual GPU required for a real pi0 fit')
    data=PiData(a.data);stats=statistics(data)
    audit=read(a.input_proof)
    if not audit.get('passed') or audit['model']!='pi0' or audit['data_audit_sha256']!=sha256_file(a.data/'audit.json'):
        raise ValueError('Actual own native CPU input proof required')
    verify(audit['code'])
    if a.updates>2:
        q=read(a.qualification)
        if (not q.get('passed') or q.get('model')!='pi0' or q.get('actual_optimizer_updates')!=2
            or q.get('actual_neural_predictions')!=8):raise ValueError('Own actual pi0 GPU/save/fresh reload8 required')
        verify(q['bound_files'])
    cfg=build_config(stats,a.output,a.updates)
    if cfg.checkpoint_dir.exists():raise FileExistsError('Preserve all official checkpoint directories')
    sidecar=cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_input_contract.json')
    contract={'model':'pi0','state_schema':SCHEMA,'action_schema':'native_EEF_delta_and_closure01_R54',
        'requested_updates':a.updates,'gradient_accumulation':4,'stats':stats,'source':str(Path(data.rows[0]['source']).parent),
        'data':str(a.data.resolve()),'data_audit_sha256':sha256_file(a.data/'audit.json'),'code':hashes(),
        'native_VLA_actions_only':True,'native_horizon':5,'added_optimizer_updates':a.updates,
        'new_optimizer_from_official_foundation_NOT_resume_claim':True,
        'long_fit_qualified_by_two_update_gate':a.updates>2,
        'input_proof':str(a.input_proof.resolve()),'input_proof_sha256':sha256_file(a.input_proof)}
    atomic_json(sidecar,contract)
    from openpi.training import data_loader
    data_loader.create_torch_dataset=lambda *_args,**_kw:data
    stock=data_loader.TorchDataLoader
    from .data import anchors
    gates=anchors(data);emitted=[]
    class Indices:
        def __iter__(self):
            rng=np.random.default_rng(202654)
            for index in range(len(data)):
                ei,fi=gates[index%len(gates)] if a.updates==2 else data.draw(rng)
                emitted.append((ei,fi))
                yield int((data.ends[ei-1] if ei else 0)+fi)
        def __len__(self):return len(data)
    def loader(dataset,*args,**kwargs):
        kwargs['shuffle']=False;kwargs['sampler']=Indices()
        return stock(dataset,*args,**kwargs)
    data_loader.TorchDataLoader=loader
    from VLA.pi0 import training
    assertion=training.assert_fp32_training_state
    def masters(state,configuration):
        from flax.traverse_util import flatten_dict
        checked=assertion(state,configuration);rows=[]
        for key,array in flatten_dict(state.params.filter(configuration.trainable_filter).to_pure_dict()).items():
            v=np.asarray(array);rows.append({'path':list(key),'shape':list(v.shape),'dtype':str(v.dtype),
                'initial_sha256':hashlib.sha256(v.tobytes()).hexdigest()})
        atomic_json(cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_master_init.json'),{**checked,'trainable_tensors':rows})
        return checked
    training.assert_fp32_training_state=masters
    # The official default uses random crop/rotation and colour jitter. Do
    # not geometrically/photometrically corrupt an absolute-coordinate GT map.
    # Disable only that image augmentation in THIS process, retaining the
    # official flow/noise/loss/LoRA/optimizer implementation.
    from openpi.models import model as model_api
    original_preprocess=model_api.preprocess_observation
    def no_image_augmentation(rng,observation,*,train=False,**kwargs):
        return original_preprocess(rng,observation,train=False,**kwargs)
    model_api.preprocess_observation=no_image_augmentation
    training.run_train(cfg)
    from collections import Counter
    atomic_json(a.output.with_suffix('.actual_sampling_receipt.json'),{'emitted_indices':len(emitted),
        'tasks':dict(Counter(data.rows[ei]['task'] for ei,fi in emitted)),
        'source_kinds':dict(Counter(data.rows[ei]['source_kind'] for ei,fi in emitted)),
        'actual_sampler_not_success_metric':True})
    from VLA.pi0.audit_scene32_saved import validate
    precision=a.output.with_suffix('.actual_saved_precision.json')
    proof=validate(cfg.checkpoint_dir,a.updates,precision)
    cp=Path(proof['checkpoint'])
    atomic_json(cp/'r54_contract.json',{**contract,'precision':str(precision.resolve()),
        'precision_sha256':sha256_file(precision)})
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'latest_checkpoint.json',{'checkpoint':str(cp)})
    atomic_json(a.output/'status.json',{'status':'completed','actual_optimizer_updates':a.updates,
        'native_flow_loss_official_OpenPI':True,'GPU_save_FP32_actual_counters_passed':True,'model_SR':None})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--input-proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,choices=(2,1500,3000),required=True);p.add_argument('--qualification',type=Path)
    main(p.parse_args())
