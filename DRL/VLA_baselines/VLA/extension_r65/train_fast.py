"""Actual official FAST LoRA optimizer; CPU proof and own reload gates required."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import anchors
from .common import DATA, SCHEMA, sources, read, verify, native_contract


def fit(a):
    import jax
    from .fast import FastData, build_config, statistics
    if jax.default_backend()!='gpu':raise ValueError('Own real GPU gate required')
    proof=read(a.input_proof)
    if not proof.get('passed') or proof['model']!='pi0_fast':raise ValueError('Own native input audit required')
    verify(proof['code'])
    if a.updates not in (2,1500):raise ValueError('Explicit gate2 then bounded1500 budget')
    if a.updates>2:
        q=read(a.reload_proof); c=native_contract(Path(q['checkpoint']))
        if not q.get('passed') or q['actual_optimizer_updates']!=2 or c['model']!='pi0_fast':
            raise ValueError('Own actual FAST2/fresh reload required')
        robot=read(a.robot_proof)
        if not robot.get('passed') or robot.get('actual_native_controls')!=8:
            raise ValueError('Own actual native8 simulator integration required')
    data=FastData(DATA);stat=statistics(data);cfg=build_config(stat,a.output,a.updates)
    if cfg.checkpoint_dir.exists():raise FileExistsError('Preserve native FAST runs')
    sidecar=cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_input_contract.json')
    contract={'model':'pi0_fast','state_schema':SCHEMA,'schema':SCHEMA,'stats':stat,
        'requested_updates':a.updates,'gradient_accumulation':4,'source':str(Path(data.rows[0]['source']).parent),
        'data':str(DATA),'data_audit_sha256':sha256_file(DATA/'audit.json'),'code':sources(),
        'native_VLA_actions_only':True,'native_horizon':32,
        'new_Adam_from_official_FAST_foundation_NOT_optimizer_resume':True}
    atomic_json(sidecar,contract)
    from openpi.training import data_loader
    data_loader.create_torch_dataset=lambda *_a,**_kw:data
    stock=data_loader.TorchDataLoader;gate=anchors(data)
    class Indices:
        def __iter__(self):
            rng=np.random.default_rng(202665)
            for index in range(len(data)):
                ei,fi=gate[index%8] if a.updates==2 else data.draw(rng)
                yield int((data.ends[ei-1] if ei else 0)+fi)
        def __len__(self):return len(data)
    def loader(dataset,*args,**kwargs):
        kwargs['shuffle']=False;kwargs['sampler']=Indices();return stock(dataset,*args,**kwargs)
    data_loader.TorchDataLoader=loader
    from VLA.pi0 import training
    assertion=training.assert_fp32_training_state
    def masters(state,config):
        from flax.traverse_util import flatten_dict
        result=assertion(state,config);rows=[]
        for key,array in flatten_dict(state.params.filter(config.trainable_filter).to_pure_dict()).items():
            v=np.asarray(array);rows.append({'path':list(key),'shape':list(v.shape),'dtype':str(v.dtype),
                'initial_sha256':hashlib.sha256(v.tobytes()).hexdigest()})
        atomic_json(cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_master_init.json'),{**result,'trainable_tensors':rows})
        return result
    training.assert_fp32_training_state=masters
    from openpi.models import model as model_api
    original=model_api.preprocess_observation
    def no_augmentation(rng,observation,*,train=False,**kwargs):
        return original(rng,observation,train=False,**kwargs)
    model_api.preprocess_observation=no_augmentation
    training.run_train(cfg)
    from VLA.pi0.audit_scene32_saved import validate
    precision=a.output.with_suffix('.precision.json');p=validate(cfg.checkpoint_dir,a.updates,precision);cp=Path(p['checkpoint'])
    atomic_json(cp/'r65_contract.json',{**contract,'actual_optimizer_updates':a.updates,
        'actual_neural_VLA_actions_only':True,'precision':str(precision.resolve()),'precision_sha256':sha256_file(precision)})
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'latest_checkpoint.json',{'checkpoint':str(cp)})


def reload(a):
    from .fast import Runtime,FastData
    c=native_contract(a.checkpoint);verify({c['precision']:c['precision_sha256']})
    runtime=Runtime(a.checkpoint);data=FastData(DATA);errors=[]
    for ei,fi in anchors(data):
        x=data.at(ei,fi);one=runtime.predict(x,202665);two=runtime.predict(x,202665)
        errors.append(float(abs(one-two).max()))
    if len(errors)!=8 or max(errors)>1e-6:raise ValueError('FAST actual decoded native8 repeatability failed')
    atomic_json(a.output,{'passed':True,'model':'pi0_fast','checkpoint':str(a.checkpoint.resolve()),
        'actual_optimizer_updates':c['actual_optimizer_updates'],'native_predictions':8,
        'additional_repeated_predictions':8,'all_saved_learned_values_exact':True,
        'strict_real_FAST_coefficients_NO_zero_fallback':True,'max_repeat_error':max(errors),
        'model_success_rate_established':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fit','reload'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int)
    p.add_argument('--input-proof',type=Path);p.add_argument('--reload-proof',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--robot-proof',type=Path)
    a=p.parse_args();fit(a) if a.mode=='fit' else reload(a)
