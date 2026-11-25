"""Native OpenPI LoRA with simulator feedback and shared EEF-delta labels.

The original joint-output experiment is preserved. This separate declared
action-space adaptation uses the SAME7 EEF commands as Smol/OpenVLA/CogACT,
reducing unnecessary joint inverse-mapping burden. Native neural heads still
choose every command; no teacher/controller runs during evaluation.
"""
import argparse
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import Scene64NativeDataset
from VLA.native_corrective_dataset import NativeCorrectiveDataset
from VLA.pi0.scene32_inputs import state32,statistics,SCHEMA
from VLA.pi0.scene32_training import build_config as _base_build_config

ACTION_SCHEMA='OpenPI_native7_shared_world_EEF_delta_plus_requested_closure_v1'


class MixedEEF:
    def __init__(self,projection,admission,model):
        self.horizon=32 if model=='pi0_fast' else 5
        self.original=Scene64NativeDataset(projection,self.horizon);self.correction=NativeCorrectiveDataset(admission,self.horizon)
        self.states=np.concatenate((self.original.states,self.correction.states));self.source=self.original.source
        self.original_length=len(self.original);self.pilot_only=False
    def __len__(self):return len(self.states)
    def __getitem__(self,index):
        raw=(self.original[index] if index<self.original_length else self.correction[index-self.original_length])
        raw=dict(raw);raw['index']=index;raw['actions']=raw['canonical_actions'].copy()
        raw['actions'][:,6]=(raw['canonical_actions'][:,6]+1)/2
        return raw
    def __getitems__(self,indices):return [self[int(i)] for i in indices]
    def weights(self):
        from collections import Counter
        counts=Counter();correction_counts=Counter()
        for r in self.original.rows:counts[r['task']['name'],r['task']['pick_name']]+=r['steps']
        targets=Counter(k[0] for k in counts)
        for r in self.correction.rows:correction_counts[r['task']['name']]+=r['steps']
        weights=[]
        for r in self.original.rows:
            t=r['task'];weights.extend([.8/(4*targets[t['name']]*counts[t['name'],t['pick_name']])]*r['steps'])
        for r in self.correction.rows:weights.extend([.2/(4*correction_counts[r['task']['name']])]*r['steps'])
        return weights


def stats(data):
    action=[]
    for component in (data.original,data.correction):
        for row in component.rows:
            with np.load(component.source/row['file'],allow_pickle=False) as z:values=z['canonical_actions'].astype(np.float64)
            values[:,6]=(values[:,6]+1)/2
            ids=np.minimum(np.arange(len(values))[:,None]+np.arange(data.horizon),len(values)-1)
            action.append(values[ids].reshape(-1,7))
    values=np.concatenate(action)
    result={k:np.asarray(v,np.float32).tolist() for k,v in [('mean',values.mean(0)),('std',values.std(0)),
        ('q01',np.quantile(values,.01,axis=0)),('q99',np.quantile(values,.99,axis=0))]}
    observed,contract=statistics(data.states)
    return {'state':observed,'actions':result},contract


def build_config(model,normalization,output,updates):
    from openpi import transforms
    from openpi.shared.normalize import NormStats
    from openpi.training import config,optimizer
    cfg=_base_build_config(model,normalization,output,2 if updates<=2 else min(1000,updates))
    if model=='pi0_fast':
        # Keep the OFFICIAL FAST32x32 coefficient layout. Only the first7
        # dimensions are physical commands; the other25 are zero padding.
        cfg=replace(cfg,model=replace(cfg.model,action_dim=32,action_horizon=32))
    norm={k:NormStats(**{n:np.asarray(v,np.float32) for n,v in item.items()}) for k,item in normalization.items()}
    @dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self,raw):
            state=state32(raw['pre_action_scene64'])
            if not np.allclose(state[:7],np.r_[raw['joints'],raw['gripper']],rtol=0,atol=1e-6):raise ValueError('Current observation mismatch')
            if model=='pi0_fast':
                image={'base_0_rgb':raw['image'],'base_1_rgb':np.zeros_like(raw['image']),'wrist_0_rgb':raw['wrist_image']}
                mask={'base_0_rgb':np.True_,'base_1_rgb':np.False_,'wrist_0_rgb':np.True_}
            else:image={'base_0_rgb':raw['image'],'left_wrist_0_rgb':raw['wrist_image']};mask={k:np.True_ for k in image}
            result={'state':state,'image':image,'image_mask':mask,'prompt':raw['prompt']}
            if 'actions' in raw:result['actions']=raw['actions'].copy()
            return result
    @dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self,raw):return {'actions':np.asarray(raw['actions'])[...,:7]}
    @dataclass(frozen=True)
    class PadActions(transforms.DataTransformFn):
        def __call__(self,raw):
            if 'actions' in raw:
                values=np.asarray(raw['actions'])
                if values.shape!=(32,7):raise ValueError('Physical FAST labels must be32x7 before padding')
                raw={**raw,'actions':np.pad(values,((0,0),(0,25)))}
            return raw
    @dataclass(frozen=True)
    class Factory(config.DataConfigFactory):
        def create(self,assets_dirs,model_config):
            native_transforms=config.ModelTransformFactory()(model_config)
            if model=='pi0_fast':native_transforms=transforms.Group(inputs=[PadActions(),*native_transforms.inputs],outputs=native_transforms.outputs)
            return config.DataConfig(repo_id=self.repo_id,asset_id='scene32_eef_ur5e',norm_stats=norm,
                use_quantile_norm=model=='pi0_fast',data_transforms=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]),
                model_transforms=native_transforms)
    # These labels are ALREADY EEF DELTAS. Subtracting observed joint angles
    # or applying AbsoluteActions would be an actual train/deploy unit bug.
    return replace(cfg,name=model+'_scene32_EEF_recovery',data=Factory(repo_id='local/'+model+'_simscene32_eef'),
        num_train_steps=updates*4,save_interval=min(updates,250)*4,keep_period=min(updates,250)*4,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=0 if updates<=2 else 100,decay_steps=max(updates,3),peak_lr=2.5e-5,decay_lr=2.5e-6),
        policy_metadata={**cfg.policy_metadata,'action_schema':ACTION_SCHEMA,'source_images_unchanged':True})


def audit(data,model,normalization,output):
    import copy
    from openpi import transforms
    from openpi.models.model import Observation
    from openpi.training import data_loader
    cfg=build_config(model,normalization,output,2);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    native=data_loader.transform_dataset(data,dc)
    indices=np.linspace(0,data.original_length-1,32,dtype=int).tolist()
    for ei,row in enumerate(data.correction.rows):
        start=data.correction.ends[ei-1] if ei else 0
        indices.extend(data.original_length+start+fi for fi in (0,row['steps']//2,row['steps']-1))
    reports=[];maximum=0
    loader=data_loader.TorchDataLoader(native,local_batch_size=2,sampler=indices,num_batches=len(indices)//2,num_workers=0,framework='pytorch')
    for number,batch in enumerate(loader):
        for bi,index in enumerate(indices[2*number:2*number+2]):
            raw=data[index];actual=native[index]
            for field in ('state','actions','tokenized_prompt','tokenized_prompt_mask'):
                np.testing.assert_array_equal(np.asarray(batch[field])[bi],actual[field])
            typed=copy.deepcopy(actual);typed['image_mask']={k:np.asarray(v) for k,v in typed['image_mask'].items()}
            Observation.from_dict(typed)
            expected=raw['canonical_actions'].copy();expected[:,6]=(expected[:,6]+1)/2
            np.testing.assert_array_equal(raw['actions'],expected)
            before=dc.data_transforms.inputs[0](raw)
            np.testing.assert_array_equal(before['actions'],expected)
            np.testing.assert_array_equal(before['state'],state32(raw['pre_action_scene64']))
            no_action={k:v for k,v in raw.items() if k!='actions'}
            inferred=transforms.compose([*dc.data_transforms.inputs,transforms.Normalize(dc.norm_stats,use_quantiles=dc.use_quantile_norm),*dc.model_transforms.inputs])(no_action)
            n=int(inferred['tokenized_prompt_mask'].sum());nt=int(actual['tokenized_prompt_mask'].sum());maximum=max(maximum,n,nt)
            if maximum>=1024:raise ValueError('Actual EEF native prefix/target truncates')
            np.testing.assert_array_equal(inferred['tokenized_prompt'][:n],actual['tokenized_prompt'][:n])
            if actual['state'].shape!=(32,):raise ValueError('Native observation dropped/truncated')
            if model=='pi0_fast' and (actual['image_mask']['base_1_rgb'] or np.any(actual['token_loss_mask'][:n]) or not np.any(actual['token_loss_mask'])):
                raise ValueError('Native FAST absent camera or action loss mask differs')
            reports.append({'index':index,'corrective':index>=data.original_length,'tokens':nt})
    return {'passed':True,'model':model,'action_schema':ACTION_SCHEMA,'actual_native_batches':len(indices)//2,
        'samples':len(reports),'corrective_native_samples':sum(r['corrective'] for r in reports),
        'source_frames':len(data),'original_frames':data.original_length,'corrective_frames':len(data.correction),
        'max_tokens':maximum,'rows':reports,'neural_weights_loaded':False,'long_fit_qualified':False,'model_SR_established':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('audit','train'))
    p.add_argument('--model',choices=('pi0','pi0_fast'),required=True)
    for name in ('projection','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=2);p.add_argument('--qualification',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve every actual EEF adaptation attempt')
    if not 1<=a.updates<=2000:raise ValueError('Bounded real-update budget')
    data=MixedEEF(a.projection,a.admission,a.model);normalization,conditioning=stats(data)
    proof=audit(data,a.model,normalization,a.output)
    proof.update(action_schema=ACTION_SCHEMA,corrective_frames=len(data.correction),rehearsal_mass=.8,corrective_mass=.2,
        projection_audit_sha256=sha256_file(a.projection/'audit.json'),recovery_admission_sha256=sha256_file(a.admission),
        action_transform='shared EEF delta already; NO DeltaActions/AbsoluteActions',conditioning=conditioning)
    if a.mode=='audit':a.output.mkdir(parents=True);atomic_json(a.output/'native_batches.json',proof);return
    if a.updates>2:
        if a.qualification is None:raise ValueError('Own EEF CUDA/save/reload8 proof required')
        q=json.loads(a.qualification.read_text())
        if (not q.get('passed') or q.get('model')!=a.model or q.get('action_schema')!=ACTION_SCHEMA
            or q.get('actual_optimizer_updates')!=2 or q.get('actual_neural_controls')!=8
            or q.get('projection_audit_sha256')!=proof['projection_audit_sha256']
            or q.get('recovery_admission_sha256')!=proof['recovery_admission_sha256']):raise ValueError('Own EEF qualification differs')
        for path,digest in q['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Own EEF qualification changed')
    from VLA.common.source_training_gate import require_training_source
    require_training_source(data.source,model='smolvla',optimizer_updates=1) # common physics evidence ONLY
    cfg=build_config(a.model,normalization,a.output,a.updates)
    contract=cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_input_contract.json')
    atomic_json(contract,{'model':a.model,'state_schema':SCHEMA,'action_schema':ACTION_SCHEMA,'stats':normalization,
        'source':str(data.source),'projection':str(a.projection.resolve()),'requested_updates':a.updates,
        'gradient_accumulation':4,'actual_native_batch_gate':proof,'action_horizon':data.horizon,
        'bound_files':{str(p.resolve()):sha256_file(p) for p in (Path(__file__),a.admission,
            Path(__file__).with_name('fast_decoder_integrity.py'),Path(__file__).resolve().parents[1]/'privileged_native_server.py')}})
    from openpi.training import data_loader
    data_loader.create_torch_dataset=lambda dc,horizon,model:data
    stock_loader=data_loader.TorchDataLoader
    def mixed_loader(dataset,*args,**kwargs):
        import torch
        if kwargs.get('sampler') is not None:raise ValueError('Single-device explicit mixed source only')
        kwargs['shuffle']=False;kwargs['sampler']=torch.utils.data.WeightedRandomSampler(data.weights(),len(data),
            replacement=True,generator=torch.Generator().manual_seed(20261015))
        return stock_loader(dataset,*args,**kwargs)
    data_loader.TorchDataLoader=mixed_loader
    from VLA.pi0 import training
    from VLA.pi0.augmentation import install_colour_safe_augmentation
    augmentation=install_colour_safe_augmentation();assertion=training.assert_fp32_training_state
    def masters(state,train_config):
        from flax.traverse_util import flatten_dict
        result=assertion(state,train_config);rows=[]
        for key,array in flatten_dict(state.params.filter(train_config.trainable_filter).to_pure_dict()).items():
            value=np.asarray(array);rows.append({'path':list(key),'shape':list(value.shape),'dtype':str(value.dtype),
                'initial_sha256':hashlib.sha256(value.tobytes()).hexdigest()})
        atomic_json(cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_master_init.json'),
            {**result,'trainable_tensors':rows,'augmentation':augmentation,'actual_master_initialization':True})
        return result
    training.assert_fp32_training_state=masters;training.run_train(cfg)


if __name__=='__main__':main()
