"""Official flow/FAST LoRA objectives with real native32 simulator state.

Native π0 action projection remains32 (only first7 are motor commands).
FAST keeps its7D action codec; its state is encoded by the native tokenizer,
which consumes32 observed values, not fake extra action labels. Full causal
scene64 language is retained. CUDA/save/reload gates are still mandatory.
"""
import argparse
from dataclasses import dataclass,replace
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import Scene64NativeDataset
from VLA.pi0.scene32_inputs import SCHEMA,FEATURES,state32,statistics


def build_config(model_name,stats,output,updates):
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config
    from VLA.pi0.scene64_training import build_config as original
    cfg=original(model_name,stats,output,updates)
    norm={key:normalize.NormStats(**{k:np.asarray(v,np.float32) for k,v in item.items()}) for key,item in stats.items()}
    @dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self,raw):
            state=state32(raw['pre_action_scene64'])
            if not np.allclose(state[:7],np.r_[raw['joints'],raw['gripper']],rtol=0,atol=1e-6):
                raise ValueError('Native motor observation and feedback timeline disagree')
            if model_name=='pi0_fast':
                images={'base_0_rgb':raw['image'],'base_1_rgb':np.zeros_like(raw['image']),'wrist_0_rgb':raw['wrist_image']}
                masks={'base_0_rgb':np.True_,'base_1_rgb':np.False_,'wrist_0_rgb':np.True_}
            else:
                images={'base_0_rgb':raw['image'],'left_wrist_0_rgb':raw['wrist_image']};masks={k:np.True_ for k in images}
            result={'state':state,'image':images,'image_mask':masks,'prompt':raw['prompt']}
            if 'actions' in raw:result['actions']=raw['actions'].copy()
            return result
    @dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self,raw):return {'actions':np.asarray(raw['actions'])[...,:7]}
    @dataclass(frozen=True)
    class Factory(config.DataConfigFactory):
        def create(self,assets_dirs,model_config):
            group=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]).push(
                inputs=[transforms.DeltaActions(transforms.make_bool_mask(6,-1))],
                outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(6,-1))])
            return config.DataConfig(repo_id=self.repo_id,asset_id='scene32_ur5e',norm_stats=norm,
                use_quantile_norm=model_name=='pi0_fast',data_transforms=group,
                model_transforms=config.ModelTransformFactory()(model_config))
    return replace(cfg,name=model_name+'_scene32_and_scene64_language',data=Factory(repo_id='local/'+model_name+'_causal_scene32'),
        policy_metadata={**cfg.policy_metadata,'state_schema':SCHEMA,'state_fields':FEATURES,
            'actual_state_dim':32,'physical_motor_DOFs':7,'native_action_dim':cfg.model.action_dim})


def audit(data,model,stats,output):
    import copy
    from openpi import transforms
    from openpi.models.model import Observation
    from openpi.training import data_loader
    cfg=build_config(model,stats,output,2);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    native=data_loader.transform_dataset(data,dc)
    ids=np.linspace(0,len(data)-1,64,dtype=int).tolist();rows=[];maxprefix=maxtotal=0
    loader=data_loader.TorchDataLoader(native,local_batch_size=2,sampler=ids,num_batches=32,num_workers=0,framework='pytorch')
    for number,batch in enumerate(loader):
        selected=ids[number*2:number*2+2]
        for bi,index in enumerate(selected):
            raw=data[index];actual=native[index]
            for field in ('state','actions','tokenized_prompt','tokenized_prompt_mask'):
                np.testing.assert_array_equal(np.asarray(batch[field])[bi],actual[field])
            if actual['state'].shape!=(32,) or not np.isfinite(actual['state']).all():raise ValueError('Feedback dropped/truncated')
            inp=dc.data_transforms.inputs[0](raw)
            np.testing.assert_array_equal(inp['state'],state32(raw['pre_action_scene64']))
            observation=copy.deepcopy(actual);observation['image_mask']={k:np.asarray(v) for k,v in observation['image_mask'].items()}
            Observation.from_dict(observation)  # actual typed native boundary
            no_action={k:v for k,v in raw.items() if k!='actions'}
            infer=transforms.compose([*dc.data_transforms.inputs,
                transforms.Normalize(dc.norm_stats,use_quantiles=dc.use_quantile_norm),*dc.model_transforms.inputs])(no_action)
            n=int(infer['tokenized_prompt_mask'].sum());nt=int(actual['tokenized_prompt_mask'].sum())
            if max(n,nt)>=1024:raise ValueError('Actual32 feedback/native action context truncated')
            np.testing.assert_array_equal(infer['tokenized_prompt'][:n],actual['tokenized_prompt'][:n])
            if model=='pi0_fast':
                if actual['image_mask']['base_1_rgb'] or np.any(actual['token_loss_mask'][:n]) or not np.any(actual['token_loss_mask']):
                    raise ValueError('Native FAST camera/loss contract differs')
            maxprefix=max(maxprefix,n);maxtotal=max(maxtotal,nt);rows.append({'index':index,'prefix_tokens':n,'total_tokens':nt})
    return {'passed':True,'model':model,'state_schema':SCHEMA,'actual_state_shape':[32],'native_action_dim':cfg.model.action_dim,
        'physical_motor_DOFs':7,'actual_native_batches':32,'samples':len(rows),'pilot_only':data.pilot_only,
        'max_prefix_tokens':maxprefix,'max_total_tokens':maxtotal,'raw_state_fields':FEATURES,
        'source_frames':len(data),'neural_model_loaded':False,'long_fit_qualified':False,'SR_established':False,
        'implementation_sha256':sha256_file(Path(__file__)),'state_implementation_sha256':sha256_file(Path(__file__).with_name('scene32_inputs.py')),'rows':rows}


def main():
    from VLA.pi0.scene64_training import native_statistics
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('audit','train'))
    p.add_argument('--model',choices=('pi0','pi0_fast'),required=True)
    for name in ('projection','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--updates',type=int,default=2);p.add_argument('--qualification',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve every attempted native32 audit/fit')
    data=Scene64NativeDataset(a.projection,5,pilot=a.pilot);stats=native_statistics(data)
    stats['state'],conditioning=statistics(data.states)
    proof=audit(data,a.model,stats,a.output);proof['projection_audit_sha256']=sha256_file(a.projection/'audit.json')
    proof['normalization']=conditioning
    if a.mode=='audit':
        a.output.mkdir(parents=True);atomic_json(a.output/'native_batches.json',proof);atomic_json(a.output/'norm_stats_values.json',stats);return
    if a.pilot:raise ValueError('Pilot data cannot be trained')
    if a.updates>2:
        if a.qualification is None:raise ValueError('Own actual32 CUDA/FP32/exact-reload/eight-control qualification required')
        q=json.loads(a.qualification.read_text())
        if (not q.get('passed') or q.get('model')!=a.model or q.get('state_schema')!=SCHEMA
            or q.get('actual_optimizer_updates')!=2 or q.get('actual_neural_controls')!=8
            or q.get('exact_saved_values_reload') is not True or q.get('saved_masters_and_moments_FP32') is not True):
            raise ValueError('Own actual32 training qualification missing')
        for file,digest in q['dependencies'].items():
            if sha256_file(file)!=digest:raise ValueError('Actual32 qualification stale')
        if q.get('projection_audit_sha256')!=proof['projection_audit_sha256']:raise ValueError('Actual32 training data changed')
    from VLA.common.source_training_gate import require_training_source
    require_training_source(data.source,model='smolvla',optimizer_updates=1) # common physical/split evidence ONLY
    cfg=build_config(a.model,stats,a.output,a.updates)
    contract=cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_input_contract.json')
    if contract.exists():raise FileExistsError('Preserve native32 run contract')
    atomic_json(contract,{'actual_checkpoint_dir':str(cfg.checkpoint_dir),'model':a.model,'state_schema':SCHEMA,
        'actual_native_batch_gate':proof,'stats':stats,'requested_updates':a.updates,'gradient_accumulation':4,
        'action_horizon':5,'source':str(data.source),'projection':str(a.projection.resolve()),
        'bound_files':{str(p.resolve()):sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('scene32_inputs.py'),
            Path(__file__).with_name('scene64_training.py'))}})
    from openpi.training import data_loader
    def source(dc,horizon,model):
        if dc.repo_id!=cfg.data.repo_id or horizon!=5:raise ValueError('Unqualified dataset/horizon')
        return data
    data_loader.create_torch_dataset=source
    import VLA.pi0.training as training
    from VLA.pi0.augmentation import install_colour_safe_augmentation
    augmentation=install_colour_safe_augmentation()
    stock_assert=training.assert_fp32_training_state
    def init_proof(state,train_config):
        from flax.traverse_util import flatten_dict
        import hashlib
        checked=stock_assert(state,train_config);rows=[]
        for key,array in flatten_dict(state.params.filter(train_config.trainable_filter).to_pure_dict()).items():
            value=np.asarray(array)
            rows.append({'path':list(key),'shape':list(value.shape),'dtype':str(value.dtype),
                'initial_sha256':hashlib.sha256(value.tobytes()).hexdigest()})
        atomic_json(cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene32_master_init.json'),
            {**checked,'trainable_tensors':rows,'augmentation':augmentation,'actual_master_initialization':True})
        return checked
    training.assert_fp32_training_state=init_proof
    training.run_train(cfg)


if __name__=='__main__':main()
