"""Native OpenPI LoRA recipe with complete causal64 language and two cameras.

No new action head, planner, privileged waypoint or algorithm substitution.
The old18 runtime/trainer stays untouched. Long fits require THIS recipe's
actual CUDA/saved-precision/exact-reload/8-control qualification, not another
family's certificate. Pilot inspection is never training authorization.
"""
import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import PROMPT_SCHEMA
from VLA.scene64_dataset import Scene64NativeDataset


def native_statistics(data):
    """TRAIN-only exact moments/quantiles after the actual DeltaActions mask."""
    states=[];actions=[];offset=0
    for row in data.rows:
        raw=data.states[offset:offset+row['steps']];offset+=row['steps']
        current=np.c_[((raw[:,:6]+np.pi)%(2*np.pi)-np.pi),raw[:,6]].astype(np.float32)
        with np.load(data.source/row['file'],allow_pickle=False) as z:
            from VLA.common.native_action_fields import native_gripper_closure
            requested=z['next_joint_targets'];closed=native_gripper_closure(z)
        ids=np.minimum(np.arange(row['steps'])[:,None]+np.arange(data.horizon),row['steps']-1)
        delta=(requested[ids]-raw[:,None,:6]+np.pi)%(2*np.pi)-np.pi
        actions.append(np.concatenate([delta,closed[ids,None]],axis=-1).reshape(-1,7))
        states.append(current)
    def exact(values):
        v=np.concatenate(values).astype(np.float64)
        return {k:np.asarray(x,np.float32).tolist() for k,x in
            [('mean',v.mean(0)),('std',v.std(0)),('q01',np.quantile(v,.01,axis=0)),('q99',np.quantile(v,.99,axis=0))]}
    return {'state':exact(states),'actions':exact(actions)}


def build_config(model_name, stats, output, updates, *, accumulation=4, horizon=5):
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.shared import normalize
    from openpi.training import config, optimizer
    from VLA.pi0.training import make_config
    from VLA.privileged_native_server import strict_fast_tokenizer_class
    if model_name not in ('pi0','pi0_fast') or not 1<=updates<=1000 or horizon not in (5,10):
        raise ValueError('Explicit bounded native model/update/horizon required')
    args=SimpleNamespace(model=model_name,steps=updates,gradient_accumulation=accumulation,action_horizon=horizon,
        repo_id='local/scene64_native_'+model_name,exp_name=Path(output).name,batch_size=1,
        overwrite=False,resume=False,save_interval=min(updates,250),asset_id='scene64_ur5e',
        fast_codec_scale=10.,fast_tokenizer_path='physical-intelligence/fast')
    cfg=make_config(args)
    model=replace(cfg.model,max_token_len=1024)
    if model_name=='pi0_fast':model=replace(model,fast_model_tokenizer=strict_fast_tokenizer_class())
    norm={key:normalize.NormStats(**{k:np.asarray(v,np.float32) for k,v in item.items()}) for key,item in stats.items()}
    @dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self, raw):
            state=np.r_[raw['joints'],raw['gripper']].astype(np.float32)
            if state.shape!=(7,):raise ValueError('Actual native measured joints/closure required')
            if model.model_type==model_api.ModelType.PI0_FAST:
                images={'base_0_rgb':raw['image'],'base_1_rgb':np.zeros_like(raw['image']),'wrist_0_rgb':raw['wrist_image']}
                masks={'base_0_rgb':np.True_,'base_1_rgb':np.False_,'wrist_0_rgb':np.True_}
                state=np.pad(state,(0,model.action_dim-7))
            else:
                images={'base_0_rgb':raw['image'],'left_wrist_0_rgb':raw['wrist_image']};masks={k:np.True_ for k in images}
            result={'state':state,'image':images,'image_mask':masks,'prompt':raw['prompt']}
            if 'actions' in raw:result['actions']=raw['actions'].copy()
            return result
    @dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self, raw):return {'actions':np.asarray(raw['actions'])[...,:7]}
    @dataclass(frozen=True)
    class Factory(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            group=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]).push(
                inputs=[transforms.DeltaActions(transforms.make_bool_mask(6,-1))],
                outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(6,-1))])
            return config.DataConfig(repo_id=self.repo_id,asset_id='scene64_ur5e',norm_stats=norm,
                use_quantile_norm=model_name=='pi0_fast',data_transforms=group,
                model_transforms=config.ModelTransformFactory()(model_config),prompt_from_task=False)
    return replace(cfg,name=model_name+'_scene64_native_language',model=model,data=Factory(repo_id=args.repo_id),
        checkpoint_base_dir=str(Path(output).resolve().parent),exp_name=Path(output).name,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=0 if updates<=2 else min(100,updates//10),
            decay_steps=max(updates,3),peak_lr=2.5e-5,decay_lr=2.5e-6),
        seed=20261015,policy_metadata={**cfg.policy_metadata,'privileged_schema':PROMPT_SCHEMA,
            'native_token_budget':1024,'synthetic_camera_valid':False,'action_producer':'native_neural_VLA'})


def audit(data, model_name, stats, output):
    from openpi import transforms
    from openpi.training import data_loader
    cfg=build_config(model_name,stats,output,2);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    prepared=data_loader.transform_dataset(data,dc)
    ids=np.linspace(0,len(data)-1,min(len(data),64),dtype=int).tolist();rows=[]
    # Actual official loader/collate, not only a standalone tokenization call.
    loader=data_loader.TorchDataLoader(prepared,local_batch_size=2,
        sampler=ids,num_workers=0,num_batches=len(ids)//2,framework='pytorch')
    prefixmax=totalmax=0
    for number,batch in enumerate(loader):
        picked=ids[number*2:number*2+2]
        for bi,index in enumerate(picked):
            raw=data[index];actual=prepared[index]
            for key in ('state','actions','tokenized_prompt','tokenized_prompt_mask'):
                np.testing.assert_array_equal(np.asarray(batch[key])[bi],actual[key])
            if model_name=='pi0_fast' and bool(actual['image_mask']['base_1_rgb']):raise ValueError('Unavailable camera treated as real')
            no_action={k:v for k,v in raw.items() if k!='actions'}
            before=transforms.compose([*dc.data_transforms.inputs,transforms.Normalize(dc.norm_stats,use_quantiles=dc.use_quantile_norm)])(no_action)
            infer=transforms.compose(dc.model_transforms.inputs)(before)
            n=int(infer['tokenized_prompt_mask'].sum());nt=int(actual['tokenized_prompt_mask'].sum())
            if n>=1024 or nt>=1024:raise ValueError('Actual normalized language/action context truncated')
            np.testing.assert_array_equal(infer['tokenized_prompt'][:n],actual['tokenized_prompt'][:n])
            if model_name=='pi0_fast':
                if np.any(actual['token_loss_mask'][:n]) or not np.any(actual['token_loss_mask']):raise ValueError('FAST action loss/prefix mask differs')
            prefixmax=max(prefixmax,n);totalmax=max(totalmax,nt)
            rows.append({'index':index,'actual_prefix_tokens':n,'actual_train_tokens':nt})
    return {'passed':True,'model':model_name,'scope':'actual_native_CPU_transform_loader_not_CUDA_qualification',
        'pilot_only':data.pilot_only,'samples':len(rows),'actual_loader_batches':len(rows)//2,
        'max_prefix_tokens':prefixmax,'max_total_tokens':totalmax,'native_budget':1024,
        'raw_image_action_text_state_preserved':True,'unavailable_camera_masked':model_name=='pi0_fast',
        'source_frames':len(data),'source_episodes':len(data.rows),'long_fit_qualified':False,'new_model_SR':None,
        'implementation_sha256':sha256_file(Path(__file__)),'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('audit','train'))
    p.add_argument('--model',choices=('pi0','pi0_fast'),required=True);p.add_argument('--projection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true')
    p.add_argument('--updates',type=int,default=2);p.add_argument('--qualification',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep every experiment and failed audit')
    data=Scene64NativeDataset(a.projection,5,pilot=a.pilot);stats=native_statistics(data)
    if a.mode=='audit':
        a.output.mkdir(parents=True);proof=audit(data,a.model,stats,a.output)
        proof['projection_audit_sha256']=sha256_file(a.projection/'audit.json')
        atomic_json(a.output/'norm_stats_values.json',stats);atomic_json(a.output/'native_batches.json',proof);return
    if a.pilot:raise ValueError('Pilot inspection never authorizes training')
    if a.updates>2:
        if a.qualification is None:raise ValueError('Own saved-FP32/exact-reload/8-control qualification required')
        proof=json.loads(a.qualification.read_text())
        if (not proof.get('passed') or proof.get('model')!=a.model or proof.get('actual_optimizer_updates')!=2
            or proof.get('actual_neural_controls')!=8 or proof.get('exact_saved_values_reload') is not True
            or proof.get('implementation_sha256')!=sha256_file(Path(__file__))):raise ValueError('Own native qualification missing')
        for name,digest in proof['dependencies'].items():
            if sha256_file(name)!=digest:raise ValueError('Own native qualification changed')
        if proof.get('projection_audit_sha256')!=sha256_file(a.projection/'audit.json'):raise ValueError('Qualified data changed')
    # Common physics/source evidence is inherited, NOT SmolVLA's fit budget.
    from VLA.common.source_training_gate import require_training_source
    require_training_source(data.source,model='smolvla',optimizer_updates=1)
    cfg=build_config(a.model,stats,a.output,a.updates)
    # Run the same real CPU loader gate before any GPU model allocation.
    before=audit(data,a.model,stats,a.output)
    before['projection_audit_sha256']=sha256_file(a.projection/'audit.json')
    sidecar=cfg.checkpoint_dir.parent/(cfg.exp_name+'.scene64_input_contract.json')
    if sidecar.exists():raise FileExistsError('Preserve native input contract')
    atomic_json(sidecar,{'model':a.model,'privileged_schema':PROMPT_SCHEMA,'actual_native_batch_gate':before,
        'stats':stats,'requested_updates':a.updates,'gradient_accumulation':4,'action_horizon':5,
        'source':str(data.source),'projection':str(a.projection.resolve()),'no_model_SR_certification':True})
    from openpi.training import data_loader
    def source(dc,horizon,model):
        if dc.repo_id!=cfg.data.repo_id or horizon!=5:raise ValueError('Unqualified native dataset/horizon')
        return data
    data_loader.create_torch_dataset=source
    from VLA.pi0.training import run_train
    # run_train audits real FP32 trainable masters/optimizer moments. No
    # 1000-update run is submitted by an input-only inspection certificate.
    run_train(cfg)


if __name__=='__main__':main()
