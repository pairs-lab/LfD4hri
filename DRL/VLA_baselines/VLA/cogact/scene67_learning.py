"""Official CogACT cognition/DiT with a learned privileged-state adapter.

Frozen official image/language features remain a required input. The native
DiT diffusion loss learns all7 actions; a zero-initialized state MLP adds
observation conditioning, never a trajectory/action target or IK correction.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import Scene64NativeDataset
from VLA.scene64_conditioning import encode_numeric,normalization
from VLA.native_corrective_dataset import NativeCorrectiveDataset

ROOT=Path(__file__).resolve().parents[2]
OFFICIAL=ROOT/'VLA/checkpoints/cogact-small/checkpoints/CogACT-Small.pt'
SCHEMA='CogACT_native_DiT_plus_learned_simscene67_conditioning_v1'


class StateAdapter(torch.nn.Module):
    def __init__(self,stats):
        super().__init__();self.register_buffer('mean',torch.tensor(stats['mean'],dtype=torch.float32))
        self.register_buffer('std',torch.tensor(stats['std'],dtype=torch.float32).clamp_min(.001))
        self.net=torch.nn.Sequential(torch.nn.Linear(67,256),torch.nn.GELU(),torch.nn.Linear(256,4096))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,observed):return self.net((observed-self.mean)/(self.std+1e-8)).unsqueeze(1)


class Captured(Exception):pass


def cognition(model,image,prompt):
    """Run the EXACT official image/EOS-language pathway, stop before DiT."""
    result={}
    builder=model.vlm.get_prompt_builder();builder.add_turn(role='human',message=f'What action should the robot take to {prompt.lower()}?')
    ids=model.vlm.llm_backbone.tokenizer(builder.get_prompt(),truncation=False).input_ids
    if len(ids)+258>=2048:raise ValueError('Actual native CogACT image/text/EOS context would truncate')
    def stop(*args,**kwargs):
        result['feature']=kwargs['z'].detach().float().cpu().numpy()[0]
        raise Captured()
    with patch.object(model.action_model.net,'forward',stop):
        try:model.predict_action(image=Image.fromarray(image).resize((224,224),Image.Resampling.LANCZOS),
            instruction=prompt,unnorm_key='berkeley_autolab_ur5',cfg_scale=1.,use_ddim=True,num_ddim_steps=10,do_sample=False)
        except Captured:pass
    if 'feature' not in result or result['feature'].shape!=(1,4096) or not np.isfinite(result['feature']).all():
        raise ValueError('Actual native image/language cognition was not captured')
    return result['feature']


class Corpus:
    def __init__(self,projection,admission,pilot=False):
        self.original=Scene64NativeDataset(projection,16);self.correction=NativeCorrectiveDataset(admission,16)
        self.states=np.concatenate((self.original.states,self.correction.states))
        self.stats,_=normalization(self.states);self.items=[];self.pilot=pilot
        counts={}
        for ei,row in enumerate(self.original.rows):
            task=row['task']['name'];counts[task]=counts.get(task,0)+1
            if pilot and counts[task]>2:continue
            start=self.original.ends[ei-1] if ei else 0;n=row['steps']
            # Cover every original scene and transitions, not just t=0 or
            # successful endpoints. Selection uses TRAIN labels/state only.
            ids=set(np.linspace(0,n-1,min(6,n),dtype=int).tolist())
            held=self.original.states[start:start+n,17]
            for transition in np.flatnonzero(np.diff(held)!=0)+1:
                ids.update(int(np.clip(transition+offset,0,n-1)) for offset in (-1,0,1))
            for fi in sorted(ids):self.items.append({'kind':'original','index':start+fi,'file':row['file'],'frame':fi,'task':task,'target':row['task']['pick_name']})
        for ei,row in enumerate(self.correction.rows):
            if pilot and any(i['kind']=='correction' and i['task']==row['task']['name'] for i in self.items):continue
            start=self.correction.ends[ei-1] if ei else 0
            for fi in range(row['steps']):self.items.append({'kind':'correction','index':start+fi,'file':row['file'],
                'frame':fi,'task':row['task']['name'],'target':row['task']['pick_name']})
        source=ROOT/'VLA/audits/scene67_deadline_r31_20261015/other_native_full_input_gates/openvla_scene64_source/dataset_statistics.json'
        self.action_stats=next(iter(json.loads(source.read_text()).values()))['action']
    def get(self,item):return (self.original if item['kind']=='original' else self.correction)[item['index']]


def cache(a):
    from VLA.cross_vla120_server import load_cogact
    a.output.mkdir(parents=True,exist_ok=False);data=Corpus(a.projection,a.admission,a.pilot)
    # Content provenance binds the actual features, source proofs and both
    # feature/runtime implementations. No observation or action is guessed.
    bound={str(p.resolve()):sha256_file(p) for p in (Path(__file__),a.projection/'audit.json',a.admission)}
    plan={'schema':SCHEMA,'pilot_only':a.pilot,'items':data.items,'stats':data.stats,'action_stats':data.action_stats,
        'bound_files':bound,'original_source':str(data.original.source),'original_source_frames':len(data.original),
        'original_source_episodes':len(data.original.rows),'correction_source_frames':len(data.correction),
        'original_projection':str(a.projection.resolve()),'native_correction_admission':str(a.admission.resolve()),
        'VLM_frozen':True,'native_image_and_EOS_prompt_path':True,'features_are_actual_neural_outputs':True,
        'official_checkpoint':str(OFFICIAL),'model_SR_established':False}
    atomic_json(a.output/'plan.json',plan)
    model,_=load_cogact(OFFICIAL,a.output/'official_load');model.requires_grad_(False)
    count=len(data.items)
    features=np.lib.format.open_memmap(a.output/'cognition.npy',mode='w+',dtype=np.float32,shape=(count,1,4096))
    states=np.lib.format.open_memmap(a.output/'state67.npy',mode='w+',dtype=np.float32,shape=(count,67))
    actions=np.lib.format.open_memmap(a.output/'actions.npy',mode='w+',dtype=np.float32,shape=(count,16,7))
    low,high,mask=[np.asarray(data.action_stats[k],bool if k=='mask' else np.float32) for k in ('q01','q99','mask')]
    for i,item in enumerate(data.items):
        raw=data.get(item);features[i]=cognition(model,raw['image'],raw['prompt']);states[i]=raw['numeric_scene67']
        actions[i]=np.where(mask,2*(raw['language_actions']-low)/(high-low+1e-8)-1,raw['language_actions']).clip(-1,1)
        # The reference constant orientation dimensions have low=high=0.
        # Native dataset normalization maps those labels to0, not -1.
        actions[i,:,mask&(np.abs(high-low)<1e-8)]=0
        if i%25==0 or i==count-1:
            features.flush();states.flush();actions.flush()
            atomic_json(a.output/'progress.json',{'completed_actual_cognition_frames':i+1,'planned_frames':count,
                'pilot_only':a.pilot,'model_SR_established':False,'long_fit_qualified':False})
    proof={**plan,'passed':True,'frames':count,'original_cached_frames':sum(i['kind']=='original' for i in data.items),
        'corrective_cached_frames':sum(i['kind']=='correction' for i in data.items),'cache_files':{
            n:sha256_file(a.output/n) for n in ('cognition.npy','state67.npy','actions.npy')},
        'conditioning_stats_sha256':hashlib.sha256(json.dumps(data.stats,sort_keys=True).encode()).hexdigest(),
        'plan_sha256':sha256_file(a.output/'plan.json'),'long_fit_qualified':False,'model_SR_established':False}
    atomic_json(a.output/'audit.json',proof)


def train(a):
    import sys
    sys.path.insert(0,str(ROOT/'VLA/third_party/cogact'))
    from action_model.action_model import ActionModel
    proof=json.loads((a.cache/'audit.json').read_text())
    if not proof.get('passed') or proof['schema']!=SCHEMA:raise ValueError('Actual native cognition cache proof required')
    for path,digest in proof['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Cognition/source implementation changed')
    for name,digest in proof['cache_files'].items():
        if sha256_file(a.cache/name)!=digest:raise ValueError('Actual neural feature cache changed')
    if a.updates>2:
        if proof['pilot_only'] or a.qualification is None:raise ValueError('Complete cache and own CUDA/reload8 proof required')
        q=json.loads(a.qualification.read_text())
        if (not q.get('passed') or q.get('schema')!=SCHEMA or q.get('actual_neural_controls')!=8
            or q.get('actual_optimizer_updates')!=2 or q.get('conditioning_stats_sha256')!=proof['conditioning_stats_sha256']):
            raise ValueError('Own CogACT numeric/control gate differs')
        for path,digest in q['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Own CogACT qualification stale')
    a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015)
    torch.set_num_threads(2)
    head=ActionModel(token_size=4096,model_type='DiT-S',in_channels=7,future_action_window_size=15,past_action_window_size=0)
    official=torch.load(OFFICIAL,map_location='cpu',mmap=True,weights_only=True)['model']['action_model']
    head.load_state_dict(official,strict=True);del official
    adapter=StateAdapter(proof['stats']);head=head.cuda().train();adapter=adapter.cuda().train()
    optimizer=torch.optim.AdamW(list(head.parameters())+list(adapter.parameters()),lr=1e-4)
    features=np.load(a.cache/'cognition.npy',mmap_mode='r');states=np.load(a.cache/'state67.npy',mmap_mode='r');actions=np.load(a.cache/'actions.npy',mmap_mode='r')
    groups={}
    for i,item in enumerate(proof['items']):groups.setdefault((item['kind'],item['task'],item['target']),[]).append(i)
    tasks=('touch','pick','move','pick_place');rng=np.random.default_rng(20261015)
    logs=[]
    for step in range(a.updates):
        indices=[]
        for _ in range(32):
            kind='correction' if rng.random()<.2 else 'original';task=tasks[int(rng.integers(4))]
            options=[g for g in groups if g[:2]==(kind,task)];key=options[int(rng.integers(len(options)))];group=groups[key]
            indices.append(group[int(rng.integers(len(group)))])
        z=torch.from_numpy(np.asarray(features[indices]).copy()).cuda();state=torch.from_numpy(np.asarray(states[indices]).copy()).cuda()
        target=torch.from_numpy(np.asarray(actions[indices]).copy()).cuda();optimizer.zero_grad(set_to_none=True)
        loss=head.loss(target,z+adapter(state))
        if not torch.isfinite(loss):raise ValueError('Actual CogACT diffusion loss is nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(list(head.parameters())+list(adapter.parameters()),1.0);optimizer.step()
        if step%25==0 or step==a.updates-1:
            logs.append({'optimizer_step':step+1,'native_diffusion_loss':float(loss.detach().cpu())})
            atomic_json(a.output/'training_progress.json',{'actual_optimizer_updates':step+1,'records':logs,'model_SR_established':False})
    payload={'action_model':head.cpu().state_dict(),'state_adapter':adapter.cpu().state_dict(),'optimizer':optimizer.state_dict(),'actual_updates':a.updates}
    torch.save(payload,a.output/'learned.pt')
    for state in payload['optimizer']['state'].values():
        if float(state['step'])!=a.updates or state['exp_avg'].dtype!=torch.float32 or state['exp_avg_sq'].dtype!=torch.float32:
            raise ValueError('Actual CogACT FP32 Adam/update evidence failed')
    if any(t.dtype!=torch.float32 for part in ('action_model','state_adapter') for t in payload[part].values() if t.is_floating_point()):
        raise ValueError('CogACT learned master was downcast')
    original=Path(proof['original_source'])
    atomic_json(a.output/'vla_checkpoint_manifest.json',{'source':str(original),'source_manifest_sha256':sha256_file(original/'dataset_manifest.json'),
        'schema':SCHEMA,'actual_optimizer_updates':a.updates,'VLM_frozen':True,'privileged_inputs':True})
    atomic_json(a.output/'scene67_contract.json',{'schema':SCHEMA,'stats':proof['stats'],'action_stats':proof['action_stats'],
        'actual_optimizer_updates':a.updates,'cache':str(a.cache.resolve()),'cache_audit_sha256':sha256_file(a.cache/'audit.json'),
        'conditioning_stats_sha256':proof['conditioning_stats_sha256'],'learned_sha256':sha256_file(a.output/'learned.pt'),
        'native_frozen_cognition':True,'learned_actions_native_DiT':True,'oracle_at_deployment':False,
        'bound_files':{str(Path(__file__).resolve()):sha256_file(Path(__file__))}})
    atomic_json(a.output/'saved_precision.json',{'passed':True,'actual_optimizer_updates':a.updates,
        'FP32_masters_and_Adam':True,'model_SR_established':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('cache','train'))
    for name in ('projection','admission','cache','qualification'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--updates',type=int,default=2)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve each actual native CogACT attempt')
    if a.mode=='cache':cache(a)
    else:train(a)


if __name__=='__main__':main()
