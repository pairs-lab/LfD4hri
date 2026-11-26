"""Genuine Seer+CLIP+MAE, native losses, explicit learned privileged branch."""
import importlib
from pathlib import Path
import sys
import types
import numpy as np
from .common import ROOT,OUT,ASSETS,read

REPO=ROOT/'VLA/third_party/seer'


def native_class():
    # Seer's directory has no models/__init__.py. The workspace's regular
    # `models` package otherwise wins over this namespace even with sys.path.
    package=types.ModuleType('models');package.__path__=[str(REPO/'models')]
    sys.modules['models']=package
    return importlib.import_module('models.seer_model').SeerAgent


class Runtime:
    horizon=3
    history=7

    def __init__(self,stats,*,device='cuda',checkpoint=None,foundation=None):
        import torch
        import clip
        cls=native_class();self.stats=stats;self.device=device
        foundation_spec=read(Path(__file__).resolve().parents[2]/'protocol21/foundations.json')['seer']
        self.topology=foundation_spec['native_topology']
        config={k:v for k,v in self.topology.items() if k not in ('transformer_heads_source',)}
        config.update(sequence_length=7,finetune_type='real',clip_device='cpu',phase='finetune',
            vit_checkpoint_path=str(ASSETS/'seer/mae_pretrain_vit_base.pth'))
        load_clip=clip.load
        def local_clip(name,*args,**kwargs):
            if name not in ('ViT-B/32','checkpoints/clip/ViT-B-32.pt'):raise ValueError('Unexpected CLIP foundation')
            return load_clip(str(ASSETS/'seer/ViT-B-32.pt'),*args,**kwargs)
        clip.load=local_clip
        try:self.model=cls(**config)
        finally:clip.load=load_clip
        from VLA.common.provenance import sha256_file
        base_path=Path(foundation) if foundation is not None else ASSETS/'seer/seer_droid.pth'
        if sha256_file(base_path)!=foundation_spec['sha256']:
            raise ValueError('Seer foundation bytes changed')
        value=torch.load(base_path,map_location='cpu',mmap=True,weights_only=True)
        state={k.removeprefix('module.'):v for k,v in value['model_state_dict'].items()}
        state['transformer_backbone_position_embedding']=state['transformer_backbone_position_embedding'][:,:7].clone()
        expected=self.model.state_dict()
        unexpected=set(state)-set(expected)
        missing=set(expected)-set(state)
        fixed={k for k,p in self.model.named_parameters() if not p.requires_grad}
        fixed.update(k for k,_ in self.model.named_buffers())
        if unexpected or any(not k.startswith(('vision_encoder.','clip_model.')) and k not in fixed for k in missing):
            raise ValueError('Unexpected Seer native missing/unexpected keys: '+str((missing,unexpected)))
        for k,v in state.items():
            if expected[k].shape!=v.shape:raise ValueError('Native checkpoint shape mismatch: '+k)
        loaded=self.model.load_state_dict(state,strict=False)
        if set(loaded.missing_keys)!=missing or loaded.unexpected_keys:raise ValueError('Strict documented key qualification differs')
        self.load_proof={'authentic_learned_native_keys':len(state),'only_missing_keys':sorted(missing),
            'frozen_encoders_loaded_from_authentic_independent_assets':True,
            'native_official_position_prefix_adaptation':[11,7]}
        del value,state,expected
        self.model.clip_model.requires_grad_(False);self.model.vision_encoder.requires_grad_(False)
        self.model.robot21=torch.nn.Sequential(torch.nn.Linear(21,384),torch.nn.SiLU(),torch.nn.Linear(384,384))
        torch.nn.init.zeros_(self.model.robot21[-1].weight);torch.nn.init.zeros_(self.model.robot21[-1].bias)
        self._privileged=None
        def add_measured_state(module,args,output):
            if self._privileged is None:raise ValueError('Current causal privileged state missing')
            if output.shape!=self._privileged.shape:raise ValueError('Native state/history privileged shape mismatch')
            return output+self._privileged
        self.model.state_projector.register_forward_hook(add_measured_state)
        if checkpoint:
            from .checkpoint_io import load_adaptation
            load_adaptation(self.model,checkpoint)
        self.model.to(device)
        # Same disclosed REAL precision recipe: trainable FP32, frozen MAE BF16.
        if device=='cuda':self.model.vision_encoder.bfloat16()
        self.model._init_model_type()
        self.trainables={k:p for k,p in self.model.named_parameters() if p.requires_grad}
        self.last_input={}

    def images(self,front,wrist,*,augment=False):
        import torch
        from PIL import Image
        streams=[]
        for values,pad in ((front,10),(wrist,4)):
            x=torch.stack([self.model.image_processor(Image.fromarray(im)) for im in values])
            if augment:
                offset=torch.randint(0,2*pad+1,(2,)).tolist()
                x=torch.nn.functional.pad(x,(pad,pad,pad,pad),mode='replicate')
                x=x[:,:,offset[0]:offset[0]+224,offset[1]:offset[1]+224]
            streams.append(x)
        return streams

    def batch(self,s,*,training=False):
        import torch
        import clip
        text=clip.tokenize([s['instruction']],truncate=False)
        if training:
            fronts=np.concatenate((s['front_history'],s['future_front_labels']))
            wrists=np.concatenate((s['wrist_history'],s['future_wrist_labels']))
        else:fronts,wrists=s['front_history'],s['wrist_history']
        front,wrist=self.images(fronts,wrists,augment=training)
        measured=np.asarray(s['robot21_history'],np.float32)
        if measured.shape!=(7,21):raise ValueError('Expected seven normalized causal robot21 observations')
        self._privileged=self.model.robot21(torch.as_tensor(measured,device=self.device,dtype=torch.float32))
        if self._privileged.shape!=(7,384):raise ValueError('Exactly seven causal state tokens required')
        result={'image_primary':front[:7][None].to(self.device), 'image_wrist':wrist[:7][None].to(self.device),
            'state':torch.as_tensor(s['state_history'][None],device=self.device,dtype=torch.float32),
            'text_token':text[:,None].repeat(1,7,1).to(self.device)}
        self.last_input={'views':2,'history':7,'native_state_dim':7,'robot_state_dim':21,
            'text_tokens':int((text!=0).sum()),'future_images_in_model_input':False,
            'native_horizon':3,'history_reset_is_repeat_initial_observation':True}
        return result,(front[7:],wrist[7:]) if training else None

    def loss(self,s):
        import torch
        import torch.nn.functional as F
        self.model.train();self.model.clip_model.eval();self.model.vision_encoder.eval()
        batch,future=self.batch(s,training=True)
        arm,grip,image,_,_,_=self.model(**batch)
        scales=np.asarray(self.stats['local_action_scale6'],np.float32)
        target=s['local_actions'].copy();target[...,:6]/=scales
        if np.max(abs(target[...,:6]))>1.000001:raise ValueError('TRAIN label exceeds calibrated native Tanh range')
        target[...,6]=(target[...,6]+1)/2
        target=torch.as_tensor(target[None],device=self.device,dtype=torch.float32)
        valid=torch.as_tensor(s['valid_actions'][None],device=self.device,dtype=torch.bool)
        arm_loss=F.smooth_l1_loss(arm[valid],target[...,:6][valid])
        grip_loss=F.binary_cross_entropy(grip[valid],target[...,6:][valid])
        # Match upstream patchify + per-patch sample-variance normalization.
        def patchify(x):
            x=x.reshape(7,3,14,16,14,16)
            x=torch.einsum('nchpwq->nhwpqc',x).reshape(7,196,768)
            return (x-x.mean(-1,keepdim=True))/(x.var(-1,keepdim=True)+1e-6).sqrt()
        image=image.reshape(7,2,196,768)
        valid_image=torch.as_tensor(s['valid_future_images'],device=self.device,dtype=torch.bool)
        if valid_image.any():
            targets=[patchify(x.to(self.device)) for x in future]
            image_loss=.5*(F.mse_loss(image[valid_image,0],targets[0][valid_image])+
                F.mse_loss(image[valid_image,1],targets[1][valid_image]))
        else:image_loss=image.sum()*0.
        loss=arm_loss+.01*grip_loss+.1*image_loss
        return {'loss':loss,'loss_components':{'action':arm_loss,'gripper':grip_loss,'image':image_loss}}

    def predict(self,s,seed=0):
        import torch
        self.model.eval()
        # Native MAE still shuffles every patch at mask_ratio=0 in eval mode.
        # Fork a query seed so restore/reload reproduces that native operation
        # exactly, without deleting its shuffling or changing any weights.
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(int(seed))
            batch,_=self.batch(s,training=False);arm,grip,_,_,_,_=self.model(**batch)
        result=np.c_[arm[0,-1].float().cpu().numpy()*np.asarray(self.stats['local_action_scale6']),
            np.where(grip[0,-1].float().cpu().numpy()>.5,1.,-1.)]
        if result.shape!=(3,7) or not np.isfinite(result).all():raise ValueError('Malformed native Seer local action')
        return result.astype(np.float32)
