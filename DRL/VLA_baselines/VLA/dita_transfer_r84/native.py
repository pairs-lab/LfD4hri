"""DINOv2 -> text-FiLM QFormer -> native continuous-action diffusion Transformer.

Extensions: shared encoder on two views, one native7+robot21 token in context.
No new small predictor replaces the native pretrained denoising Transformer.
"""
from functools import lru_cache
import sys
import numpy as np
from PIL import Image
from .common import BASE,TEXT,REPO,OUT,read

class Runtime:
    horizon=10
    def __init__(self,stats,*,checkpoint=None,device='cuda',load_text=True):
        import torch
        import torch.nn as nn
        sys.path.insert(0,str(REPO/'scripts'))
        from vision_tokenizers import vit_tokenizer
        create=vit_tokenizer.create_model
        # Native source requests the DINO foundation before loading the entire
        # robot checkpoint. Avoid duplicate downloads; EVERY native key is then
        # strictly loaded from the official robot weights, no random frozen ViT.
        def from_config(name,**kwargs):
            kwargs['pretrained']=False
            return create(name,**kwargs)
        vit_tokenizer.create_model=from_config
        from llama_dp import RobotTransformerNet
        try:
            network=RobotTransformerNet(output_tensor_spec={},trajectory_dim=7,
                time_sequence_length=11,num_layers=12,dropout_rate=0.,
                freeze_backbone=False,use_qformer=True,use_wrist_img=False,
                use_depth_img=False,dim_align_type=0,prediction_type='epsilon',
                scheduler_type=0,num_inference_steps=10,attn_implementation='eager',
                use_action_head_diff=False,input_size='(224,224)')
        finally:vit_tokenizer.create_model=create
        from pathlib import Path
        artifact=read(Path(__file__).resolve().parents[2]/'protocol21/foundations.json')['dita']
        from VLA.common.provenance import sha256_file
        if sha256_file(BASE)!=artifact['sha256']:
            raise ValueError('Official robot checkpoint bytes changed')
        weights=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True)['parameter']
        missing=set(network.state_dict())-set(weights)
        unexpected=set(weights)-set(network.state_dict())
        if missing or unexpected:
            raise ValueError('Native Dita key mismatch: '+str({'missing':sorted(missing),'unexpected':sorted(unexpected)}))
        network.load_state_dict(weights,strict=True)
        self.model=nn.ModuleDict({'native':network,'sensor28':nn.Linear(28,768)})
        nn.init.zeros_(self.model['sensor28'].weight);nn.init.zeros_(self.model['sensor28'].bias)
        if checkpoint:
            from safetensors.torch import load_model
            load_model(self.model,str(checkpoint),strict=True)
        self.model.to(device)
        self.device=device;self.stats=stats
        self.trainables={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        self.last_input={}
        self.text_encoder=self.tokenizer=None
        if load_text:
            from transformers import CLIPTextModel,AutoTokenizer
            self.text_encoder=CLIPTextModel.from_pretrained(TEXT).eval()
            self.text_encoder.requires_grad_(False)
            self.tokenizer=AutoTokenizer.from_pretrained(TEXT)

    @lru_cache(maxsize=256)
    def language(self,instruction):
        import torch
        if self.text_encoder is None:raise ValueError('Native frozen CLIP text encoder not loaded')
        text=self.tokenizer([instruction],padding='max_length',max_length=77,truncation=False,return_tensors='pt')
        if text['input_ids'].shape!=(1,77):raise ValueError('Native CLIP instruction unexpectedly too long')
        with torch.no_grad():
            return self.text_encoder(**text).last_hidden_state[:,None].detach().clone()

    def context(self,s,*,augment=False):
        import torch
        from torchvision.transforms import functional as F
        from torchvision.transforms import RandomResizedCrop
        language=self.language(s['instruction']).to(self.device)
        tensors=[]
        for name in ('front','wrist'):
            frames=[]
            # Two measured RGB timestamps: previous/current, reset repeats only.
            for rgb in s['history_images'][name]:
                im=Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)
                if augment:
                    # Disclosed target-domain mild photometric augmentation; no
                    # unpaired random geometry between the two physical frames.
                    im=F.adjust_brightness(im,float(torch.empty(()).uniform_(.9,1.1)))
                v=F.pil_to_tensor(im).float()/255
                frames.append(F.normalize(v,[.485,.456,.406],[.229,.224,.225]))
            tensors.append(torch.stack(frames)[None].to(self.device))
        lang=language.expand(1,2,77,768)
        tokens=[self.model['native'].image_tokenizer(v,lang) for v in tensors]
        numeric=torch.as_tensor(s['history28'][-2:][None],device=self.device)
        if numeric.shape != (1,2,28):raise ValueError('Dita requires causal history2 x native7+robot21')
        sensors=self.model['sensor28'](numeric).unsqueeze(2)
        context=torch.cat([*tokens,sensors],dim=2)
        obs={'image':tensors[0],'natural_language_embedding':lang}
        self.last_input={'actual_views':2,'RGB_timestamps':2,'native_QFormer_tokens_per_view':32,
            'context_shape':list(context.shape),'measured_sensor_shape':[1,2,28],
            'unseen_future_or_expert_phase':False}
        return obs,context

    def loss(self,s):
        import torch
        import torch.nn.functional as F
        self.model.train()
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            obs,features=self.context(s,augment=True)
            target=torch.as_tensor(s['normalized_action11'][None],device=self.device)
            noise=torch.randn_like(target)
            steps=torch.randint(0,100,(1,),device=self.device)
            scheduler=self.model['native'].noise_scheduler
            noisy=scheduler.add_noise(target,noise,steps)
            predicted=self.model['native'](obs,act=None,noisy_action_tokens=noisy,timesteps=steps,
                num_pred_action=10,reused_context_image_tokens=features)
            mask=torch.as_tensor(s['action_valid11'][None,:,None],device=self.device)
            value=((predicted.float()-noise).square()*mask).sum()/(mask.sum()*7)
        return {'loss':value,'loss_components':{'native_DDPM_epsilon_MSE':value}}

    def predict(self,s,seed):
        import torch
        from VLA.replacement_r84.data import denormalize_actions
        self.model.eval()
        with torch.random.fork_rng(devices=[0] if self.device=='cuda' else []),torch.no_grad(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            torch.manual_seed(int(seed));obs,features=self.context(s)
            trajectory=torch.randn((1,11,7),device=self.device)
            scheduler=self.model['native'].noise_scheduler_eval
            scheduler.set_timesteps(10,device=self.device)
            for t in scheduler.timesteps:
                predicted=self.model['native'](obs,act=None,noisy_action_tokens=trajectory,timesteps=t[None],
                    num_pred_action=10,reused_context_image_tokens=features)
                trajectory=scheduler.step(predicted,t,trajectory).prev_sample
            result=trajectory[0,-10:].float().cpu().numpy()
        if result.shape!=(10,7) or not np.isfinite(result).all():raise ValueError('Malformed native Dita chunk')
        canonical=denormalize_actions(result,self.stats)
        # Match official absolute OPEN01 binarization, converted to our common
        # actuator signed-close convention; never use object pose to repair it.
        canonical[:,6]=np.where(result[:,6]>.5,-1.,1.)
        return canonical
