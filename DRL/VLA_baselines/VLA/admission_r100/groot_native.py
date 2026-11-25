"""Authentic N1.5 Eagle+flow expert; learned custom-embodiment31 adaptation."""
import importlib
import ast
import copy
import sys
import types
from pathlib import Path
import numpy as np

from .common import ROOT,OUT,ASSETS,read,atomic_json
from VLA.refinement_r99.data import Data as ParentData
from VLA.replacement_r84.data import observation74,normalize_actions,denormalize_actions
from VLA.family_transfer_r54.data import action_stats
from VLA.new_vlas.action_contracts import native8

REPO=ROOT/'VLA/third_party/gr00t_n15'


def imports():
    sys.path.insert(0,str(REPO))
    import gr00t
    # Avoid importing unrelated deployment/video datasets merely to import a
    # model or a transform. Actual native modules remain unmodified on disk.
    for name,path in (('gr00t.model',REPO/'gr00t/model'),('gr00t.data.transform',REPO/'gr00t/data/transform')):
        if name not in sys.modules:
            package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
    module=importlib.import_module('gr00t.model.gr00t_n1')
    return module


def isolate_unused_radio():
    """Load actual RADIOConfig only; selected SigLIP never executes RADIO.

    Upstream eagerly imports FlashAttention for an unselected RADIO branch.
    Do not install/fake a flash kernel or swap the selected SigLIP encoder.
    Any accidental RADIO construction fails closed instead of running a proxy.
    """
    from typing import Optional
    from transformers import PretrainedConfig
    file=REPO/'gr00t/model/backbone/eagle2_hg_model/radio_model.py'
    tree=ast.parse(file.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RADIOConfig']
    if len(nodes)!=1:raise ValueError('Native optional RADIO configuration changed')
    name='transformers_modules.eagle2_hg_model.radio_model'
    module=types.ModuleType(name);module.__file__=str(file)
    namespace={'PretrainedConfig':PretrainedConfig,'Optional':Optional,'copy':copy,'__name__':name}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),namespace)
    module.RADIOConfig=namespace['RADIOConfig']
    class UnselectedRADIO:
        def __init__(self,*args,**kwargs):raise ValueError('RADIO is not the authentic selected SigLIP checkpoint')
    module.RADIOModel=UnselectedRADIO
    sys.modules[name]=module


class Data(ParentData):
    def __init__(self):super().__init__();self.horizon=16

    def at(self,ei,fi):
        s=super().at(ei,fi)
        s['native_prefix6']=self.numeric(ei)['native8'][fi,:6].copy()
        return s


def statistics(data):
    return {'scene67':data.stats['scene67'],'actions':action_stats(data),'native_state_width':80,
        'valid_causal_features':74,'native_horizon':16,'valid_physical_action_channels':7,
        'custom_embodiment_id':31,'action_gripper':'new embodiment signed CLOSE, canonical>0 close'}


def native_batch(s,stats,transform,training=False):
    from PIL import Image
    from VLA.privileged_visual_r53.scene_map import render
    from gr00t.model.transforms import collate
    raw=s['raw64'];prefix=s.get('native_prefix6')
    if prefix is None:prefix=native8(s['ee_position'],s['ee_rotation_6d'],raw[27])[:6]
    state=observation74(prefix,raw,stats)
    images=[s['front'],s['wrist'],s.get('map',render(raw,s['task']))]
    video=np.stack([np.asarray(Image.fromarray(im).resize((224,224),Image.Resampling.BILINEAR)) for im in images])[None]
    transform.training=training
    data={'video':video,'annotation.instruction':[s['instruction']],'state':state[None]}
    if training:data['action']=normalize_actions(s['action'],stats)
    result=transform(data)
    if result['state'].shape!=(1,80) or not np.array_equal(result['state'][0,:74],state) or np.any(result['state'][0,74:]):
        raise ValueError('Native transform truncated/altered privileged state or padding')
    if training:result['action_mask'][s['action_is_pad']]=False
    return collate([result],transform.eagle_processor),result


def make_native_model(metadata_only=False,device='cpu'):
    import torch
    from torch import nn
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoModel
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    module=imports()
    from gr00t.model.backbone.eagle_backbone import EagleBackbone,DEFAULT_EAGLE_PATH
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    isolate_unused_radio()
    Eagle=get_class_from_dynamic_module('modeling_eagle2_5_vl.Eagle2_5_VLForConditionalGeneration',
        DEFAULT_EAGLE_PATH,local_files_only=True)
    class SDPAEagle(EagleBackbone):
        def __init__(self,tune_llm=False,tune_visual=False,select_layer=-1,reproject_vision=False,
                use_flash_attention=False,load_bf16=False,eagle_path=None,project_to_dim=1536):
            nn.Module.__init__(self)
            if reproject_vision:raise ValueError('Native selected configuration has no vision reprojection')
            config=AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH,trust_remote_code=True,local_files_only=True)
            if config.vision_config.model_type!='siglip_vision_model':raise ValueError('Only authentic selected SigLIP architecture')
            config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
            config.text_config._attn_implementation='sdpa'
            vision=SiglipVisionModel(config.vision_config);language=Qwen3ForCausalLM(config.text_config)
            self.eagle_model=Eagle(config,vision_model=vision,language_model=language)
            self.eagle_linear=nn.Linear(2048,project_to_dim) if project_to_dim is not None else nn.Identity()
            while len(self.eagle_model.language_model.model.layers)>select_layer:
                self.eagle_model.language_model.model.layers.pop(-1)
            self.select_layer=select_layer;self.set_trainable_parameters(tune_llm,tune_visual)
    original=module.EagleBackbone;module.EagleBackbone=SDPAEagle
    try:
        config=module.GR00T_N1_5_Config(**read(ASSETS/'gr00t_n15/config.json'))
        with init_empty_weights(include_buffers=False):model=module.GR00T_N1_5(config,str(ASSETS/'gr00t_n15'))
    finally:module.EagleBackbone=original
    expected=model.state_dict();index=read(ASSETS/'gr00t_n15/model.safetensors.index.json')['weight_map']
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
    missing=set(expected)-set(index);extras=set(index)-set(expected)
    allowed_aliases={name for group in aliases.values() if len(group)>1 and any(k in index for k in group) for name in group}
    if extras or missing-allowed_aliases:raise ValueError('Full native GR00T checkpoint key mismatch: '+str((extras,missing-allowed_aliases)))
    from safetensors import safe_open
    for shard in sorted(set(index.values())):
        with safe_open(ASSETS/'gr00t_n15'/shard,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key not in expected or list(expected[key].shape)!=list(f.get_slice(key).get_shape()):
                    raise ValueError('Native GR00T tensor shape mismatch: '+key)
    proof={'passed':True,'native_checkpoint_keys':len(index),'expected_native_keys':len(expected),
        'only_missing_proven_tied_aliases':sorted(missing),'attention_backend':'SDPA, unchanged dense native weights/shapes',
        'native_total_parameters':sum(p.numel() for p in model.parameters()),'metadata_only':metadata_only}
    if metadata_only:return model,proof
    from safetensors.torch import load_file
    for shard in sorted(set(index.values())):
        # Stream authentic shards directly to the admitted device, avoiding a
        # redundant5.4GB resident CPU copy beside the active R99 processes.
        state=load_file(str(ASSETS/'gr00t_n15'/shard),device=device)
        model.load_state_dict(state,strict=False,assign=True);del state
    for group in aliases.values():
        if len(group)<2:continue
        source=next((name for name in group if name in index),None)
        if source is None:continue
        p=dict(model.named_parameters(remove_duplicate=False))[source]
        for name in group:
            parent,key=name.rsplit('.',1);setattr(model.get_submodule(parent),key,p)
    if any(p.is_meta for p in model.parameters()):raise ValueError('Unloaded meta tensors remain in authentic full model')
    return model,proof


def slot_projectors(model):
    import torch
    from torch import nn
    from gr00t.model.action_head.flow_matching_action_head import CategorySpecificLinear
    class NativeSlot(nn.Module):
        def __init__(self,old):
            super().__init__();self.register_buffer('base_W',old.W.detach());self.register_buffer('base_b',old.b.detach())
            self.slot_W=nn.Parameter(old.W[31].detach().float().clone());self.slot_b=nn.Parameter(old.b[31].detach().float().clone())
        def forward(self,x,cat_ids):
            if not torch.all(cat_ids==31):raise ValueError('Only declared new embodiment31 admitted')
            return torch.bmm(x,self.slot_W[None].expand(len(x),-1,-1))+self.slot_b[None,None,:]
    for name,old in list(model.action_head.named_modules()):
        if isinstance(old,CategorySpecificLinear):
            parent,key=name.rsplit('.',1) if '.' in name else ('',name)
            setattr(model.action_head.get_submodule(parent) if parent else model.action_head,key,NativeSlot(old))


class Runtime:
    horizon=16
    def __init__(self,stats,*,device='cuda',checkpoint=None):
        import torch
        from torch import nn
        from peft import LoraConfig,inject_adapter_in_model
        self.stats=stats;self.device=device
        self.model,self.load_proof=make_native_model(device=device)
        self.model.requires_grad_(False)
        old=self.model.action_head.state_encoder.layer1
        expanded=nn.Parameter(torch.zeros(old.W.shape[0],80,old.W.shape[2],dtype=old.W.dtype))
        with torch.no_grad():expanded[:,:64].copy_(old.W)
        old.W=expanded;old.input_dim=80
        self.model.action_head.config.max_state_dim=80
        slot_projectors(self.model)
        inject_adapter_in_model(LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
            target_modules=['to_q','to_k','to_v','to_out.0']),self.model.action_head.model)
        # Native custom-projector rows + native action-expert LoRA. Other slots
        # stay bitwise frozen, avoiding full32-slot optimizer tensors on16GB.
        for name,p in self.model.named_parameters():
            if 'lora_' in name or name.endswith(('slot_W','slot_b')):p.requires_grad_(True);p.data=p.data.float()
        if checkpoint:
            from .checkpoint_io import load_adaptation
            load_adaptation(self.model,checkpoint)
        self.model.to(device)
        self.trainables={k:p for k,p in self.model.named_parameters() if p.requires_grad}
        self.load_proof.update({'state_input_expanded':[64,80],'valid_causal_features':74,
            'native_custom_embodiment':31,'native_projector_slots_trained':[31],
            'native_DiT_attention_LoRA_rank':8,'no_external_action_policy_or_target_residual':True})
        from gr00t.model.transforms import GR00TTransform
        from gr00t.data.embodiment_tags import EmbodimentTag
        self.transform=GR00TTransform(max_state_dim=80,max_action_dim=32,state_horizon=1,action_horizon=16,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,training=True)
        self.last_input={}

    def batch(self,s,*,training=False):
        import torch
        batch,_=native_batch(s,self.stats,self.transform,training)
        for name,value in batch.items():
            if isinstance(value,torch.Tensor):batch[name]=value.to(self.device)
        self.last_input={'actual_views':3,'learned_state_dim':80,'valid_state_features':74,
            'embodiment_id':31,'native_horizon':16,'future_data_in_observation':False,
            'native_Eagle_tokens':int(batch['eagle_input_ids'].shape[-1])}
        return batch

    def loss(self,s):
        import torch
        self.model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device=='cuda'):
            result=self.model(self.batch(s,training=True))
        return {'loss':result['loss'],'loss_components':{'action':result['loss']}}

    def predict(self,s,seed=0):
        import torch
        self.model.eval()
        with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device=='cuda'):
            torch.manual_seed(seed);prediction=self.model.get_action(self.batch(s))['action_pred'][0,:,:7].float().cpu().numpy()
        result=denormalize_actions(prediction,self.stats);result[:,6]=np.where(result[:,6]>0,1.,-1.)
        if result.shape!=(16,7) or not np.isfinite(result).all():raise ValueError('Malformed authentic GR00T action')
        return result
