"""R91 authentic FALCON-ESM with qualified R100 paired depth and full reload."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .common import OUT,read
from .falcon_depth import DEST
from VLA.refinement_r99.data import Data as ParentData
from VLA.falcon_esm_r91.native import Runtime as NativeRuntime
from VLA.falcon_esm_r91.data import geometry,statistics
from VLA.common.provenance import sha256_file


def align_esm_patch_tokens(output,dtype):
    """Native DINO emits FP32 normalized features under BF16 autocast.

    Native ESM's indexed depth-token assignment requires its source and
    destination dtypes to agree. Cast the frozen encoder boundary, not its
    geometry, masks, selected weights, spatial network or downstream actions.
    """
    if not isinstance(output,dict) or 'x_norm_patchtokens' not in output:
        raise ValueError('Selected authentic DINO output schema changed')
    return {**output,'x_norm_patchtokens':output['x_norm_patchtokens'].to(dtype=dtype)}


class Data(ParentData):
    def __init__(self):
        super().__init__();proof=read(DEST/'DEPTH_ADMISSION.json')
        if not proof['passed'] or proof['episodes']!=len(self.rows):raise ValueError('Full original+recovery depth admission required')
        if sha256_file(DEST/'DEPTH_INDEX.json')!=proof['depth_index_sha256']:raise ValueError('Depth index changed')
        self.depth_rows=read(DEST/'DEPTH_INDEX.json');self.horizon=10
        for row,source in zip(self.depth_rows,self.rows,strict=True):
            if row['source_sha256']!=source['source_sha256'] or row['frames']!=source['frames']:
                raise ValueError('Actual RGB/depth/labels source alignment mismatch')

    @lru_cache(maxsize=2)
    def depth(self,ei):
        row=self.depth_rows[ei]
        if sha256_file(row['file'])!=row['sha256']:raise ValueError('Actual consumed depth payload changed')
        with np.load(row['file'],allow_pickle=False) as values:
            if not np.array_equal(values['source_frame_indices'],np.arange(row['frames'])):
                raise ValueError('Actual consumed depth chronology differs')
            return {k:values[k] for k in ('front_depth_z_m','view_world_to_OpenGL_camera','projection_OpenGL')}

    def at(self,ei,fi):
        s=super().at(ei,fi);d=self.depth(ei);projection=d['projection_OpenGL']
        if projection.ndim>1:projection=projection[fi]
        K,cv=geometry(d['view_world_to_OpenGL_camera'][fi],projection)
        s.update(depth_z_m=d['front_depth_z_m'][fi].astype(np.float32),intrinsics=K,extrinsics_OpenCV=cv)
        s['action'][s['action_is_pad'],:6]=0
        return s


class Runtime(NativeRuntime):
    def __init__(self,stats,*,device='cuda',checkpoint=None):
        super().__init__(stats,device=device)
        aggregator=self.model.act_head.vggt_net.aggregator
        aggregator.patch_embed.register_forward_hook(lambda module,args,result:
            align_esm_patch_tokens(result,aggregator.depth_placeholder.dtype))
        self.trainables=self.stage('joint_refinement')
        # Reentrant checkpointing discards gradients when the frozen image
        # stem produces inputs with requires_grad=False. Native LoRA INSIDE
        # that checkpoint must still train; non-reentrant checkpointing does
        # so without unfreezing the authentic frozen image stem or ESM.
        self.model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self._training_samples=0;self.branch_gradients={};self._gradient_groups={}
        for name,param in self.trainables.items():
            branch=None
            if 'lora_B' in name:
                branch='vision_LoRA' if name.startswith('backbone.vision_model.') else (
                    'language_LoRA' if name.startswith('backbone.text_model.') else 'image_to_text_LoRA')
            elif name=='state67.2.weight':branch='privileged67'
            elif name=='act_head.spatial_injector.proj1.0.weight':branch='native_ESM_spatial_injector'
            elif name=='act_head.actions.mlp.6.weight':branch='native_action_head'
            if branch:
                self.branch_gradients[name]=0.;self._gradient_groups[name]=branch
                def observe(grad,key=name):
                    if self._training_samples<=8:
                        self.branch_gradients[key]=max(self.branch_gradients[key],float(grad.detach().abs().sum()))
                param.register_hook(observe)
        if set(self._gradient_groups.values())!={'vision_LoRA','language_LoRA','image_to_text_LoRA',
                'privileged67','native_ESM_spatial_injector','native_action_head'}:
            raise ValueError('Expected actual native/privileged/vision/language gradient branches missing')
        if checkpoint:
            from .checkpoint_io import load_adaptation
            load_adaptation(self.model,checkpoint)
        self.load_proof['R100_native_embodiment_and_LoRA_joint_adaptation']=True
        self.load_proof['native_ESM_retained_not_3DPC_ablation']=True
        self.load_proof['frozen_DINO_normalized_patch_embedding_cast_to_native_ESM_dtype']=str(aggregator.depth_placeholder.dtype)
        self.load_proof['native_backbone_gradient_checkpointing']='non-reentrant, preserves visual LoRA behind frozen stem'

    def verify_gradients(self):
        if not self._training_samples:return
        missing=[n for n,v in self.branch_gradients.items() if v<=0]
        if missing:raise ValueError('Native adaptation gradient is disconnected: '+str(missing[:8]))
        self.load_proof['actual_native_branch_gradient_checks']={branch:{
            'parameters_checked':sum(b==branch for b in self._gradient_groups.values()),
            'positive_gradient_parameters':sum(self._gradient_groups[n]==branch and v>0 for n,v in self.branch_gradients.items()),
            'max_gradient_L1':max(v for n,v in self.branch_gradients.items() if self._gradient_groups[n]==branch)}
            for branch in sorted(set(self._gradient_groups.values()))}

    def loss(self,s):
        self._training_samples+=1
        loss,parts=super().loss(s)
        return {'loss':loss,'loss_components':{k:v for k,v in parts.items() if k.startswith('loss_')}}

    def predict(self,s,seed=0):
        import torch
        self.verify_gradients()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            return super().predict(s)
