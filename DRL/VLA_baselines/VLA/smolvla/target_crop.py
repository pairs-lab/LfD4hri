"""A third native VLA image: causal GT-localized target, NOT an action plan.

Raw front/wrist views and all67 measured state features are retained. Only the
CURRENT visible target silhouette selects a context crop. Empty/occluded masks
fall back to the current full front view; there is no hidden object rerender,
future frame, grasp waypoint, action repair or change to PyBullet physics.
"""
from bisect import bisect_right
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import model_input, NUMERIC_SCHEMA
from VLA.smolvla.scene67_dataset import Scene67Dataset as OriginalScene67Dataset
from VLA.smolvla.adapter import SmolVLAPolicyAdapter

SCHEMA='SmolVLA_native67_plus_current_GT_target_context_crop_v1'
CAMERA='observation.images.target_crop'
NATIVE_CAMERA='observation.images.camera3'


def target_crop(rgb, mask):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or mask.dtype!=np.bool_ or mask.shape!=rgb.shape[:2]:
        raise ValueError('Matching current RGB uint8 and target silhouette bool required')
    y,x=np.nonzero(mask)
    if not len(x):return rgb.copy(),{'visible_pixels':0,'fallback_current_full_front':True,'box_xyxy':[0,0,rgb.shape[1],rgb.shape[0]]}
    # Symmetric 2x context, not an expert grasp point or target-centred action.
    side=max(32,int(np.ceil(2*max(int(x.max()-x.min()+1),int(y.max()-y.min()+1)))))
    side=min(side,min(rgb.shape[:2]));cx=(int(x.min())+int(x.max())+1)/2;cy=(int(y.min())+int(y.max())+1)/2
    left=int(np.clip(np.floor(cx-side/2),0,rgb.shape[1]-side));top=int(np.clip(np.floor(cy-side/2),0,rgb.shape[0]-side))
    box=[left,top,left+side,top+side]
    crop=np.asarray(Image.fromarray(rgb).crop(tuple(box)).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.BILINEAR)).copy()
    return crop,{'visible_pixels':len(x),'fallback_current_full_front':False,'box_xyxy':box}


def target_view(front,wrist,front_mask,wrist_mask):
    if front.shape!=wrist.shape or front_mask.shape!=wrist_mask.shape:
        raise ValueError('Matching real camera dimensions required')
    # Current visible pixels only. When the hand hides the front target, the
    # actual wrist image may still contain it; do not synthesize hidden RGB.
    source='wrist' if np.count_nonzero(wrist_mask)>np.count_nonzero(front_mask) else 'front'
    rgb,mask=(wrist,wrist_mask) if source=='wrist' else (front,front_mask)
    crop,receipt=target_crop(rgb,mask)
    return crop,{**receipt,'source_camera':source}


class TargetCropDataset(torch.utils.data.Dataset):
    def __init__(self,base,projection,recovery_admission=None):
        self.core=OriginalScene67Dataset(base,projection,recovery_admission)
        self.meta=deepcopy(self.core.meta)
        self.meta.info.features[CAMERA]=deepcopy(self.core.meta.info.features['observation.images.front'])
        self.meta.stats[CAMERA]=deepcopy(self.core.meta.stats['observation.images.front'])
        self.ends=np.cumsum([r['steps'] for r in self.core.parent_rows]).tolist()
        if CAMERA not in self.meta.camera_keys:raise ValueError('Third camera did not enter native metadata')
    def __getattr__(self,name):
        if name=='core':raise AttributeError(name)
        return getattr(self.core,name)
    def __len__(self):return len(self.core)
    @property
    def features(self):return self.meta.features
    @lru_cache(maxsize=1)
    def mask_episode(self,path):
        with np.load(path,allow_pickle=False) as z:return z['front_pick_mask'].copy(),z['wrist_pick_mask'].copy()
    def raw_masks(self,index):
        if index<self.core.original_length:
            ei=bisect_right(self.ends,index);fi=index-(self.ends[ei-1] if ei else 0)
            path=Path(self.core.audit['source'])/self.core.parent_rows[ei]['file']
        else:
            local=index-self.core.original_length;ei=bisect_right(self.core.correction_ends,local)
            fi=local-(self.core.correction_ends[ei-1] if ei else 0);path=Path(self.core.corrections[ei]['path'])
        return tuple(v[fi].copy() for v in self.mask_episode(str(path)))
    def __getitem__(self,index):
        sample=self.core[int(index)];front=sample['observation.images.front']
        views=[]
        for key in ('observation.images.front','observation.images.wrist'):
            image=sample[key][-1] if sample[key].ndim==4 else sample[key]
            views.append(np.rint(image.permute(1,2,0).numpy()*255).astype(np.uint8))
        crop,_=target_view(*views,*self.raw_masks(int(index)))
        tensor=torch.from_numpy(crop).permute(2,0,1).float().div(255)
        if front.ndim==4:tensor=tensor.unsqueeze(0)
        if tensor.shape!=front.shape:raise ValueError('Native temporal image axis changed')
        return {**sample,CAMERA:tensor}
    def __getitems__(self,indices):return [self[int(i)] for i in indices]


class TargetCropSmolVLA(SmolVLAPolicyAdapter):
    name='smolvla_scene67_target_crop';comparable=False
    def __init__(self,env,checkpoint,**kwargs):
        root=Path(checkpoint);contract=json.loads((root/'scene67_contract.json').read_text())
        crop=json.loads((root/'target_crop_contract.json').read_text());cfg=json.loads((root/'config.json').read_text())
        if (contract['schema']!=SCHEMA or crop['schema']!=SCHEMA or crop['raw_numeric_schema']!=NUMERIC_SCHEMA
                or NATIVE_CAMERA not in cfg['input_features'] or cfg['max_state_dim']!=67):
            raise ValueError('Own fitted67+GTcrop checkpoint required; do not omit the third view')
        for path,digest in crop['input_bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Privileged crop conditioning implementation changed')
        super().__init__(env,checkpoint=checkpoint,**kwargs)
    def _load(self):
        if self.model is not None:return
        super()._load()
        from safetensors import safe_open
        contract=json.loads((Path(self.checkpoint)/'fp32_master_contract.json').read_text());loaded=dict(self.model.named_parameters())
        with safe_open(Path(self.checkpoint)/'model.safetensors',framework='pt',device='cpu') as saved:
            for name in contract['trainable_parameter_names']:
                if not torch.equal(saved.get_tensor(name),loaded[name].detach().cpu()):raise ValueError('Lossy GTcrop FP32 reload')
        self.exact_trainable_reload_passed=True
    def build_sensor_batch(self,observation):
        batch=super().build_sensor_batch(observation);self.last_input=model_input(self.env,observation)
        batch['observation.state']=torch.from_numpy(self.last_input['numeric67']).unsqueeze(0)
        crop,receipt=target_view(observation.front_rgb,observation.wrist_rgb,
            self.env.target_mask('front','pick'),self.env.target_mask('wrist','pick'))
        batch[NATIVE_CAMERA]=torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float().div(255)
        self.last_input['target_crop_receipt']=receipt
        return batch
