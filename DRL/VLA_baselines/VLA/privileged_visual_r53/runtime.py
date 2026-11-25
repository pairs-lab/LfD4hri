"""Add a THIRD visual observation, not a new action head/controller.

X-VLA already supports three views. Adapter's official variable-image vision
backbone and sequence-length-independent native head support768 task tokens
for3x256 patches. Preserve all trainable keys/shapes/parameter identities.
"""
from types import MethodType
import numpy as np
from PIL import Image
import torch

from VLA.new_vlas.runtime import create_runtime as original_create
from .scene_map import contract


def configure(runtime):
    before = runtime.parameters()
    batch_original = runtime.batch
    if runtime.family == 'xvla':
        if 'observation.images.image3' not in runtime.config.image_features or runtime.config.num_image_views != 3:
            raise ValueError('Official X-VLA third visual slot missing')
        original_prepare = runtime.model._prepare_images
        def prepare(policy,batch):
            images,mask = original_prepare(batch)
            if images.shape[1] != 3 or mask.tolist() != [[True,True,True]]:
                raise ValueError('Actual X-VLA forward did not consume all three views')
            runtime.r53_native_visual_receipt = {'actual_views':3,'actual_valid_mask':mask.tolist()}
            return images,mask
        runtime.model._prepare_images = MethodType(prepare,runtime.model)
    else:
        vision = runtime.model.get_base_model().vision_backbone
        if vision.get_num_patches() != 256 or runtime.head.num_task_tokens != 512:
            raise ValueError('Unexpected official native Adapter image/task layout')
        vision.set_num_images_in_input(3)
        runtime.head.num_task_tokens = 768
        original_forward = vision.forward
        def forward(module,pixels,*args,**kwargs):
            if pixels.shape[1] != 18:
                raise ValueError('Actual Adapter vision did not receive three6-channel fused views')
            features = original_forward(pixels,*args,**kwargs)
            if features.shape[1] != 768:
                raise ValueError('Actual Adapter vision token count is not3x256')
            runtime.r53_native_visual_receipt = {'actual_views':3,'actual_image_channels':18,'actual_visual_tokens':768}
            return features
        vision.forward = MethodType(forward,vision)
    def batch(this, sample):
        third = sample['privileged_map']
        if third.shape != (384,384,3) or third.dtype != np.uint8:
            raise ValueError('Exact causal privileged image required; no silent missing view')
        result = batch_original(sample)
        if this.family == 'xvla':
            mean = torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
            std = torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
            pixel = torch.as_tensor(third,device='cuda').permute(2,0,1).float()/255
            result['observation.images.image3'] = ((pixel-mean)/std)[None]
        else:
            extra = this.processor.image_processor(Image.fromarray(third),return_tensors='pt')['pixel_values']
            if extra.shape != (1,6,224,224) or result['pixel_values'].shape != (1,12,224,224):
                raise ValueError('Official DINO/SigLIP per-view channel layout changed')
            result['pixel_values'] = torch.cat((result['pixel_values'],extra.to(result['pixel_values'].device)),dim=1)
        return result
    runtime.batch = MethodType(batch,runtime)
    after = runtime.parameters()
    if set(before) != set(after) or any(before[k] is not after[k] for k in before):
        raise ValueError('Third observation must not replace/add any learned action parameter')
    runtime.privileged_visual_observation = contract()
    return runtime


def create_runtime(family,stats):
    return configure(original_create(family,stats))
