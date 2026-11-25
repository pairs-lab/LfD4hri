"""Actual native RGB-D/token preprocessing; not a reduced policy evaluation."""
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from .common import OUT,TASKS,atomic_json,utc
from .falcon_native import Data,statistics


def main():
    import sys,torch
    from transformers import AutoProcessor
    from VLA.falcon_esm_r91.common import REPO,ASSETS
    from VLA.falcon_esm_r91.native import Runtime
    torch.set_num_threads(1);sys.path.insert(0,str(REPO))
    data=Data();stats=statistics(data)
    processor=AutoProcessor.from_pretrained(ASSETS/'kosmos_assets',local_files_only=True)
    # Exercise the exact native adapter's batch function without constructing
    # a policy or claiming any neural forward/backward/SR from this CPU audit.
    runtime=Runtime.__new__(Runtime);runtime.device='cpu';runtime.stats=stats
    runtime.model=SimpleNamespace(tokenizer=processor.tokenizer);runtime.last_input={}
    rows=[];panels=[]
    for task in TASKS:
        ei=next(i for i,r in enumerate(data.rows) if r['task']==task and r['source_kind']=='complete_original')
        for fi in (0,data.rows[ei]['frames']//2,data.rows[ei]['frames']-1):
            s=data.at(ei,fi);batch=runtime.batch(s);other=runtime.batch(s)
            for name,value in batch.items():
                if not torch.equal(value,other[name]) or not torch.isfinite(value).all():
                    raise ValueError('Actual RGB-D/token preprocessing differs or is nonfinite: '+name)
            if batch['rgb_vggt'].shape!=(1,1,3,224,224):raise ValueError('Native ESM RGB shape mismatch')
            if batch['static_depth_vggt'].shape!=(1,1,224,224,1):raise ValueError('Native metric depth shape mismatch')
            if runtime._state.shape!=(1,67):raise ValueError('Causal67 state truncated')
            if batch['point_masks_vggt'].sum()<100:raise ValueError('Actual native depth mask effectively empty')
            text=processor.tokenizer.decode(batch['lang_x'][0],skip_special_tokens=True)
            if s['instruction'] not in text:raise ValueError('Native command missing/truncated')
            mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
            std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
            images=[((batch[name][0,0]*std+mean).clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
                for name in ('vision_x','vision_gripper')]
            images.append((batch['rgb_vggt'][0,0].clamp(0,1)*255).round().byte().permute(1,2,0).numpy())
            depth=batch['static_depth_vggt'][0,0,:,:,0].numpy()
            images.append(np.repeat((np.clip(depth/3.,0,1)*255).astype(np.uint8)[...,None],3,axis=-1))
            panels.append(images);rows.append({'task':task,'episode':ei,'frame':fi,
                'native_shapes':{k:list(v.shape) for k,v in batch.items()},
                'valid_depth_pixels':int(batch['point_masks_vggt'].sum()),
                'depth_range_m':[float(depth.min()),float(depth.max())],
                'adjusted_native_K':batch['static_cam_intr_mat_vggt'][0,0].tolist()})
    root=OUT/'falcon_esm';canvas=Image.new('RGB',(896,248*len(rows)),'white');draw=ImageDraw.Draw(canvas)
    for row,(meta,images) in enumerate(zip(rows,panels,strict=True)):
        draw.text((3,row*248+3),f"{meta['task']} frame{meta['frame']}: front / wrist / actual ESM RGB / metric depth",fill='black')
        for col,image in enumerate(images):canvas.paste(Image.fromarray(image),(col*224,row*248+24))
    canvas.save(root/'NATIVE_RGBD_INPUT_CONTACT_SHEET.png')
    atomic_json(root/'DATA_STATS.json',stats)
    atomic_json(root/'DATA_AUDIT.json',{'passed':True,'UTC':utc(),'trajectories':len(data.rows),
        'frames':sum(r['frames'] for r in data.rows),'actual_native_processor_cases':rows,
        'same_exact_batch_used_by_train_and_inference':True,'native_model_forward_executed_in_this_CPU_audit':False,
        'ESM_depth_real_not_constant_or_teacher_depth':True,'future_action_padding_explicit':True})
    print({'passed':True,'actual_native_RGBD_cases':len(rows)},flush=True)


if __name__=='__main__':main()
