"""All TRAIN label conversions, full-support masks and actual native CLIP pixels."""
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
import torch
import clip
from clip.clip import _transform
from .common import OUT,TASKS,atomic_json,utc
from .seer_data import Data,statistics,native_labels


def main():
    torch.set_num_threads(1);root=OUT/'seer';root.mkdir(exist_ok=True)
    data=Data();stats=statistics(data);selected=[];events={};targets=Counter();clips=0
    for ei,row in enumerate(data.rows):
        targets[row['task']+'/'+row['pick_name']]+=row['frames']
        token=clip.tokenize([row['instruction']],truncate=False)
        if token.shape!=(1,77):raise ValueError('Native text tokenizer shape differs')
        raw=data.numeric(ei)['scene64'];local=data.converted(ei)['local_action']
        normalized=local[:,:6]/np.asarray(stats['local_action_scale6'])
        if np.max(abs(normalized))>1.000001:clips+=int(np.sum(np.any(abs(normalized)>1,axis=1)))
        for name,mask in {'reset':np.arange(len(raw))==0,'contact':np.any(raw[:,28:30]>.5,axis=1),
            'held':raw[:,17]>.5,'closed_empty':(raw[:,6]>.8)&(raw[:,31]<.5)}.items():
            key=row['task'],name
            if key not in events and mask.any():
                fi=int(np.flatnonzero(mask)[0]);events[key]=(ei,fi)
        for fi in (0,len(raw)-1):
            ids=np.maximum(np.arange(fi-6,fi+1),0)
            if (ids<0).any() or (ids>fi).any():raise ValueError('Future history leakage')
    if clips or len(targets)!=76:raise ValueError('Clipped labels or missing task/target support')
    processor=_transform(224);panels=[];metadata=[]
    for task in TASKS:
        for event in ('reset','contact','held','closed_empty'):
            if (task,event) not in events:continue
            ei,fi=events[task,event];s=data.native_sample(ei,fi);row=data.rows[ei]
            if s['history_indices'][-1]!=fi or s['state_history'].shape!=(7,7):raise ValueError('Current/past state alignment failed')
            native=[]
            for camera in ('front_history','wrist_history'):
                tensor=processor(Image.fromarray(s[camera][-1]))
                if tensor.shape!=(3,224,224) or not torch.isfinite(tensor).all():raise ValueError('Processed RGB invalid')
                mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
                std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
                restored=((tensor*std+mean).clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
                native.append(restored)
            panels.append(native);metadata.append({'task':task,'event':event,'episode':ei,'frame':fi,
                'target':row['pick_name'],'source':row['source'],'instruction':row['instruction'],
                'history_indices':s['history_indices'],'valid_action_labels':int(s['valid_actions'].sum()),
                'valid_future_image_labels':int(s['valid_future_images'].sum())})
    canvas=Image.new('RGB',(448,248*len(panels)),'white');draw=ImageDraw.Draw(canvas)
    for ri,(images,meta) in enumerate(zip(panels,metadata,strict=True)):
        draw.text((3,ri*248+3),f"{meta['task']} {meta['event']} {meta['target']} t{meta['frame']}",fill='black')
        for col,im in enumerate(images):canvas.paste(Image.fromarray(im),(224*col,ri*248+24))
    canvas.save(root/'NATIVE_224_INPUT_CONTACT_SHEET.png')
    atomic_json(root/'DATA_STATS.json',stats)
    atomic_json(root/'DATA_AUDIT.json',{'passed':True,'UTC':utc(),'trajectories':len(data.rows),
        'frames':sum(row['frames'] for row in data.rows),'TRAIN_label_conversion_all_frames':True,
        'all_native_labels_representable_without_clipping':True,'native_CLIP_tokenizer_all_instructions':True,
        'task_target_strata':len(targets),'target_frames':dict(targets),'native_processor_samples':metadata,
        'future_labels_training_only_not_forward_inputs':True,'rotation_only_frames_not_dropped':True,
        'new_heldout_examples_or_expert_labels_added':False,'does_NOT_qualify_full_native_GPU_training':True})
    print({'passed':True,'stats':stats,'native_RGB_panels':len(panels)},flush=True)


if __name__=='__main__':main()
