"""Native Eagle preprocessing/state/action masks, without allocating3B weights."""
import numpy as np
import torch
from PIL import Image,ImageDraw
from .common import OUT,TASKS,atomic_json,utc
from .groot_native import Data,statistics,imports,isolate_unused_radio,native_batch


def main():
    torch.set_num_threads(1);imports();isolate_unused_radio()
    from gr00t.model.transforms import GR00TTransform
    from gr00t.data.embodiment_tags import EmbodimentTag
    data=Data();stats=statistics(data)
    transform=GR00TTransform(max_state_dim=80,max_action_dim=32,state_horizon=1,action_horizon=16,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,training=True)
    records=[];panels=[]
    for task in TASKS:
        ei=next(i for i,row in enumerate(data.rows) if row['task']==task and row['source_kind']=='complete_original')
        for fi in (0,data.rows[ei]['frames']//2,data.rows[ei]['frames']-1):
            s=data.at(ei,fi);batch,native=native_batch(s,stats,transform,True)
            infer,_=native_batch(s,stats,transform,False)
            for k in ('eagle_input_ids','eagle_pixel_values','eagle_attention_mask','state'):
                if not torch.equal(batch[k],infer[k]):raise ValueError('TRAIN/inference native preprocessing differs: '+k)
            if native['embodiment_id']!=31 or native['state_mask'].sum()!=74:
                raise ValueError('Wrong native embodiment or feature mask')
            expected=7*min(16,data.rows[ei]['frames']-fi)
            if native['action_mask'].sum()!=expected:raise ValueError('Wrong native valid action mask')
            pixels=batch['eagle_pixel_values']
            if not torch.isfinite(pixels).all() or pixels.shape[0]!=3:raise ValueError('Expected exactly3 selected-camera tiles')
            text=transform.eagle_processor.tokenizer.decode(batch['eagle_input_ids'][0],skip_special_tokens=True)
            if s['instruction'] not in text:raise ValueError('Requested command lost/truncated in native tokens')
            records.append({'task':task,'episode':ei,'frame':fi,'state_shape':list(batch['state'].shape),
                'pixel_shape':list(pixels.shape),'text_tokens':int(batch['eagle_input_ids'].shape[-1]),
                'valid_action_elements':int(native['action_mask'].sum()),'native_state_features':74})
            # Eagle config uses ImageNet normalization, inspect inversion of
            # its ACTUAL processor output, not independently resized RGB.
            mean=torch.tensor(transform.eagle_processor.image_processor.image_mean)[:,None,None]
            std=torch.tensor(transform.eagle_processor.image_processor.image_std)[:,None,None]
            display=((pixels.cpu()*std+mean).clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
            panels.append(display)
    root=OUT/'gr00t_n15';root.mkdir(exist_ok=True)
    canvas=Image.new('RGB',(672,248*len(panels)),'white');draw=ImageDraw.Draw(canvas)
    for ri,(images,meta) in enumerate(zip(panels,records,strict=True)):
        draw.text((3,ri*248+3),f"{meta['task']} frame{meta['frame']}",fill='black')
        for col,im in enumerate(images):canvas.paste(Image.fromarray(im),(224*col,ri*248+24))
    canvas.save(root/'NATIVE_EAGLE_INPUT_CONTACT_SHEET.png')
    atomic_json(root/'DATA_STATS.json',stats)
    atomic_json(root/'DATA_AUDIT.json',{'passed':True,'UTC':utc(),'trajectories':len(data.rows),
        'frames':sum(row['frames'] for row in data.rows),'native_processor_cases':records,
        'train_inference_tensor_parity':True,'custom_embodiment':31,'no_state_truncation_74_to80':True,
        'all_unexecuted_or_padded_action_channels_masked':True,'full_GPU_model_gate_still_required':True})
    print({'passed':True,'native_cases':len(records),'frames':sum(row['frames'] for row in data.rows)},flush=True)


if __name__=='__main__':main()
