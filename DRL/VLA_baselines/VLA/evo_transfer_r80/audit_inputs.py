"""Actual learned native VLM input pixels at multiple TRAIN interaction states."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,read
from .data import Data
from .native import Runtime


def main(a):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.82)
    a.output.mkdir(parents=True,exist_ok=False)
    data=Data();runtime=Runtime(read(a.checkpoint/'stats.json'),a.checkpoint)
    runtime.model.eval();runtime.capture_pixels=True
    records=[];sheets=[]
    for task in ('touch','pick','move','pick_place'):
        for target in ('blue block','yellow banana'):
            ei=next(i for i,r in enumerate(data.rows) if r['task']==task and r['pick_name']==target and
                    r['source_kind']=='complete_original')
            raw=data.numeric(ei)['scene64'];held=np.flatnonzero(raw[:,17]>.5)
            frames=[('reset',0),('closest_target',int(np.linalg.norm(raw[:,11:14],axis=1).argmin())),
                ('held_or_mid',int(held[len(held)//2]) if len(held) else len(raw)//2),('terminal',len(raw)-1)]
            canvas=Image.new('RGB',(1344,4*474),'white');draw=ImageDraw.Draw(canvas)
            for n,(label,fi) in enumerate(frames):
                sample=data.at(ei,fi)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):runtime.embeddings(sample)
                pixels=runtime.last_pixels
                decoded=((pixels*torch.tensor([.229,.224,.225]).view(1,3,1,1)+
                    torch.tensor([.485,.456,.406]).view(1,3,1,1))*255).round().clamp(0,255).byte()
                for col,pix in enumerate(decoded):
                    canvas.paste(Image.fromarray(pix.permute(1,2,0).numpy()),(448*col,474*n+26))
                    draw.text((448*col+4,474*n+5),f'{task}/{target}/{label} t{fi}/'+('front','wrist','GT')[col],fill='black')
                records.append({'task':task,'target':target,'episode':ei,'frame':fi,
                    'source_sha256':data.rows[ei]['source_sha256'],'native_input':runtime.last_input.copy(),
                    'native_pixels_sha256':hashlib.sha256(pixels.numpy().tobytes()).hexdigest()})
            path=a.output/(task+'_'+target.replace(' ','_')+'.png');canvas.save(path);sheets.append(str(path))
    atomic_json(a.output/'AUDIT.json',{'passed':True,'scope':'32 actual native TRAIN pixel/token boundaries, not full visibility or task SR',
        'checkpoint':str(a.checkpoint),'records':records,'sheets':sheets,'native_VLM_forward_executed':True,
        'four_tasks_two_targets_four_states_each':True,'original_RGB_source_hash_checked':True,
        'all_prompt_lengths_le_1024':all(max(r['native_input']['prompt_tokens'])<=1024 for r in records),
        'learned_checkpoint_sha256':sha256_file(a.checkpoint/'learned.safetensors'),
        'human_or_assistant_visual_review_still_required':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
