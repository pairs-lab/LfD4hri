"""Export actual native-preprocessed TRAIN rotation examples for visual review."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from VLA.common.artifact_io import atomic_json
from VLA.flower_rotation_r76.sampler import Data
from VLA.flower_r69.native import image_tensor,action_targets,canonical_actions,MEAN,STD


def main(out):
    torch.set_num_threads(2);data=Data();out.mkdir(parents=True,exist_ok=False)
    selected=[]
    for task in ('pick','move','pick_place'):
        candidates=[]
        for ei,row in enumerate(data.rows):
            if row['task']!=task or row['source_kind']!='complete_original':continue
            n=data.numeric(ei);a=n['canonical7'];raw=n['scene64']
            flag=(np.max(abs(a[:,3:6]),axis=1)>.01)&(np.linalg.norm(a[:,:3],axis=1)<=.001)&((a[:,6]>.5)==(raw[:,10]>.5))
            ids=np.flatnonzero(flag)
            if len(ids):candidates.append((ei,int(ids[len(ids)//2])))
        if not candidates:raise ValueError('No actual rotation-only TRAIN example: '+task)
        ei,fi=next(((ei,fi) for ei,fi in candidates if 'gelatin' in data.rows[ei]['pick_name']),candidates[0])
        selected.extend((ei,int(np.clip(fi+offset,0,data.rows[ei]['frames']-1))) for offset in (-2,0,2))
    canvas=Image.new('RGB',(672,9*250),'white');draw=ImageDraw.Draw(canvas);records=[]
    for index,(ei,fi) in enumerate(selected):
        s=data.at(ei,fi);row=data.rows[ei];tensors=[]
        for col,name in enumerate(('front','wrist','map')):
            tensor=image_tensor(s[name]);tensors.append(tensor)
            rgb=np.clip(np.rint((tensor.permute(1,2,0).numpy()*np.asarray(STD)+np.asarray(MEAN))*255),0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(col*224,index*250+26))
        draw.text((3,index*250+3),f"{s['task']} t{fi} {row['pick_name']} rpy="+','.join(f'{x:.3f}' for x in s['action'][0,3:6]),fill='black')
        native=action_targets(s['action'],data.stats);decoded=canonical_actions(native,data.stats)
        error=float(np.max(abs(decoded-s['action'])))
        if error>5e-4:raise ValueError('Action conversion error beyond previously declared boundary clipping')
        if s['numeric67'].shape!=(67,):raise ValueError('Incomplete causal numeric input')
        records.append({'episode':ei,'frame':fi,'task':s['task'],'target':row['pick_name'],
            'source':row['source'],'source_sha256':row['source_sha256'],
            'preprocessed_shape':[3,3,224,224],
            'native_input_images_sha256':hashlib.sha256(torch.stack(tensors).numpy().tobytes()).hexdigest(),
            'canonical_local_rotation_label':s['action'][0,3:6].tolist(),
            'action_roundtrip_max_error':error,'label_not_added_to_model_observation':True})
    canvas.save(out/'ACTUAL_NATIVE_ROTATION_INPUTS.png')
    atomic_json(out/'INPUT_AUDIT.json',{'passed':True,'actual_native_preprocessing_used':True,
        'TRAIN_only':True,'frames':records,'normalizer_unchanged':True,'no_model_SR_claim':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
