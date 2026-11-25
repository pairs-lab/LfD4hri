"""Export the ACTUAL normalized native OpenPI two-camera input for inspection."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import Scene64NativeDataset
from VLA.pi0.scene64_training import build_config


def main():
    from openpi.training import data_loader
    from openpi.models.model import Observation
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('projection','batch-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=('pi0','pi0_fast'),required=True);a=p.parse_args()
    proof=json.loads(a.batch_audit.read_text())
    if (not proof['passed'] or proof['model']!=a.model or proof['implementation_sha256']!=sha256_file(Path(__file__).with_name('scene64_training.py'))
        or proof['projection_audit_sha256']!=sha256_file(a.projection/'audit.json')):raise ValueError('Actual native batch proof stale')
    stats=json.loads(a.batch_audit.with_name('norm_stats_values.json').read_text())
    data=Scene64NativeDataset(a.projection,5,pilot=proof['pilot_only']);cfg=build_config(a.model,stats,a.output,2)
    native=data_loader.transform_dataset(data,cfg.data.create(cfg.assets_dirs,cfg.model));a.output.mkdir(parents=True,exist_ok=False)
    selected=[];tasks=set()
    for ei,row in enumerate(data.rows):
        task=row['task']['name']
        if task in tasks:continue
        tasks.add(task);first=0 if ei==0 else data.ends[ei-1]
        selected.extend((task,phase,first+offset) for phase,offset in [('initial',0),('middle',row['steps']//2),('terminal',row['steps']-1)])
    sheet=Image.new('RGB',(224*2,250*len(selected)),'white');draw=ImageDraw.Draw(sheet);rows=[]
    for number,(task,phase,index) in enumerate(selected):
        actual=native[index];network=copy.deepcopy(actual)
        # The actual loader converts scalar transform masks to ndarray
        # leaves before Observation.from_dict. Preserve that typed boundary
        # here instead of passing unbatched np.bool_ scalar sentinels.
        network['image_mask']={k:np.asarray(v) for k,v in network['image_mask'].items()}
        observation=Observation.from_dict(network)
        names=['base_0_rgb','wrist_0_rgb' if a.model=='pi0_fast' else 'left_wrist_0_rgb']
        draw.text((4,number*250),f'{task} {phase} frame={index}',fill='black')
        for col,name in enumerate(names):
            normalized=np.asarray(observation.images[name]);pixels=np.asarray(actual['image'][name])
            if normalized.shape!=(224,224,3) or pixels.shape!=(224,224,3):raise ValueError('Actual image shape differs')
            np.testing.assert_allclose(normalized,pixels.astype(np.float32)/127.5-1,rtol=0,atol=2e-7)
            np.save(a.output/f'{number}_{name}_actual_normalized.npy',normalized,allow_pickle=False)
            Image.fromarray(pixels.astype(np.uint8)).save(a.output/f'{number}_{name}.png')
            sheet.paste(Image.fromarray(pixels.astype(np.uint8)),(col*224,number*250+25))
        rows.append({'task':task,'phase':phase,'index':index,'actual_prompt_tokens':int(actual['tokenized_prompt_mask'].sum()),
            'actual_image_mask':{k:bool(v) for k,v in observation.image_masks.items()}})
    sheet.save(a.output/'actual_two_camera_224_contact_sheet.png')
    atomic_json(a.output/'export.json',{'passed':True,'model':a.model,'actual_native_image_normalization_checked':True,
        'rows':rows,'batch_audit_sha256':sha256_file(a.batch_audit),'model_weights_loaded':False,
        'human_visual_reviewed':False,'all_frames_visible_or_success_certified':False})


if __name__=='__main__':main()
