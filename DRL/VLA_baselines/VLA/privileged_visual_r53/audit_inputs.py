"""CPU audit with official native image transforms, no model/SR assertion."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image,ImageDraw
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.fit_feedback_r52 import gate_anchors
from VLA.new_vlas.action_contracts import pack_native20,native8,normalize_bounds
from VLA.stable_native_r44.conditioning import install,bounded_append
from .data import MapData
from .scene_map import render,contract


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    install();torch.set_num_threads(1)
    data = MapData(args.data,verify_rgb_hashes=False)
    anchors = gate_anchors(data)
    if args.family == 'xvla':
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        holder = SimpleNamespace(config=XVLAConfig.from_pretrained('VLA/checkpoints/xvla-base'))
    else:
        from VLA.new_vlas.adapter_loading import import_native
        _,_,Processor,_,_,_ = import_native()
        processor = Processor.from_pretrained('VLA/checkpoints/vla-adapter-spatial-pro')
    sheet = Image.new('RGB',(672,248*len(anchors)), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    rows=[]
    for index,(ei,fi) in enumerate(anchors):
        sample = data.sample(ei,fi,args.family)
        row,raw = data.rows[ei],data.numeric(ei)['scene64'][fi]
        if sha256_file(row['source']) != row['source_sha256']:
            raise ValueError('Viewed source payload changed')
        with np.load(row['source'],allow_pickle=False) as a:
            for role in ('front','wrist'):
                np.testing.assert_array_equal(sample[role],a[role+'_rgb'][fi])
            xyz,six = a['ee_position'][fi],a['ee_rotation_6d'][fi]
        prefix = pack_native20(xyz,six,np.asarray(raw[6])) if args.family == 'xvla' else normalize_bounds(native8(xyz,six,raw[27]),data.stats['adapter']['proprio'])
        np.testing.assert_array_equal(sample['state'],bounded_append(prefix,raw,data.stats['scene67']))
        # Runtime receives the same pre-action64 in JSON, not post-action state.
        transported = np.asarray(json.loads(json.dumps(raw.tolist())),np.float32)
        np.testing.assert_array_equal(sample['privileged_map'],render(transported,sample['task']))
        roles = ('front','wrist','privileged_map')
        if args.family == 'xvla':
            mean,std = (torch.tensor(v)[:,None,None] for v in ([.485,.456,.406],[.229,.224,.225]))
            batch = {'observation.images.'+key:((torch.from_numpy(sample[role]).permute(2,0,1).float()/255-mean)/std)[None]
                for key,role in zip(('image','image2','image3'),roles,strict=True)}
            actual,mask = XVLAPolicy._prepare_images(holder,batch)
            old,_ = XVLAPolicy._prepare_images(holder,{k:v for k,v in batch.items() if not k.endswith('image3')})
            if actual.shape != (1,3,3,224,224) or mask.tolist() != [[True,True,True]]:
                raise ValueError('Official native third camera not active')
            torch.testing.assert_close(actual[:,:2],old[:,:2],rtol=0,atol=0)
            pixels = [actual[0,k]*std+mean for k in range(3)]
        else:
            pixels=[];fused=[]
            for role in roles:
                value=processor(Image.fromarray(sample[role]),return_tensors='pt')['pixel_values'][0]
                if value.shape != (6,224,224):raise ValueError('Native fused channel shape changed')
                fused.append(value)
                norm=processor.tvf_normalize_params[0]
                pixels.append(value[:3]*torch.tensor(norm['std'])[:,None,None]+torch.tensor(norm['mean'])[:,None,None])
            full = torch.cat(fused,dim=0)
            torch.testing.assert_close(full[:12],torch.cat(fused[:2],dim=0),rtol=0,atol=0)
            if full.shape != (18,224,224):raise ValueError('Third fused image missing')
        for k,tensor in enumerate(pixels):
            im=Image.fromarray(np.uint8(np.rint(np.clip(tensor.permute(1,2,0).numpy(),0,1)*255)))
            sheet.paste(im,(k*224,index*248+24))
        draw.text((3,index*248+3),sample['task']+' / '+row['source_kind'],fill='#111111')
        rows.append({'episode':ei,'frame':fi,'task':sample['task'],'source_kind':row['source_kind'],
            'GT_map_RGB_sha256':hashlib.sha256(sample['privileged_map'].tobytes()).hexdigest(),
            'JSON_runtime_raw64_to_training_GT_map_byte_exact':True,
            'two_original_RGB_views_unchanged_and_native_prefix_parity':True})
    sheet.save(args.output/'actual_three_view_native224.png')
    atomic_json(args.output/'audit.json',{'passed':True,'family':args.family,'profile':contract(),
        'scope':'8_representative_TRAIN_native_image_and_state_input_checks_NOT_actual_GPU_model_or_SR',
        'native_real_GPU_gradient_memory_gate_still_required':True,
        'data_audit_sha256':sha256_file(args.data/'audit.json'),
        'two_original_camera_inputs_unchanged':True,'extra_simulator_steps_or_actions':0,'rows':rows,
        'script_sha256':sha256_file(Path(__file__))})
    print(json.dumps({'family':args.family,'passed':True,'anchors':len(rows),'image':str(args.output/'actual_three_view_native224.png')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('xvla','vla_adapter'),required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
