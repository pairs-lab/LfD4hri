"""CPU-only real native preprocessing; no neural success claims."""
import argparse
from collections import Counter
from types import SimpleNamespace
import hashlib
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import DATA, SCHEMA, hashes
from .data import Data, anchors,phase_anchors


def main(a):
    from PIL import Image,ImageDraw
    a.output.mkdir(parents=True,exist_ok=False)
    data=Data(a.data,verify_rgb_hashes=a.verify_all_rgb)
    rng=np.random.default_rng(202654);counts=Counter()
    for _ in range(4000):
        ei,fi=data.draw(rng);row=data.rows[ei];action=data.numeric(ei)['canonical7'][fi]
        counts['task_'+row['task']]+=1;counts['source_'+row['source_kind']]+=1
        counts['XY_over5mm']+=int(np.linalg.norm(action[:2])>.005)
    if min(counts['task_'+t] for t in ('touch','pick','move','pick_place'))<700 or counts['XY_over5mm']<800:
        raise ValueError('Actual family sampler failed coverage/minimum lateral exposure')
    rows=[];pictures=[]
    selected,events=phase_anchors(data) if a.phase_events else (anchors(data),{})
    if a.model=='smolvla':
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from .smol import config,processors,statistics,prepared
        cfg=config(device='cpu');stats=statistics(data);pre,_=processors(cfg,stats)
        proxy=SimpleNamespace(config=cfg)
        for ei,fi in selected:
            x=data.at(ei,fi);train=prepared(pre,x,train=True);infer=prepared(pre,x)
            for k in infer:
                if isinstance(infer[k],torch.Tensor):torch.testing.assert_close(infer[k],train[k],rtol=0,atol=0)
            images,masks=SmolVLAPolicy.prepare_images(proxy,train)
            if len(images)!=3 or not all(bool(v.all()) for v in masks):raise ValueError('Third actual Smol view missing')
            ordinary={k:v for k,v in train.items() if k!='observation.images.camera3'}
            old_images,_=SmolVLAPolicy.prepare_images(proxy,ordinary)
            for native,old in zip(images[:2],old_images,strict=True):torch.testing.assert_close(native,old,rtol=0,atol=0)
            expected=np.clip((x['numeric67']-np.asarray(stats['observation.state']['mean'],np.float32))/
                (np.asarray(stats['observation.state']['std'],np.float32)+np.float32(1e-8)),-5,5)
            np.testing.assert_allclose(infer['observation.state'].numpy()[0],expected,atol=2e-6,rtol=0)
            native=[((v[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8') for v in images]
            pictures.append(native)
            if train['action'].shape!=(1,5,7):raise ValueError('Native action horizon changed')
            # Exercise the REAL official output processor on an entire chunk,
            # not only a single action accepted by the old select_action API.
            _,post=processors(cfg,stats)
            decoded=post(train['action']).numpy()[0]
            np.testing.assert_allclose(decoded,x['action'],atol=1e-7,rtol=0)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(cfg.vlm_model_name,local_files_only=True)
            full=tokenizer(x['instruction']+'\n',truncation=False)['input_ids']
            if len(full)>cfg.tokenizer_max_length:raise ValueError('Actual original language truncates')
            rows.append({'episode':ei,'frame':fi,'task':x['task'],'native_images':[list(v.shape) for v in images],
                'map_sha256':hashlib.sha256(x['map'].tobytes()).hexdigest(),'tokens':len(full),
                'train_runtime_state_tokens_RGB_exact':True})
    else:
        from openpi import transforms
        from openpi.training import data_loader
        from .pi0 import statistics,build_config,pi_sample
        stats=statistics(data);cfg=build_config(stats,a.output/'unused_CPU_config',2)
        dc=cfg.data.create(cfg.assets_dirs,cfg.model)
        for ei,fi in selected:
            x=data.at(ei,fi);raw=pi_sample(x)
            transformed=data_loader.transform_dataset([raw],dc)[0]
            no_action={k:v for k,v in raw.items() if k!='actions'}
            infer=transforms.compose([*dc.data_transforms.inputs,
                transforms.Normalize(dc.norm_stats,use_quantiles=False),*dc.model_transforms.inputs])(no_action)
            n=int(infer['tokenized_prompt_mask'].sum())
            if n>=cfg.model.max_token_len:raise ValueError('Actual full scene64 language truncated')
            for k in ('state','tokenized_prompt','tokenized_prompt_mask'):
                np.testing.assert_array_equal(infer[k],transformed[k])
            if len(infer['image'])!=3 or not all(bool(v) for v in infer['image_mask'].values()):raise ValueError('Native pi0 map3 dropped')
            for key in infer['image']:np.testing.assert_array_equal(infer['image'][key],transformed['image'][key])
            np.testing.assert_array_equal(raw['actions'][:,:6],x['action'][:,:6])
            np.testing.assert_array_equal(raw['actions'][:,6],(x['action'][:,6]+1)/2)
            if transformed['actions'].shape!=(5,32):raise ValueError('Native pi0 padded action shape differs')
            # Official model transforms retain uint8 here; model parses to[-1,1].
            native=[]
            for v in infer['image'].values():
                v=np.asarray(v)
                if v.dtype!=np.uint8:v=((v+1)*127.5).clip(0,255).astype('uint8')
                native.append(v)
            pictures.append(native)
            rows.append({'episode':ei,'frame':fi,'task':x['task'],'native_images':[list(v.shape) for v in native],
                'map_sha256':hashlib.sha256(x['map'].tobytes()).hexdigest(),'tokens':n,
                'EEF_delta_NO_joint_DeltaActions_or_AbsoluteActions':True,'train_runtime_state_tokens_RGB_exact':True})
    size=256 if a.model=='smolvla' else 224
    sheet=Image.new('RGB',(3*size,len(pictures)*(size+24)),(245,245,245));d=ImageDraw.Draw(sheet)
    for ri,views in enumerate(pictures):
        d.text((4,ri*(size+24)+3),f"{rows[ri]['task']} TRAIN {rows[ri]['episode']}:{rows[ri]['frame']}",fill='black')
        for vi,v in enumerate(views):sheet.paste(Image.fromarray(v).resize((size,size)),(vi*size,ri*(size+24)+24))
    sheet.save(a.output/'actual_native_three_view.png')
    for task in ('touch','pick','move','pick_place'):
        ids=[i for i,r in enumerate(rows) if r['task']==task]
        if not ids:raise ValueError('Missing task in native input audit')
        task_sheet=Image.new('RGB',(3*size,len(ids)*(size+24)),(245,245,245));draw=ImageDraw.Draw(task_sheet)
        for ri,i in enumerate(ids):
            row=rows[i];row['diagnostic_events_NOT_policy_input']=events.get((row['episode'],row['frame']),[])
            draw.text((4,ri*(size+24)+3),str(row['diagnostic_events_NOT_policy_input']),fill='black')
            for vi,v in enumerate(pictures[i]):task_sheet.paste(Image.fromarray(v).resize((size,size)),(vi*size,ri*(size+24)+24))
        task_sheet.save(a.output/(task+'_native_preview.png'))
    atomic_json(a.output/'stats.json',stats)
    atomic_json(a.output/'audit.json',{'passed':True,'schema':SCHEMA,'model':a.model,'data':str(a.data.resolve()),
        'data_audit_sha256':sha256_file(a.data/'audit.json'),'code':hashes(),'samples':len(rows),'rows':rows,
        'sampler4000':dict(counts),'all_source_payload_hashes_rechecked':a.verify_all_rgb,
        'scope':'ACTUAL_native_CPU_preprocessing_NOT_GPU_fit_or_model_SR','neural_weights_loaded':False,
        'original_two_RGB_preserved':True,'native_state_dimension':67 if a.model=='smolvla' else 32,
        'native_horizon':5,'physical_action_dimensions':7,'model_SR':None})


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('smolvla','pi0'),required=True)
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-all-rgb',action='store_true');p.add_argument('--phase-events',action='store_true');main(p.parse_args())
