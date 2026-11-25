"""CPU-only native current-observation audit, all task/contact phases."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data, phase_anchors
from .common import DATA, BASE, QWEN, SCHEMA, sources, privileged_coverage


def jepa_audit(out):
    import torch
    from transformers import AutoProcessor, AutoConfig
    from lerobot.policies.vla_jepa.qwen_interface import Qwen3VLInterface
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
    from .jepa import config, stats, batch
    from PIL import Image,ImageDraw
    data=Data(DATA,horizon=7);stat=stats(data);cfg=config();cfg.device='cpu'
    qwen=Qwen3VLInterface.__new__(Qwen3VLInterface);torch.nn.Module.__init__(qwen)
    qwen.config=cfg;qwen.processor=AutoProcessor.from_pretrained(str(QWEN),local_files_only=True)
    qwen.processor.tokenizer.padding_side=cfg.tokenizer_padding_side
    rows=AutoConfig.from_pretrained(str(QWEN),local_files_only=True).text_config.vocab_size
    qwen.model=SimpleNamespace(device=torch.device('cpu'),
        get_input_embeddings=lambda:SimpleNamespace(weight=torch.empty((rows,1),device='meta')))
    tokens,_,_=qwen.expand_tokenizer()
    action_prompt=''.join(t*cfg.num_action_tokens_per_timestep for t in tokens[:3])
    embodied=cfg.embodied_action_token*cfg.num_embodied_action_tokens_per_instruction
    proxy=SimpleNamespace(config=cfg,model=SimpleNamespace(qwen=qwen,config=cfg))
    selected,events=phase_anchors(data);records=[];images_by_task={t:[] for t in data.groups}
    for ei,fi in selected:
        x=data.at(ei,fi);train=batch(x,stat,device='cpu',train=True);infer=batch(x,stat,device='cpu')
        for key in infer:
            if isinstance(infer[key],torch.Tensor):torch.testing.assert_close(infer[key],train[key],rtol=0,atol=0)
        one=VLAJEPAPolicy._prepare_model_inputs(proxy,train,training=False)
        two=VLAJEPAPolicy._prepare_model_inputs(proxy,infer,training=False)
        a=qwen.build_inputs(one['images'],one['instructions'],action_prompt,embodied)
        b=qwen.build_inputs(two['images'],two['instructions'],action_prompt,embodied)
        for key in a:
            if isinstance(a[key],torch.Tensor):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        if len(one['images'][0])!=3 or a['image_grid_thw'].shape[0]!=3 or one['state'].shape!=(1,1,67):
            raise ValueError('Native current-image/state input missing')
        if train['action'].shape!=(1,7,7):raise ValueError('Native JEPA horizon differs')
        decoded=train['action'].numpy()[0]*np.asarray(stat['action']['std'])+np.asarray(stat['action']['mean'])
        np.testing.assert_allclose(decoded,x['action'],rtol=0,atol=1e-7)
        if not torch.isfinite(a['pixel_values']).all() or not torch.isfinite(one['state']).all():
            raise ValueError('Nonfinite actual native inputs')
        pictures=[(p.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8') for p in one['images'][0]]
        images_by_task[x['task']].append((ei,fi,pictures,events.get((ei,fi),[])))
        records.append({'episode':ei,'frame':fi,'task':x['task'],
            'native_Qwen_pixel_values_shape':list(a['pixel_values'].shape),
            'native_image_grid_thw':a['image_grid_thw'].tolist(),
            'native_pixel_values_sha256':hashlib.sha256(a['pixel_values'].numpy().tobytes()).hexdigest(),
            'native_token_ids_sha256':hashlib.sha256(a['input_ids'].numpy().tobytes()).hexdigest(),
            'native_instruction_tokens':int(a['attention_mask'].sum()),
            'state67_train_runtime_exact':True,'action_roundtrip_max_error':float(abs(decoded-x['action']).max())})
    for task,rows in images_by_task.items():
        sheet=Image.new('RGB',(672,248*len(rows)),(245,245,245));draw=ImageDraw.Draw(sheet)
        for i,(ei,fi,pics,event) in enumerate(rows):
            draw.text((3,i*248+3),f'{task} TRAIN {ei}:{fi} {event}',fill='black')
            for vi,picture in enumerate(pics):sheet.paste(Image.fromarray(picture),(224*vi,i*248+24))
        sheet.save(out/(task+'_native_current_views.png'))
    counts=Counter();rng=np.random.default_rng(202665)
    for _ in range(4000):
        ei,fi=data.draw(rng);counts[data.rows[ei]['task']]+=1
        counts['XY_over5mm']+=int(np.linalg.norm(data.numeric(ei)['canonical7'][fi,:2])>.005)
    if min(counts[t] for t in data.groups)<700 or counts['XY_over5mm']<800:
        raise ValueError('Actual recovery sampler lacks balanced tasks/lateral motion')
    return stat,records,{'actual_sampler4000':dict(counts),'actual_native_Qwen_processor_exercised':True,
        'images_in_contact_sheets':'exact3 current224RGB tensors entering native Qwen processor; patch/token SHA recorded',
        'train_runtime_inputs_identical':True,'native_horizon':7}


def fast_audit(out):
    from openpi import transforms
    from openpi.training import data_loader
    from .fast import FastData, statistics, build_config, CODEC_SCALE, DECODING_STEPS
    from VLA.family_transfer_r54.pi0 import pi_sample
    from VLA.scene64_conditioning import parse_prompt
    from scipy.fft import dct
    data=FastData(DATA);stat=statistics(data);cfg=build_config(stat,out/'CPU_only_no_fit',2)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model);selected,events=phase_anchors(data);rows=[]
    # Fixed TRAIN-only recovery samples supplement all phase/contact anchors.
    rng=np.random.default_rng(202665)
    selected=list(dict.fromkeys(selected+[data.draw(rng) for _ in range(128)]))
    codec=next(t.tokenizer for t in dc.model_transforms.inputs if hasattr(t,'tokenizer'))._fast_tokenizer
    for ei,fi in selected:
        x=data.at(ei,fi);raw=pi_sample(x)
        np.testing.assert_allclose(parse_prompt(raw['prompt']),x['raw64'],rtol=0,atol=.000051)
        before=dc.data_transforms.inputs[0](raw)
        expected=transforms.Normalize(dc.norm_stats,use_quantiles=True)(before)
        # The official processor can silently clamp negative DCT coefficients.
        # Check the actual quantized coefficients too, not only their count.
        coefficients=np.around(dct(expected['actions'],axis=0,norm='ortho')*CODEC_SCALE).astype(np.int64)
        pieces=codec(expected['actions'][None])[0]
        recovered=np.asarray(list(map(ord,codec.bpe_tokenizer.decode(pieces))))+codec.min_token
        np.testing.assert_array_equal(recovered,coefficients.flatten())
        train=data_loader.transform_dataset([raw],dc)[0]
        no_action={k:v for k,v in raw.items() if k!='actions'}
        infer=transforms.compose([*dc.data_transforms.inputs,
            transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])(no_action)
        n=int(infer['tokenized_prompt_mask'].sum());nt=int(train['tokenized_prompt_mask'].sum())
        if max(n,nt)>=cfg.model.max_token_len:raise ValueError('Native FAST full64/32×32 labels truncate')
        postfix=int(train['token_loss_mask'].sum())
        if postfix>=DECODING_STEPS:raise ValueError('Native FAST inference decoding budget truncates TRAIN labels')
        np.testing.assert_array_equal(infer['tokenized_prompt'][:n],train['tokenized_prompt'][:n])
        if len(infer['image'])!=3 or not all(bool(v) for v in infer['image_mask'].values()):
            raise ValueError('Actual FAST GT3 missing')
        if np.any(train['token_loss_mask'][:n]) or not np.any(train['token_loss_mask']):
            raise ValueError('Actual FAST action/prefix loss mask wrong')
        out_transform=transforms.compose([*dc.model_transforms.outputs,
            transforms.Unnormalize(dc.norm_stats,use_quantiles=True),*dc.data_transforms.outputs])
        reconstructed=out_transform({'actions':train['tokenized_prompt'].copy(),'state':infer['state']})['actions']
        errors=abs(reconstructed-raw['actions']).max(axis=0)
        grip=bool(np.array_equal(reconstructed[:,6]>.5,raw['actions'][:,6]>.5))
        # Units differ: XYZ is metres; rotation is radians (not metres).
        # Explicit fidelity bounds:3mm translation,5mrad rotation, same grasp intent.
        if reconstructed.shape!=(32,7) or max(errors[:3])>.003 or max(errors[3:6])>.005 or not grip:
            atomic_json(out/'CODEC_FAILURE.json',{'episode':ei,'frame':fi,
                'error_per_channel':errors.tolist(),'gripper_intent_exact':grip})
            raise ValueError('Actual native FAST codec changes motor labels/gripper intent')
        rows.append({'episode':ei,'frame':fi,'task':x['task'],'prefix_tokens':n,'train_tokens':nt,
            'postfix_tokens':postfix,'native_codec_error_per_channel':errors.tolist(),
            'native_gripper_intent_exact':grip,'quantized_DCT_coefficients_exact':True,
            'strict_32x32_coefficients_checked_before_decoder':True})
    return stat,rows,{'native_horizon':32,'full64_prompt_NO_truncation':True,
        'native32_STATE_and3_views':True,'actual_native_label_codec_exercised':True,
        'TRAIN_only_codec_scale':CODEC_SCALE,'codec_bounds':{'XYZ_m':.003,'RPY_rad':.005},
        'original_phase_anchors_plus_fixed_TRAIN_recovery_samples':True}


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    stat,rows,extra=jepa_audit(a.output) if a.model=='vla_jepa' else fast_audit(a.output)
    atomic_json(a.output/'stats.json',stat)
    atomic_json(a.output/'audit.json',{'passed':True,'model':a.model,'schema':SCHEMA,
        'data_audit_sha256':sha256_file(DATA/'audit.json'),'code':sources(),
        'all4_tasks':set(r['task'] for r in rows)=={'touch','pick','move','pick_place'},
        'privileged_coverage':privileged_coverage(a.model),
        'CPU_only_no_model_weights_or_GPU':True,'samples':len(rows),'rows':rows,
        'actual_robot_episodes_established':False,**extra})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('vla_jepa','pi0_fast'),required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
