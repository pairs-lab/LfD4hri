"""CPU input/label/weights audit, including actual native processor images."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.intern_r72.common import BASE,OUT,read,bindings
from VLA.intern_r72.data import Data,Inputs,targets,decode
from VLA.intern_r72.native import foundation


def unpatch(batch):
    pixels=batch['pixel_values'].numpy();offset=0;result=[]
    for t,h,w in batch['image_grid_thw'].numpy():
        n=int(t*h*w);p=16;m=2
        x=pixels[offset:offset+n].reshape(t,h//m,w//m,m,m,3,2,p,p)
        x=x.transpose(0,6,5,1,3,7,2,4,8).reshape(t*2,3,h*p,w*p)
        if not np.array_equal(x[0],x[1]): raise ValueError('Static RGB temporal patch convention differs')
        result.append(np.uint8(np.clip((x[0].transpose(1,2,0)*.5+.5)*255,0,255)))
        offset+=n
    if offset!=len(pixels): raise ValueError('Unaccounted native patch tokens')
    return result


def main():
    out=OUT/'cpu_preflight';out.mkdir(parents=True,exist_ok=False)
    sha=sha256_file(BASE/'model.safetensors')
    if sha!='9fe03a224b88f041c33dd1aae6bd24364bbae8d85158cf0a32ae9d237bd98851':raise ValueError('Pinned foundation SHA differs')
    revisions=read(OUT/'PROCESSOR_REVISIONS.json')
    actual={p.read_text().splitlines()[0] for p in (BASE/'qwen_assets/.cache/huggingface/download').glob('*.metadata')}
    if actual!={revisions['Qwen_metadata_expected']}:raise ValueError('Qwen processor metadata revision differs')
    data=Data();stats={**data.stats,'intern_action':data.intern_action_stats};inputs=Inputs(stats)
    atomic_json(out/'stats.json',stats);receipts=[];images=[];err=0.
    for task in ('touch','pick','move','pick_place'):
        ei=next(i for i,r in enumerate(data.rows) if r['task']==task and r['source_kind']=='complete_original')
        raw=data.numeric(ei)['scene64']
        contact=np.flatnonzero(raw[:,28:30].any(axis=1))
        frames=[0,int(contact[0]) if len(contact) else len(raw)-1]
        for fi in frames:
            sample=data.at(ei,fi);batch=inputs(sample,training=True)
            normalized=targets(sample['action'],sample['action_is_pad'],stats['intern_action'])
            valid=~sample['action_is_pad']
            err=max(err,float(np.abs(decode(normalized,stats['intern_action'])[valid]-sample['action'][valid]).max()))
            if normalized[sample['action_is_pad']].any():raise ValueError('Nonexistent target actions were fabricated')
            receipts.append({'task':task,'episode':ei,'frame':fi,**inputs.last_receipt,
                'valid_steps':int(valid.sum()),'future_or_teacher_phase_in_input':False})
            images.append(unpatch(batch))
    sheet=Image.new('RGB',(768,len(images)*282),'white');draw=ImageDraw.Draw(sheet)
    for row,(views,r) in enumerate(zip(images,receipts)):
        for col,pix in enumerate(views):
            sheet.paste(Image.fromarray(pix).resize((256,256)),(256*col,282*row+26))
            draw.text((256*col+2,282*row+4),f"{r['task']} t{r['frame']} / "+('front','wrist','GT')[col],fill='black')
    sheet.save(out/'ACTUAL_QWEN_PROCESSOR_INPUTS.png')
    model,loaded=foundation()
    atomic_json(out/'CPU_PREFLIGHT.json',{'passed_input_label_and_native_weight_loading':True,
        'native_optimizer_step_verified':False,'native_robot_SR_measured':False,
        'foundation_sha256':sha,'source':loaded,'TRAIN_episodes':len(data.rows),'TRAIN_frames':len(data),
        'action_roundtrip_max_abs_error':err,'receipts':receipts,'code':bindings(),
        'new_continuous_state_token67_plus_native_bins_and_measured_sensor_text':True,
        'FAST_tail_recipe':'variable-length REAL future prefix only; never pad/fabricate future motion',
        'flow_loss_recipe':'native flow per-slot loss; mask nonexistent timesteps and25 unused output dimensions',
        'no_camera_or_robot_or_controller_or_scoring_changes':True})
    print(json.dumps({'passed':True,'roundtrip_error':err,'native_inputs':receipts}),flush=True)


if __name__=='__main__':main()
