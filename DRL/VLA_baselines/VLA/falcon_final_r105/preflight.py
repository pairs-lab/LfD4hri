"""Real full-model CUDA reload and grasp-stage native input inspection."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import OUT, PARENT, read, atomic_json, utc
from VLA.common.provenance import sha256_file


def main(args):
    import torch
    from VLA.admission_r100.falcon_native import Data, Runtime
    from VLA.admission_r100.train import qualified
    from VLA.admission_r100.workflow import foundations
    from VLA.admission_r100.common import verify
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    checkpoint = args.checkpoint.resolve(); meta = qualified(checkpoint,'falcon_esm')
    verify(foundations('falcon_esm'))
    data = Data(); runtime = Runtime(meta['stats'],checkpoint=checkpoint)
    refs = read(checkpoint/'RELOAD_ANCHORS.json'); errors=[]
    for i,((ei,fi),wanted) in enumerate(zip(refs['anchors'],refs['predictions'],strict=True)):
        value = runtime.predict(data.at(ei,fi),2026100+i)
        errors.append(float(np.max(np.abs(value-np.asarray(wanted,np.float32)))))
    if len(errors)!=8 or max(errors)>1e-6:
        raise ValueError('Actual fresh CUDA resume/reload anchors differ')
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    result = {'passed':True,'UTC':utc(),'checkpoint':str(checkpoint),
        'actual_full_native_model':True,'fresh_reload_anchor_max_errors':errors,
        'weights_sha256':sha256_file(checkpoint/'weights.safetensors'),
        'CPU_test_NOT_substituted_for_actual_CUDA_forward':True}
    if args.inspect:
        audit = read(OUT/'CPU_AUDIT.json'); records=[]; sheets={}
        color_mean = torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
        color_std = torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
        for case in audit['native_images_to_review']:
            s=data.at(case['episode'],case['frame'])
            if sha256_file(case['source'])!=case['source_sha256']:
                raise ValueError('Selected real input source changed')
            pred=runtime.predict(s,20261026)
            batch=runtime.batch(s)
            images=[]
            for key in ('vision_x','vision_gripper'):
                actual=batch[key][0,0].detach().cpu()
                images.append(((actual*color_std+color_mean).clamp(0,1)*255).round().byte().permute(1,2,0).numpy())
            images.append((batch['rgb_vggt'][0,0].detach().cpu().clamp(0,1)*255).round().byte().permute(1,2,0).numpy())
            depth=batch['static_depth_vggt'][0,0,:,:,0].detach().cpu().numpy()
            images.append(np.repeat((np.clip(depth/3,0,1)*255).astype(np.uint8)[...,None],3,axis=2))
            for value in batch.values():
                if not torch.isfinite(value).all(): raise ValueError('Nonfinite actual native input')
            if pred.shape!=(10,7) or runtime._state.shape!=(1,67):
                raise ValueError('Native horizon or privileged input was truncated')
            text=runtime.model.tokenizer.decode(batch['lang_x'][0],skip_special_tokens=True)
            if s['instruction'] not in text:
                raise ValueError('Requested object/goal instruction missing from native tokens')
            with np.load(s_data_source(data,case['episode']),allow_pickle=False) as original:
                visibility={key:int(np.count_nonzero(original[key][case['frame']]))
                    for key in ('front_pick_mask','wrist_pick_mask') if key in original.files}
            record={**case,'decoded_command':text,'predicted_first_canonical7':pred[0].tolist(),
                'teacher_first_canonical7':s['action'][0].tolist(),
                'first_action_absolute_error':np.abs(pred[0]-s['action'][0]).tolist(),
                'valid_native_depth_pixels':int(batch['point_masks_vggt'].sum()),
                'actual_native_depth_range_m':[float(depth.min()),float(depth.max())],
                'original384_target_segmentation_pixels':visibility,
                'numeric_clipped_feature_count':int((runtime._state.abs()>=5).sum()),
                'correct_bilateral_hold_at_this_input':bool(s['raw64'][17]>.5)}
            records.append(record);sheets.setdefault(case['task'],[]).append((record,images))
            print({'inspected':len(records),'task':case['task'],'stage':case['stage']},flush=True)
        for task,panels in sheets.items():
            sheet=Image.new('RGB',(896,248*len(panels)),'white');draw=ImageDraw.Draw(sheet)
            for i,(record,images) in enumerate(panels):
                draw.text((3,i*248+3),f"{task} {record['stage']} frame{record['frame']}: front / wrist / ESM / depth0..3m",fill='black')
                for j,img in enumerate(images):sheet.paste(Image.fromarray(img),(j*224,i*248+24))
            sheet.save(output/f'{task}_native_inputs.png')
        result.update(actual_native_input_cases=records,
            image_files=[str(output/f'{task}_native_inputs.png') for task in sheets],
            future_phase_and_teacher_actions_are_labels_only_NOT_input=True,
            model_inputs_and_camera_settings_unchanged=True)
    result['peak_GPU_MiB']=torch.cuda.max_memory_allocated()/2**20
    atomic_json(output/'GPU_PREFLIGHT.json',result)
    print({'passed':True,'reload_max_error':max(errors),'output':str(output)},flush=True)


def s_data_source(data,episode):
    return data.rows[episode]['source']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=PARENT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=.86)
    main(p.parse_args())
