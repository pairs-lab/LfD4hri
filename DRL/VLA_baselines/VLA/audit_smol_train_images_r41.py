"""Export the exact original TRAIN20 images, masks and proposed GT-view.

CPU-only, no simulation, NN allocation, training or800 outcome access.
The third view is the EXISTING frozen target_view function, not a recreated
crop. Visual clarity is reviewed by opening the sheets, never certified by
mask pixel count alone. All20 examples are exported, not just good cases.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import validate_scene
from VLA.smolvla.target_crop import target_view

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
FIXTURE=BASE/'native_TRAIN_action_probe_r37_v2'
PREDICTIONS=BASE/'Smol_actual_TRAIN20_now_r40_v2/actual_TRAIN20/audit.json'


@lru_cache(maxsize=1)
def payload(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('front_rgb','wrist_rgb','front_pick_mask','wrist_pick_mask',
            'joint_positions','gripper_closure','prompt','canonical_actions')}


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    proof=json.loads((FIXTURE/'audit.json').read_text())
    if not proof['passed'] or sha256_file(FIXTURE/'requests_with_LOCAL_gold.json')!=proof['request_file_sha256']:
        raise ValueError('Exact original TRAIN20 fixture required')
    items=json.loads((FIXTURE/'requests_with_LOCAL_gold.json').read_text())
    pred=json.loads(PREDICTIONS.read_text())
    if pred['completed_predictions']!=20 or len(items)!=20:
        raise ValueError('All20 original TRAIN predictions required')
    for p,h in proof['bound_files'].items():
        if sha256_file(p)!=h:raise ValueError('Original TRAIN source changed')
    receipts=[];pages=[]
    for start in range(0,20,4):
        sheet=Image.new('RGB',(1152,4*448),(242,242,242));draw=ImageDraw.Draw(sheet)
        for row,(item,prediction) in enumerate(zip(items[start:start+4],pred['rows'][start:start+4],strict=True)):
            if item['number']!=prediction['number'] or item['source_file']!=prediction['source_file']:
                raise ValueError('Prediction/image source mismatch')
            d=payload(item['source_file']);fi=item['source_frame'];raw=validate_scene(item['payload']['scene64'])
            np.testing.assert_allclose(raw[:7],np.r_[d['joint_positions'][fi],d['gripper_closure'][fi]],rtol=0,atol=1e-6)
            np.testing.assert_array_equal(d['canonical_actions'][fi],item['canonical_gold_LOCAL_ONLY'])
            if item['payload']['instruction']!=str(d['prompt'].item()):raise ValueError('Instruction changed')
            front,wrist=d['front_rgb'][fi],d['wrist_rgb'][fi]
            fm,wm=d['front_pick_mask'][fi],d['wrist_pick_mask'][fi]
            crop,receipt=target_view(front,wrist,fm,wm)
            prefix=f"{item['number']:02d}_{item['task']}_{item['stage']}"
            for col,(name,array) in enumerate((('front',front),('wrist',wrist),('proposed_GT_crop',crop))):
                Image.fromarray(array).save(out/(prefix+'_'+name+'.png'))
                sheet.paste(Image.fromarray(array),(col*384,row*448+54))
            text=f"#{item['number']:02d} {item['task']} | {item['target']} | {item['stage']} | XYZ error {prediction['metrics']['XYZ_error_mm']:.2f}mm"
            draw.text((6,row*448+4),text,fill=(0,0,0))
            draw.text((6,row*448+20),f"raw front | raw wrist | proposed native third view ({receipt['source_camera']}); frame {fi}",fill=(0,0,0))
            draw.text((6,row*448+36),f"target-pad XYZ={np.round(raw[11:14],4)}; actual held={raw[17]:.0f}; source pixels front/wrist={int(fm.sum())}/{int(wm.sum())}",fill=(0,0,0))
            receipts.append({**{k:item[k] for k in ('number','task','target','stage','source_file','source_frame')},
                'instruction':item['payload']['instruction'],'target_pad_relative_m':raw[11:14].tolist(),
                'actual_correct_bilateral_hold':bool(raw[17]),'front_visible_target_pixels':int(fm.sum()),
                'wrist_visible_target_pixels':int(wm.sum()),'existing_frozen_target_view_receipt':receipt,
                'actual_native_first_action':prediction['actual_neural_action'],
                'gold_LOCAL_only':item['canonical_gold_LOCAL_ONLY'],'metrics':prediction['metrics']})
        page=out/f'actual_TRAIN20_images_page{start//4+1}.png';sheet.save(page);pages.append(str(page))
    atomic_json(out/'audit.json',{'passed_exact_source_export':True,'observations':20,'pages':pages,'rows':receipts,
        'GPU_models_loaded':0,'simulator_controls':0,'finetuning_updates':0,'800_outcomes_read':False,
        'proposed_crop_is_existing_implementation_NOT_new_observation_recipe':True,
        'visual_clarity_NOT_certified_by_pixel_count':True,
        'current_model_receives_only_front_wrist_and_numeric67_NOT_crop':True,
        'all20_exported_not_only_successful_predictions':True,
        'bound_files':{str(p):sha256_file(p) for p in (Path(__file__).resolve(),PREDICTIONS,FIXTURE/'audit.json',
            FIXTURE/'requests_with_LOCAL_gold.json',ROOT/'VLA/smolvla/target_crop.py')}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
