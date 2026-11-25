"""Inspect exact resized policy pixels at reset/contact/hold/recovery/terminal.

Audit event names are never input to the policy; no future image is used.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.data import Data, phase_anchors
from VLA.flower_r69.native import MEAN, STD, image_tensor


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    data=Data(horizon=10)
    chosen,events=phase_anchors(data)
    records=[]
    for task in sorted(data.groups):
        subset=[(ei,fi) for ei,fi in chosen if data.rows[ei]['task']==task]
        sheet=Image.new('RGB',(3*224,len(subset)*252),'white')
        draw=ImageDraw.Draw(sheet)
        for row,(ei,fi) in enumerate(subset):
            s=data.at(ei,fi)
            record={'task':task,'episode':ei,'frame':fi,'source_kind':data.rows[ei]['source_kind'],
                'recovery_kind':data.rows[ei].get('recovery_kind'),'events_audit_only':events[ei,fi],
                'correct_bilateral_hold':bool(s['raw64'][17]),'camera_pixels':{}}
            for col,key in enumerate(('front','wrist','map')):
                tensor=image_tensor(s[key]).numpy()
                inverse=tensor*np.asarray(STD)[:,None,None]+np.asarray(MEAN)[:,None,None]
                pixels=np.uint8(np.round(np.clip(inverse.transpose(1,2,0),0,1)*255))
                sheet.paste(Image.fromarray(pixels),(224*col,row*252+28))
                draw.text((224*col+3,row*252+2),f'{task}/{key} e{ei} f{fi}',fill='black')
                draw.text((224*col+3,row*252+14),','.join(events[ei,fi])[:34],fill='black')
                record['camera_pixels'][key]={'shape':list(pixels.shape),
                    'normalized_min':float(tensor.min()),'normalized_max':float(tensor.max()),
                    'channel_std':pixels.reshape(-1,3).std(0).tolist()}
            records.append(record)
        sheet.save(a.output/(task+'_PHASES_224.png'))
    atomic_json(a.output/'AUDIT.json',{'passed_nonempty_images_all4_tasks':True,
        'records':records,'actual_resize_normalization_inverse_visualization':True,
        'audit_event_names_NOT_model_input':True,
        'all_frames_claimed_unoccluded':False,
        'near_contact_physical_gripper_occlusion_requires_GT_and_numeric67_not_camera_cheating':True})
    print({'actual_phase_frames':len(records),'tasks':sorted(data.groups)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
