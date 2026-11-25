"""Export actual normalized model images at measured teacher grasp transitions."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.data import Data
from VLA.flower_r69.native import image_tensor, MEAN, STD


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    data = Data(horizon=10); receipts = []
    for target in ('red gelatin_box', 'cyan block', 'yellow block'):
        for ei, row in enumerate(data.rows):
            if row['task'] != 'pick' or row['pick_name'] != target or row['source_kind'] != 'complete_original':
                continue
            n = data.numeric(ei); raw, cmd = n['scene64'], n['canonical7']
            contact = np.flatnonzero(raw[:, 28:30].any(axis=1) & (raw[:, 31] < .5))
            held = np.flatnonzero((raw[:, 31] > .5) & (raw[:, 30] < .5))
            close = np.flatnonzero((cmd[:, 6] > .2) & (raw[:, 10] < .5))
            if not (len(contact) and len(held) and len(close)):
                continue
            events = [('pre_close', max(0, int(close[0])-1)), ('first_close', int(close[0])),
                ('contact_unattached', int(contact[0])), ('correct_attached', int(held[0])),
                ('terminal', len(raw)-1)]
            sheet = Image.new('RGB', (672, len(events)*252), 'white'); draw = ImageDraw.Draw(sheet)
            for j, (name, fi) in enumerate(events):
                sample = data.at(ei, fi)
                for col, key in enumerate(('front', 'wrist', 'map')):
                    t = image_tensor(sample[key]).numpy()
                    pixels = np.uint8(np.rint(np.clip(t*np.asarray(STD)[:, None, None]+
                        np.asarray(MEAN)[:, None, None], 0, 1).transpose(1,2,0)*255))
                    sheet.paste(Image.fromarray(pixels), (224*col, 252*j+28))
                    draw.text((224*col+2, 252*j+3), f'{target} t{fi} {name}/{key}', fill='black')
                receipts.append({'episode': ei, 'target': target, 'frame': fi, 'event': name,
                    'causal_input_pre_action': True, 'image_shape': [3,224,224],
                    'source': row['source'], 'source_sha256': row['source_sha256'],
                    'RGB_byte_SHA256': {k: hashlib.sha256(sample[k].tobytes()).hexdigest() for k in ('front','wrist','map')},
                    'requested_signed_close': float(cmd[fi,6]), 'measured_closure': float(raw[fi,6]),
                    'pad_height_fraction': float(raw[fi,38]), 'attached_correct': bool(raw[fi,31] and not raw[fi,30])})
            sheet.save(a.output / (target.replace(' ','_') + '_TEACHER_INPUTS.png'))
            break
        else:
            raise ValueError('No qualified contact/hold demonstration found for ' + target)
    atomic_json(a.output / 'MODEL_INPUT_RECEIPTS.json', {'frames': len(receipts), 'rows': receipts,
        'uses_actual_native_image_tensor_then_inverse_normalization_for_review': True,
        'selection_TRAIN_only': True, 'model_prediction_OR_robot_SR_claimed': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
