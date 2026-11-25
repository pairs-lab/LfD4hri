"""Actual X-VLA224 validation images, not a resized generic preview."""
import argparse
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from VLA.audit_new_vla_quality_r44 import ROOT, read
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    holder = SimpleNamespace(config=XVLAConfig.from_pretrained(ROOT/'VLA/checkpoints/xvla-base'))
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    items = read(args.fixture/'requests_with_LOCAL_gold.json')
    files = []
    for page in range((len(items)+9)//10):
        subset = items[page*10:(page+1)*10]
        sheet = Image.new('RGB', (448, 256*len(subset)), '#eeeeee')
        drawing = ImageDraw.Draw(sheet)
        for index, item in enumerate(subset):
            batch = {}
            for key, role in (('image', 'front'), ('image2', 'wrist')):
                rgb = np.array(Image.open(BytesIO(base64.b64decode(item['payload'][role+'_png']))))
                batch['observation.images.'+key] = ((torch.from_numpy(rgb).permute(2, 0, 1).float()/255-mean)/std)[None]
            actual, valid = XVLAPolicy._prepare_images(holder, batch)
            if valid.tolist() != [[True, True, False]]: raise ValueError('Native camera ordering changed')
            y = index*256
            drawing.text((3, y+2), item['payload']['instruction'], fill='#111111')
            drawing.text((3, y+15), f"VALIDATION {item['stage']} t={item['frame']}", fill='#111111')
            for view in range(2):
                decoded = actual[0, view]*std+mean
                pixels = np.uint8(np.rint(np.clip(decoded.permute(1, 2, 0).numpy(), 0, 1)*255))
                sheet.paste(Image.fromarray(pixels), (224*view, y+32))
        path = args.output/f'validation_native224_{page:02d}.png'
        sheet.save(path); files.append(path)
    atomic_json(args.output/'audit.json', {'split': 'validation_NOT_TRAIN', 'actual_native_preprocessing': True,
        'anchors': len(items), 'GPU_models_loaded': False, 'visual_review_pending': True,
        'images_sha256': {str(p): sha256_file(p) for p in files}})
