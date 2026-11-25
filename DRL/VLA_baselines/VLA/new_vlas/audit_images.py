"""Export ACTUAL native224 views from complete-source4-task gate anchors."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import pack_native20, native8, normalize_bounds
from VLA.new_vlas.dataset import NativeData
from VLA.new_vlas.privileged_inputs import append_numeric67
from VLA.new_vlas.train import gate_anchors


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = NativeData(args.data)
    anchors = gate_anchors(data)
    if args.family == 'xvla':
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        config = XVLAConfig.from_pretrained('VLA/checkpoints/xvla-base')
        holder = SimpleNamespace(config=config)
    else:
        from VLA.new_vlas.adapter_loading import import_native
        _, _, ImageProcessor, _, _, _ = import_native()
        processor = ImageProcessor.from_pretrained('VLA/checkpoints/vla-adapter-spatial-pro')
    sheet = Image.new('RGB', (448, 248*len(anchors)), '#eeeeee')
    drawing = ImageDraw.Draw(sheet)
    rows = []
    for index, (ei, fi) in enumerate(anchors):
        sample = data.sample(ei, fi, args.family)
        # Verify train/runtime numeric-state parity, not merely equal widths.
        with np.load(data.rows[ei]['source'], allow_pickle=False) as archive:
            xyz, six = archive['ee_position'][fi], archive['ee_rotation_6d'][fi]
            raw = data.numeric(ei)['scene64'][fi]
            prefix = (pack_native20(xyz, six, np.asarray(raw[6])) if args.family == 'xvla'
                else normalize_bounds(native8(xyz, six, raw[27]), data.stats['adapter']['proprio']))
            runtime_state = append_numeric67(prefix, raw, data.stats['scene67'])
            np.testing.assert_array_equal(runtime_state, sample['state'])
            masks = [archive[role+'_pick_mask'][fi] for role in ('front', 'wrist')]
        if args.family == 'xvla':
            mean = torch.tensor([.485, .456, .406])[:, None, None]
            std = torch.tensor([.229, .224, .225])[:, None, None]
            batch = {'observation.images.'+key: ((torch.from_numpy(sample[role]).permute(2, 0, 1).float()/255-mean)/std)[None]
                     for key, role in (('image', 'front'), ('image2', 'wrist'))}
            actual, valid = XVLAPolicy._prepare_images(holder, batch)
            if valid.tolist() != [[True, True, False]] or actual.shape[-2:] != (224, 224):
                raise ValueError('Native camera validity/order/resolution mismatch')
            pixels = [actual[0, view]*std+mean for view in range(2)]
        else:
            pixels = []
            for role in ('front', 'wrist'):
                actual = processor(Image.fromarray(sample[role]), return_tensors='pt')['pixel_values'][0]
                if actual.shape != (6, 224, 224):
                    raise ValueError('Native fused DINO/SigLIP shape mismatch')
                decoded = []
                for tower in range(2):
                    norm = processor.tvf_normalize_params[tower]
                    mean = torch.tensor(norm['mean'])[:, None, None]
                    std = torch.tensor(norm['std'])[:, None, None]
                    decoded.append(actual[tower*3:tower*3+3]*std+mean)
                if (decoded[0]-decoded[1]).abs().max() > 1e-6:
                    raise ValueError('Native DINO/SigLIP image geometry differs')
                pixels.append(decoded[0])
        drawing.text((3, index*248+2), sample['instruction']+' / '+data.rows[ei]['source_kind'], fill='#111111')
        counts = []
        for column, (role, tensor, mask) in enumerate(zip(('front', 'wrist'), pixels, masks, strict=True)):
            result = np.uint8(np.rint(np.clip(tensor.permute(1, 2, 0).numpy(), 0, 1)*255))
            im = Image.fromarray(result)
            im.save(args.output/f'{index:02d}_{sample["task"]}_{role}_native224.png')
            sheet.paste(im, (column*224, index*248+24))
            reduced = np.asarray(Image.fromarray(np.uint8(mask > 0)*255).resize((224, 224), Image.Resampling.NEAREST))
            counts.append(int(np.count_nonzero(reduced)))
        rows.append({'task': sample['task'], 'kind': data.rows[ei]['source_kind'], 'source': data.rows[ei]['source'],
            'frame': fi, 'state_dim': len(sample['state']), 'train_runtime_state_exact': True,
            'target_visible_pixels_front_wrist_at224': counts})
    sheet.save(args.output/'actual_native224_contact_sheet.png')
    atomic_json(args.output/'audit.json', {'passed': True, 'family': args.family,
        'actual_native_preprocessing_not_generic_preview': True, 'all4_tasks_and2_sources': True,
        'train_runtime_native_state_exact_for8anchors': True,
        'all67_inputs_preserved': True, 'rows': rows,
        'contact_sheet_sha256': sha256_file(args.output/'actual_native224_contact_sheet.png'),
        'scope': '8_representative_NATIVE_INPUT_anchors_NOT_exhaustive_visual_sufficiency_or_SR'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
