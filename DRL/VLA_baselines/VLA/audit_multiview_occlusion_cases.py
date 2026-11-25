"""Check both real native views at ALL worst OpenVLA occlusion cases.

Selected diagnostic only, not full-source observability or neural recognition.
Simulator masks are used solely to measure geometry and never sent to a policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from VLA.audit_native_camera_tensors import transformer
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import verify_source_payload_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front-audit', type=Path, required=True)
    parser.add_argument('--model', choices=('pi0', 'smolvla', 'rdt'), required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--processor')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.front_audit.read_text())
    source = Path(audit['source'])
    index = json.loads((source / 'index.json').read_text())
    if (sha256_file(source / 'index.json') != audit['source_index_sha256'] or
        sha256_file(source / 'dataset_manifest.json') != audit['source_manifest_sha256']):
        raise ValueError('Source changed from the actual all-frame camera audit')
    verify_source_payload_fingerprint(audit['source_payload_fingerprint'], source, index)
    cases = [r for r in audit['worst_frames'] if r['category'] == 'all']
    transform, method = transformer(args)
    protected = [args.front_audit, source / 'index.json', source / 'dataset_manifest.json',
                 Path(__file__), Path(__file__).with_name('audit_native_camera_tensors.py')]
    frozen = {str(p.resolve()): sha256_file(p) for p in protected}
    args.output.mkdir(parents=True)
    rows, panels = [], []
    for case in cases:
        key = 'pick' if case['role'] == 'pick' else 'place'
        row = {k: case[k] for k in ('file', 'task', 'frame', 'role', 'target_name')}
        row['views'] = {}
        panel = Image.new('RGB', (448, 248), 'black')
        ImageDraw.Draw(panel).text((4, 4), f"{case['task']} {case['role']} t={case['frame']} {case['target_name']}", fill='white')
        with np.load(source / case['file'], allow_pickle=False) as episode:
            for column, view in enumerate(('front', 'wrist')):
                image = episode[view + '_rgb'][case['frame']]
                mask = episode[view + '_' + key + '_mask'][case['frame']]
                outputs = transform(image, mask)
                if len(outputs) != 1:
                    raise ValueError('Expected one native image geometry per real camera')
                backbone, native, resized_mask = outputs[0]
                pixels = int(resized_mask.sum())
                row['views'][view] = {'method': backbone, 'native_shape': list(native.shape),
                    'target_pixels_native': pixels,
                    'target_area_equivalent_224': pixels * 224**2 / (native.shape[0] * native.shape[1]),
                    'completely_hidden': pixels == 0}
                panel.paste(Image.fromarray(native).resize((224, 224)), (224 * column, 24))
        rows.append(row)
        panels.append(panel)
    sheet = Image.new('RGB', (896, 248 * ((len(panels) + 1) // 2)), 'black')
    for number, panel in enumerate(panels):
        sheet.paste(panel, ((number % 2) * 448, (number // 2) * 248))
    sheet.save(args.output / 'actual_native_front_wrist_cases.png')
    verify_source_payload_fingerprint(audit['source_payload_fingerprint'], source, index)
    if any(sha256_file(Path(p)) != h for p, h in frozen.items()):
        raise RuntimeError('Diagnostic inputs changed')
    (args.output / 'audit.json').write_text(json.dumps({
        'scope': 'ALL worst front occlusion cases selected from supplied all-frame audit_NOT_full_camera_qualification',
        'model': args.model, 'method': method, 'frozen_files': frozen, 'source': str(source),
        'neural_weights_loaded': False, 'training_authorized': False,
        'segmentation_given_to_policy': False, 'rows': rows,
        'all_views_completely_hidden_cases': sum(all(v['completely_hidden'] for v in r['views'].values()) for r in rows)
    }, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'cases': len(rows),
        'views_area_equivalent_224': [[round(v['target_area_equivalent_224'], 1) for v in r['views'].values()] for r in rows]}))


if __name__ == '__main__':
    main()
