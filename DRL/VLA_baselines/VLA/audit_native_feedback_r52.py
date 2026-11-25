"""CPU audit of the new recovery bank and ACTUAL native image processing.

This does not instantiate a GPU policy or estimate success. Report visibility
loss instead of rejecting a real grasp merely because its wrist view is close.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.fit_feedback_r52 import gate_anchors
from VLA.new_vlas.dataset import NativeData
from VLA.scene64_conditioning import encode_numeric
from VLA.stable_native_r44.conditioning import install, bounded_append, contract


def saturation(raw, stats):
    value = np.asarray([encode_numeric(r) for r in raw])
    standardized = (value-np.asarray(stats['mean']))/(np.asarray(stats['std'])+1e-8)
    return {'frames': len(raw), 'any_slot_over_clip5': int(np.any(abs(standardized) > 5, axis=1).sum()),
            'q4_over_clip5': int((abs(standardized[:, 3]) > 5).sum()),
            'max_abs_q4_standardized': float(abs(standardized[:, 3]).max())}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    install()
    # Full RGB-byte verification also runs in the optimizer admission gate.
    # Here verify ALL new RGB sources plus the eight actually viewed anchors,
    # rather than rehashing ~100GB of unchanged original images for each model.
    data = NativeData(args.data, verify_rgb_hashes=False)
    old = json.loads(Path('VLA/audits/new_vlas_r43_20261016/native_complete_data_v4_native_contracts/stats.json').read_text())
    raw_all, canon_all, rows = [], [], []
    for i, row in enumerate(data.rows):
        if row['source_kind'] != 'pose_corrective':
            continue
        numeric = data.numeric(i)
        if sha256_file(row['source']) != row['source_sha256']:
            raise ValueError('Corrective RGB payload changed')
        raw_all.append(numeric['scene64']); canon_all.append(numeric['canonical7'])
        with np.load(row['source'], allow_pickle=False) as a:
            pixels = {}
            for role in ('front', 'wrist'):
                masks = a[role+'_pick_mask']
                counts, edges = [], []
                for mask in masks:
                    visible = np.asarray(Image.fromarray(np.uint8(mask > 0)*255).resize((224, 224), Image.Resampling.NEAREST)) > 0
                    counts.append(int(visible.sum()))
                    edges.append(bool(visible[0].any() or visible[-1].any() or visible[:, 0].any() or visible[:, -1].any()))
                pixels[role] = {'frames': len(counts), 'zero_target_pixel_frames': int(np.count_nonzero(np.asarray(counts) == 0)),
                    'frames_target_touches_border': int(np.count_nonzero(edges)),
                    'visible_pixel_min_median_max': [int(min(counts)), float(np.median(counts)), int(max(counts))]}
        rows.append({'task': row['task'], 'target': row['pick_name'], 'intervention': row['recovery_kind'],
            'source': row['source'], 'frames': row['frames'], 'target_mask_resized224_visibility_NOT_generic_image_geometry_proof': pixels})
    raw, canonical = np.concatenate(raw_all), np.concatenate(canon_all)
    # Reuse the official model transforms, not a manually resized RGB preview.
    import VLA.new_vlas.audit_images as images
    images.gate_anchors = gate_anchors
    images.append_numeric67 = bounded_append
    for i, _ in gate_anchors(data):
        if sha256_file(data.rows[i]['source']) != data.rows[i]['source_sha256']:
            raise ValueError('Actually viewed gate-anchor RGB changed')
    images.NativeData = lambda root: NativeData(root, verify_rgb_hashes=False)
    images.main(SimpleNamespace(family=args.family, data=args.data, output=args.output/'actual_native224'))
    proof = json.loads((args.output/'actual_native224/audit.json').read_text())
    atomic_json(args.output/'audit.json', {'passed': True, 'family': args.family,
        'scope': 'data_input_and_range_quality_NOT_VLA_success_or_exhaustive_visual_sufficiency',
        'data_audit_sha256': sha256_file(args.data/'audit.json'),
        'audit_implementation_sha256': sha256_file(Path(__file__)),
        'actual_bounded_train_and_runtime_conditioning': contract(),
        'original_episodes_retained': data.proof['all_original_episodes_retained'],
        'RGB_payload_hashes_verified_here': 'all_new_sources_and8_actual_native_anchors; full_source_rehash_required_again_by_optimizer_gate',
        'new_episodes': len(rows), 'new_frames': len(raw),
        'task_counts': dict(Counter(r['task'] for r in rows)),
        'target_identities': sorted({r['target'] for r in rows}),
        'nonzero_roll_pitch_label_frames': int(np.any(abs(canonical[:, 3:5]) > 1e-5, axis=1).sum()),
        'corrective_states_under_old_stats': saturation(raw, old['scene67']),
        'corrective_states_under_NEW_stats': saturation(raw, data.stats['scene67']),
        'normalizer_floor_overrides_inherited_pre_floor_first18_flag': True,
        'native224_image_and_runtime_state_proof': proof,
        'rows': rows, 'camera_occlusion_is_reported_NOT_hidden': True})
    print(json.dumps({'family': args.family, 'new_episodes': len(rows), 'new_frames': len(raw),
        'actual_native224': str(args.output/'actual_native224/actual_native224_contact_sheet.png')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
