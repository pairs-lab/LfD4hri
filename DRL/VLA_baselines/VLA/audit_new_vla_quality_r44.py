"""Read-only coverage, causal feedback and real native-image audit.

This does not alter training, collect labels, open reserved test seeds or
assert that a dataset/loss guarantees a task success rate. Run on CPU while
the hash-bound native training pipeline continues unchanged.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
N = ROOT / 'VLA/audits/new_vlas_r43_20261016'
DATA = N / 'native_complete_data_v4_native_contracts'
SOURCE = ROOT / 'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013'
CORRECTIONS = ROOT / 'VLA/audits/scene67_deadline_r31_20261015/native_corrective_full38_r32/fresh38'
TASKS = ('touch', 'pick', 'move', 'pick_place')


def read(path):
    return json.loads(Path(path).read_text())


def physical_signature(task):
    # Ignore instruction, requested identity, source role and named empty goal.
    # A physical goal object IS included; all coordinates are initial metadata.
    from VLA.common.spatial import FIXED_DESTINATION
    objects = {obj['name']: (obj['position'], obj['fixed']) for obj in task['scene_objects']}
    objects[task['pick_name']] = (task['pick_position'], False)
    if task.get('place_name') and task['place_name'] not in FIXED_DESTINATION:
        objects[task['place_name']] = (task['place_position'], task['place_name'].endswith(' bowl'))
    content = [(name, [round(float(x), 6) for x in position], bool(fixed))
               for name, (position, fixed) in sorted(objects.items())]
    return hashlib.sha256(json.dumps(content, separators=(',', ':')).encode()).hexdigest()


def summary(values):
    x = np.asarray(values)
    return {'count': len(x), 'min': float(x.min()), 'p01': float(np.quantile(x, .01)),
            'median': float(np.median(x)), 'p99': float(np.quantile(x, .99)), 'max': float(x.max())}


def inventory(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows = read(DATA / 'index.json')
    preferred_task = {name: TASKS[i % len(TASKS)] for i, name in
                      enumerate(sorted({r['pick_name'] for r in rows}))}
    train = read(SOURCE / 'qualified_train/index.json')
    validation = read(SOURCE / 'validation_candidate/index.json')
    corrections = read(CORRECTIONS / 'index.json')
    by_path = {str((SOURCE / 'qualified_train' / r['file']).resolve()): r for r in train}
    by_path.update({str((CORRECTIONS / r['file']).resolve()): r for r in corrections})
    train_groups = {r['scene_group'] for r in train}
    val_groups = {r['scene_group'] for r in validation}
    train_geometry = {physical_signature(r['task']) for r in train}
    val_geometry = {physical_signature(r['task']) for r in validation}
    corrective_geometry = {physical_signature(r['task']) for r in corrections}
    if any(str(SOURCE / 'qualified_train' / r['source_file']) not in by_path for r in corrections):
        raise ValueError('Correction parent must resolve to TRAIN, not held-out data')
    from VLA.common.tasks import select_tasks, randomize_rl_scene
    from VLA.run_cross_vla120 import SEEDS
    development = []
    for seed in SEEDS[:10]:
        rng = np.random.default_rng(seed)
        development.extend(asdict(randomize_rl_scene(task, rng)) for task in select_tasks('all'))
    development_geometry = {physical_signature(task) for task in development}
    leakage = {'train_scene_groups': len(train_groups), 'validation_scene_groups': len(val_groups),
        'group_overlap': len(train_groups & val_groups), 'train_physical_scenes': len(train_geometry),
        'validation_physical_scenes': len(val_geometry),
        'physical_geometry_overlap_rounded_1um': len(train_geometry & val_geometry),
        'corrective_physical_scenes': len(corrective_geometry),
        'corrective_scenes_not_from_TRAIN': len(corrective_geometry - train_geometry),
        'corrective_validation_geometry_overlap': len(corrective_geometry & val_geometry),
        'development40_train_geometry_overlap': len(development_geometry & train_geometry),
        'development40_corrective_geometry_overlap': len(development_geometry & corrective_geometry),
        'reserved_final_scenes_not_opened': True}
    coverage = defaultdict(lambda: {'episodes': 0, 'frames': 0, 'targets': Counter(), 'destinations': Counter(),
                                    'target_destination_pairs': Counter(), 'physical_scenes': set(),
                                    'xy_grid_5x5': Counter(), 'kind': Counter()})
    feedback = defaultdict(Counter)
    numerical_ranges = []
    visible = defaultdict(list)
    worst = []
    anchors = []
    chosen_objects = set()
    correction_counts = Counter()
    prompt_errors = []
    state_alignment_errors = []
    for ei, row in enumerate(rows):
        meta = by_path[row['source']]
        task = meta['task']
        category = coverage[row['task']]
        category['episodes'] += 1
        category['frames'] += row['frames']
        category['targets'][row['pick_name']] += 1
        category['kind'][row['source_kind']] += 1
        category['physical_scenes'].add(physical_signature(task))
        if task.get('place_name'):
            category['destinations'][task['place_name']] += 1
            category['target_destination_pairs'][row['pick_name'] + ' -> ' + task['place_name']] += 1
        xy = np.asarray(task['pick_position'][:2])
        grid = np.clip(((xy - [-.25, -.75]) / .1).astype(int), 0, 4)
        category['xy_grid_5x5'][f'{grid[0]},{grid[1]}'] += 1
        with np.load(DATA / row['cache'], allow_pickle=False) as archive:
            raw, action = archive['scene64'], archive['canonical7']
        if raw.shape != (row['frames'], 64) or action.shape != (row['frames'], 7):
            raise ValueError('Frame correspondence mismatch')
        numerical_ranges.append(np.stack((raw.min(0), raw.max(0))))
        held, attached = raw[:, 17] > .5, raw[:, 31] > .5
        empty_closed = (raw[:, 10] > .5) & ~attached
        opening, closing = action[:, 6] < -.5, action[:, 6] > .5
        acquire = (~held[:-1]) & held[1:]
        group = feedback[row['task'] + '/' + row['source_kind']]
        group.update({'frames': len(raw), 'correct_bilateral_hold': int(held.sum()),
            'wrong_attached': int((raw[:, 30] > .5).sum()), 'any_attached': int(attached.sum()),
            'previous_closed_empty': int(empty_closed.sum()),
            'empty_closed_reopen_labels': int((empty_closed & opening).sum()),
            'empty_closed_continue_close_labels': int((empty_closed & closing).sum()),
            'verified_hold_acquisitions': int(acquire.sum()),
            'upward_command_while_no_attachment': int(((action[:, 2] > .003) & ~attached).sum()),
            'closing_with_far_horizontal_target_5cm': int((closing & (np.linalg.norm(raw[:, 11:13], axis=1) > .05)).sum())})
        if row['task'] != 'touch':
            group['episodes_with_reopen_after_empty_close'] += int(np.any(empty_closed & opening))
        with np.load(row['source'], allow_pickle=False) as archive:
            if str(archive['prompt']) != row['instruction'] or task['prompt'] != row['instruction']:
                prompt_errors.append(ei)
            if not np.array_equal(archive['canonical_actions'], action):
                state_alignment_errors.append({'episode': ei, 'kind': 'canonical_cache_vs_source'})
            if not np.array_equal(archive['joint_positions'], raw[:, :6]):
                state_alignment_errors.append({'episode': ei, 'kind': 'measured_joints_vs_causal_state'})
            np.testing.assert_allclose(archive['gripper_closure'], raw[:, 6], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(archive['ee_rotation_6d'], raw[:, 18:24])
            next_grasp = archive['next_grasped_object'].astype(bool)
            next_verified = archive['next_verified_grasp'].astype(bool)
            group['upward_closed_empty_AND_next_empty'] += int(((action[:, 2] > .003) & closing & ~attached & ~next_grasp).sum())
            group['next_grasped_frames'] += int(next_grasp.sum())
            group['next_verified_frames'] += int(next_verified.sum())
            if len(raw) > 1:
                np.testing.assert_allclose(archive['next_observed_joint_positions'][:-1], raw[1:, :6], atol=1e-7, rtol=0)
                np.testing.assert_allclose(archive['next_requested_gripper_closure'][:-1], raw[1:, 10], atol=1e-7, rtol=0)
            pixel_counts = []
            for role in ('front', 'wrist'):
                # Exact PIL NEAREST geometry, independently checked on frame0.
                mask = archive[role + '_pick_mask']
                # Derive PIL's actual integer lookup (including its boundary
                # rounding), rather than assuming an algebraic float formula.
                indices = np.asarray(Image.fromarray(np.arange(mask.shape[-1], dtype=np.int32)[None])
                    .resize((224, 1), Image.Resampling.NEAREST))[0]
                reduced = mask[:, indices][:, :, indices]
                reference = np.asarray(Image.fromarray(mask[0]).resize((224, 224), Image.Resampling.NEAREST))
                np.testing.assert_array_equal(reference, reduced[0])
                count = reduced.sum((1, 2)).astype(int)
                pixel_counts.append(count)
                visible[row['task'] + '/' + role].extend(count.tolist())
                visible['all/' + role].extend(count.tolist())
                visible['reset/' + role].append(int(count[0]))
            best = np.maximum(*pixel_counts)
            visible['all/best_camera'].extend(best.tolist())
            visible['reset/best_camera'].append(int(best[0]))
            group['target_both_views_under25_pixels_at224'] += int((best < 25).sum())
            group['held_target_both_views_under25_pixels_at224'] += int(((best < 25) & held).sum())
            worst.extend((int(best[fi]), ei, int(fi)) for fi in np.argsort(best)[:1])
        if (row['source_kind'] == 'complete_original' and row['pick_name'] not in chosen_objects
                and row['task'] == preferred_task[row['pick_name']]):
            # The first episode for each target is reproducible, not a best-looking selection.
            chosen_objects.add(row['pick_name'])
            close_ids = np.flatnonzero(closing)
            near = int(close_ids[0]) if len(close_ids) else int(np.linalg.norm(raw[:, 11:14], axis=1).argmin())
            held_ids = np.flatnonzero(held)
            carry = int(held_ids[len(held_ids)//2]) if len(held_ids) else len(raw)-1
            anchors.extend({'episode': ei, 'frame': fi, 'phase': phase, 'object': row['pick_name'],
                'task': row['task'], 'kind': row['source_kind']} for phase, fi in
                (('initial', 0), ('first_close_or_nearest', near), ('held_or_terminal', carry)))
        if row['source_kind'] == 'fresh_corrective' and correction_counts[row['task']] < 2:
            correction_counts[row['task']] += 1
            anchors.append({'episode': ei, 'frame': 0, 'phase': 'corrective_handoff',
                'object': row['pick_name'], 'task': row['task'], 'kind': row['source_kind']})
        if (ei+1) % 100 == 0:
            atomic_json(out / 'status.json', {'status': 'auditing', 'episodes': ei+1,
                'total_episodes': len(rows), 'training_unmodified': True})
            print(json.dumps({'episodes_audited': ei+1}), flush=True)
    for value, ei, fi in sorted(worst)[:8]:
        anchors.append({'episode': ei, 'frame': fi, 'phase': f'worst_best_view_{value}px',
            'object': rows[ei]['pick_name'], 'task': rows[ei]['task'], 'kind': rows[ei]['source_kind']})
    for task, category in coverage.items():
        category['physical_scenes'] = len(category['physical_scenes'])
        category['observed_target_destination_pairs'] = len(category['target_destination_pairs'])
    ranges = np.stack(numerical_ranges)
    visibility = {key: {**summary(values), 'under25_pixels': int((np.asarray(values) < 25).sum()),
        'under49_pixels': int((np.asarray(values) < 49).sum())} for key, values in visible.items()}
    result = {'scope': 'full_current_TRAIN_quality_and_leakage_NOT_success_rate',
        'episodes': len(rows), 'frames': sum(r['frames'] for r in rows),
        'source_images': 2*sum(r['frames'] for r in rows), 'coverage': dict(coverage),
        'leakage': leakage, 'feedback': dict(feedback), 'target_visibility_native224': visibility,
        'causal_raw64_min': ranges[:, 0].min(0).tolist(), 'causal_raw64_max': ranges[:, 1].max(0).tolist(),
        'prompt_mismatch_episodes': prompt_errors, 'alignment_errors': state_alignment_errors,
        'mask_geometry': 'exact224_PIL_NEAREST_mask_not_RGB_visibility_judgment',
        'data_audit_sha256': sha256_file(DATA / 'audit.json'),
        'bound_files': {str(p): sha256_file(p) for p in (Path(__file__), DATA / 'index.json',
            SOURCE / 'qualified_train/index.json', SOURCE / 'validation_candidate/index.json',
            CORRECTIONS / 'index.json')}, 'visual_inspection_still_required': True,
        'adequacy_or_high_success_NOT_proven': True, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(out / 'audit.json', result)
    atomic_json(out / 'anchors.json', anchors)
    atomic_json(out / 'status.json', {'status': 'completed_numeric_and_masks',
        'episodes': len(rows), 'frames': result['frames'], 'visual_review_pending': True})
    print(json.dumps({'leakage': leakage, 'feedback': dict(feedback), 'visibility': visibility['all/best_camera']}))


def exposure(args):
    rows = read(DATA / 'index.json')
    root = ROOT / 'VLA' / args.family / 'privileged_r43_native_v3_20261016/test1/train'
    status = read(root / 'status.json')
    last = status['actual_optimizer_updates']
    counts, by_target, by_kind = Counter(), Counter(), Counter()
    anchors, episodes, horizon_labels = set(), set(), set()
    micros = 0
    horizon = 30 if args.family == 'xvla' else 8
    for line in (root / 'exposure.jsonl').read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # Live append tail is not a completed optimizer update.
        if record['optimizer_update'] > last:
            continue
        ei, fi = record['episode'], record['frame']
        row = rows[ei]
        anchors.add((ei, fi)); episodes.add(ei)
        horizon_labels.update((ei, min(frame, row['frames']-1)) for frame in range(fi, fi+horizon))
        counts[row['task']] += 1; by_target[row['task'] + '/' + row['pick_name']] += 1
        by_kind[row['source_kind']] += 1
        micros += 1
    result = {'family': args.family, 'train_status_snapshot': status,
        'completed_optimizer_updates': last, 'logged_micro_samples_since_two_step_gate': micros,
        'unique_image_state_anchors': len(anchors), 'unique_episode_sources': len(episodes),
        'unique_anchors_fraction_of_available_TRAIN': len(anchors)/sum(r['frames'] for r in rows),
        'unique_future_label_frames_covered_by_chunks': len(horizon_labels),
        'sample_counts_by_action': dict(counts), 'sample_counts_by_target': dict(by_target),
        'sample_counts_by_kind': dict(by_kind), 'gate8_anchors_not_in_this_log': True,
        'low_loss_does_not_establish_generalization_or_SR': True}
    atomic_json(args.output / (args.family + '_exposure_snapshot.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sample_counts_by_target', 'train_status_snapshot')}))


def images(args):
    import torch
    torch.set_num_threads(2)
    rows, anchors = read(DATA / 'index.json'), read(args.output / 'anchors.json')
    out = args.output / (args.family + '_native224')
    out.mkdir(parents=True, exist_ok=False)
    if args.family == 'xvla':
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        holder = SimpleNamespace(config=XVLAConfig.from_pretrained(ROOT / 'VLA/checkpoints/xvla-base'))
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        def preprocess(rgb):
            batch = {'observation.images.'+key: ((torch.from_numpy(rgb[role]).permute(2, 0, 1).float()/255-mean)/std)[None]
                     for key, role in (('image', 'front'), ('image2', 'wrist'))}
            actual, valid = XVLAPolicy._prepare_images(holder, batch)
            assert valid.tolist() == [[True, True, False]]
            return [actual[0, view]*std+mean for view in range(2)]
    else:
        from VLA.new_vlas.adapter_loading import import_native
        _, _, Processor, _, _, _ = import_native()
        processor = Processor.from_pretrained(ROOT / 'VLA/checkpoints/vla-adapter-spatial-pro')
        def preprocess(rgb):
            result = []
            for role in ('front', 'wrist'):
                actual = processor(Image.fromarray(rgb[role]), return_tensors='pt')['pixel_values'][0]
                assert actual.shape == (6, 224, 224)
                norm = processor.tvf_normalize_params[0]
                result.append(actual[:3]*torch.tensor(norm['std'])[:, None, None]+torch.tensor(norm['mean'])[:, None, None])
            return result
    # Three phases per object across columns, five objects per sheet. Additional
    # corrective/worst-view anchors are a separate sheet, not silently omitted.
    original = [a for a in anchors if a['phase'] in ('initial', 'first_close_or_nearest', 'held_or_terminal')]
    extras = [a for a in anchors if a not in original]
    pages = [(original[i:i+15], 3) for i in range(0, len(original), 15)]
    pages.extend((extras[i:i+8], 1) for i in range(0, len(extras), 8))
    records = []
    for page, (page_anchors, cols) in enumerate(pages):
        sheet = Image.new('RGB', (448*cols, 256*((len(page_anchors)+cols-1)//cols)), '#eeeeee')
        drawing = ImageDraw.Draw(sheet)
        for index, anchor in enumerate(page_anchors):
            row = rows[anchor['episode']]
            with np.load(row['source'], allow_pickle=False) as archive:
                rgb = {role: archive[role+'_rgb'][anchor['frame']] for role in ('front', 'wrist')}
            pixels = preprocess(rgb)
            x, y = (index % cols)*448, (index//cols)*256
            drawing.text((x+3, y+2), row['instruction'][:65], fill='#111111')
            drawing.text((x+3, y+15), anchor['phase']+f" t={anchor['frame']} / front | wrist", fill='#111111')
            names = []
            for view, tensor in enumerate(pixels):
                result = np.uint8(np.rint(np.clip(tensor.permute(1, 2, 0).numpy(), 0, 1)*255))
                assert result.shape == (224, 224, 3)
                im = Image.fromarray(result)
                name = f"e{anchor['episode']:04d}_t{anchor['frame']:03d}_{view}.png"
                im.save(out / name); names.append(name)
                sheet.paste(im, (x+view*224, y+32))
            records.append({**anchor, 'source': row['source'], 'images': names})
        sheet.save(out / f'contact_sheet_{page:02d}.png')
    atomic_json(out / 'audit.json', {'actual_native224_preprocessing': True, 'family': args.family,
        'objects': sorted({a['object'] for a in original}), 'anchors': len(records), 'rows': records,
        'pages': len(pages), 'GPU_used': False, 'visual_review_still_required': True,
        'contact_sheets_sha256': {str(p): sha256_file(p) for p in out.glob('contact_sheet_*.png')}})
    print(json.dumps({'family': args.family, 'anchors': len(records), 'pages': len(pages)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inventory', 'exposure', 'images'))
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), default='xvla')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args)
