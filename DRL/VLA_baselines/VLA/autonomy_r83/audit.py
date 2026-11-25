"""Read-only TRAIN audit and DEV-only failure evidence. No final outcomes tune models."""
import argparse
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.autonomy_r83.common import OUT, R80, ROOT, TASKS, DEV, CONF, FINAL, read, code, metrics


def main():
    from VLA.flower_rotation_r76.sampler import Data, RECIPE
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'CPU_AUDIT.json'
    if target.exists():
        raise FileExistsError('Preserve existing audit')
    data = Data()
    counts = Counter()
    targets = {task: sorted(data.groups[task]['complete_original']) for task in TASKS}
    if any(len(values) != 19 for values in targets.values()):
        raise ValueError('Missing TRAIN target coverage')
    for i, row in enumerate(data.rows):
        n = data.numeric(i)
        a, raw = n['canonical7'], n['scene64']
        if a.shape != (row['frames'], 7) or raw.shape != (row['frames'], 64):
            raise ValueError('Source alignment/shape mismatch')
        if not np.isfinite(a).all() or not np.isfinite(raw).all():
            raise ValueError('Nonfinite TRAIN labels or observations')
        counts['episodes/' + row['task']] += 1
        counts['frames/' + row['task']] += len(a)
    sampled = Counter()
    rng = np.random.default_rng(202683)
    for _ in range(8000):
        ei, fi = data.draw(rng)
        row = data.rows[ei]
        sampled[row['task'] + '/' + row['pick_name']] += 1
    if len(sampled) != 76 or min(sampled.values()) < 50:
        raise ValueError('Sampler drops/underexposes a task/target stratum')
    images = {}
    for task in TASKS:
        ei = next(i for i, r in enumerate(data.rows) if r['task'] == task and r['source_kind'] == 'complete_original')
        n = data.numeric(ei)
        chosen = sorted(set([0, len(n['scene64']) // 3, 2 * len(n['scene64']) // 3, len(n['scene64']) - 1]))
        canvas = Image.new('RGB', (768, len(chosen) * 280), 'white')
        draw = ImageDraw.Draw(canvas)
        for ri, fi in enumerate(chosen):
            sample = data.at(ei, fi)
            for col, role in enumerate(('front', 'wrist', 'map')):
                image = sample[role]
                if image.shape != (384, 384, 3) or image.dtype != np.uint8 or image.std() < 2:
                    raise ValueError('Empty/wrong RGB source')
                canvas.paste(Image.fromarray(image).resize((256, 256)), (col * 256, ri * 280 + 24))
                draw.text((col * 256 + 2, ri * 280 + 3), f'{task}/{role}, frame{fi}', fill='black')
        path = OUT / (task + '_INPUT_REVIEW.png')
        canvas.save(path)
        images[str(path)] = sha256_file(path)
    # All cause labels below are trace heuristics, not proofs of cognitive mechanisms.
    failure = {}
    source = R80 / 'development40_fit4096/progress.json'
    if source.exists():
        p = read(source)
        categories = Counter()
        for row in p['rows']:
            trace = read(row['trajectory'])[row['task']]
            held = any(s['after'].get('correct_object_held', False) for s in trace)
            contact = bool(row.get('events', {}).get('contact'))
            categories['safety_abort' if row.get('safety_violation') else
                       'held_but_not_completed' if held else
                       'contact_without_hold' if contact else 'approach_without_verified_contact'] += 1
        failure = {'source': str(source), 'source_sha256': sha256_file(source),
                   'heuristic_NOT_causal_diagnosis': dict(categories), 'per_action': metrics(p['rows'])}
    # Geometry only, not policy outcomes; qualify all future splits against TRAIN.
    from VLA.run_feedback_refinement_r52 import qualify_final_seeds
    geometry = {}
    sets = []
    for role, seeds in (('development_selection', DEV), ('development_confirmation', CONF),
                        ('fresh_secondary_final800', FINAL)):
        result = qualify_final_seeds(seeds)
        current = {r['physical_scene_sha256'] for r in result['cases']}
        if any(current & previous for previous in sets):
            raise ValueError('New development/confirmation/final physical geometry overlaps')
        sets.append(current)
        geometry[role] = result
    from VLA.audit_training_readiness import physical_scene_key
    train_keys = set()
    sources = {}
    for row in data.rows:
        if row.get('scene_group'):
            train_keys.add(row['scene_group'])
            continue
        source_index = __import__('pathlib').Path(row['source']).parent / 'index.json'
        if source_index not in sources:
            sources[source_index] = read(source_index)
        matches = [r for r in sources[source_index] if r.get('file', r.get('filename')) ==
                   __import__('pathlib').Path(row['source']).name]
        if len(matches) != 1:
            raise ValueError('Cannot recover TRAIN physical scene')
        train_keys.add(physical_scene_key(matches[0]['task']))
    if any(train_keys & values for values in sets):
        raise ValueError('Future evaluation geometry overlaps current TRAIN')
    atomic_json(OUT / 'GEOMETRY_EXCLUSION.json', {'passed': True, 'cohorts': geometry,
        'TRAIN_unique_geometries': len(train_keys), 'all_TRAIN_and_inter_split_overlaps': 0,
        'only_geometry_read_no_policy_outcomes_for_scene_selection': True})
    atomic_json(target, {'passed': True, 'TRAIN_episodes': len(data.rows), 'TRAIN_frames': len(data),
        'counts': dict(counts), 'targets': targets, 'sampler': RECIPE,
        'actual_8000_draw_counts': dict(sampled), 'image_sha256': images,
        'images_are_source_observations_NOT_future_native_GPU_gate': True,
        'DEV_only_failure_evidence': failure, 'code': code(),
        'datasets_NOT_recollected_or_duplicated': True,
        'future_GPU_optimizer_reload_and_behavioral_gates_still_required': True})


if __name__ == '__main__':
    main()
