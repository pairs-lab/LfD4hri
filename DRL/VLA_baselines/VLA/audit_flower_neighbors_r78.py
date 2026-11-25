"""Pinned, CPU-only candidate inventory and actual Evo-1 pixel-boundary audit.

This does NOT load foundation weights, run a VLM, optimize, or measure robot SR.
It never changes frozen trainers, datasets, checkpoints, or running processes.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VLA/audits/flower_neighbors_r78_20261020'
SOURCES = (
    ('evo1', 'https://github.com/MINT-SJTU/Evo-1.git',
     '5fd14b015013c4fd0aacf5f8f48f868ca9b870a2'),
    ('evo_depth', 'https://github.com/MINT-SJTU/Evo-Depth.git',
     '617e3f901321573001a02ebb0e8e3ff8802122bd'),
)
HUB = (
    ('MINT-SJTU/Evo1_LIBERO', '3ddc6c9cafc57d5e678d0c7c2129060066feaaf3', ''),
    ('MINT-SJTU/EVO-Depth-LIBERO', '5da43342257e4cf98728d83ca20d36e72babab0f', 'libero_spatial/'),
    ('OpenGVLab/InternVL3-1B', '4415a3b810e636d11dfa86b0e9ba40bb00535aa8', ''),
    ('depth-anything/DA3-BASE', 'f4a6c9b3c95e41c82048423d3493a81ec3fa810e', ''),
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Keep original audit receipt: ' + str(path))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'DRL-native-contract-audit/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def inventory():
    rows = []
    for folder, expected_url, expected_commit in SOURCES:
        local = ROOT / 'VLA/third_party' / folder
        def git(*args):
            return subprocess.check_output(['git', '-C', str(local), *args], text=True).strip()
        if git('remote', 'get-url', 'origin') != expected_url:
            raise ValueError('Unexpected source remote: ' + folder)
        if git('rev-parse', 'HEAD') != expected_commit or git('status', '--porcelain'):
            raise ValueError('Source is not the inspected, clean revision: ' + folder)
        selected = [p for p in local.rglob('*.py') if '.git' not in p.parts
                    and ('Evo_1' in p.parts or 'Evo_depth' in p.parts)]
        rows.append({'source': expected_url, 'commit': expected_commit,
                     'clean': True, 'local': str(local),
                     'reviewed_package_Python_hashes': {str(p.relative_to(local)): sha(p) for p in sorted(selected)},
                     'checkout_bytes': sum(p.stat().st_size for p in local.rglob('*') if p.is_file())})
    hubs = []
    for repo, revision, prefix in HUB:
        model = fetch(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true')
        if model['sha'] != revision or model.get('gated'):
            raise ValueError('Unpinned or gated repository: ' + repo)
        config = fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{prefix}config.json')
        files = []
        for item in model['siblings']:
            name = item['rfilename']
            # Do not buy/download four suite checkpoints for Evo-Depth.
            if prefix and not name.startswith(prefix):
                continue
            if name.endswith(('.json', '.pt', '.bin', '.safetensors')):
                files.append({'path': name, 'bytes': item.get('size'),
                              'lfs_sha256': (item.get('lfs') or {}).get('sha256')})
        hubs.append({'repository': repo, 'revision': revision, 'public_ungated': True,
                     'license_metadata': (model.get('cardData') or {}).get('license'),
                     'selected_files': files, 'selected_config': config,
                     'weights_downloaded_by_this_audit': False})
    save(OUT / 'PINNED_SOURCE_AND_CONFIG.json', {
        'created_Vietnam': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
        'sources': rows, 'models': hubs, 'scope': 'source/config/availability, NOT full-model qualification',
        'native_full_model_forward_passed': False, 'native_optimizer_and_reload_passed': False,
        'GPU_16GB_fit_demonstrated': False, 'new_robot_SR_measured': False,
        'running_jobs_or_frozen_sources_modified': False})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_audit():
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from VLA.family_transfer_r54.data import Data

    torch.set_num_threads(2)
    source = ROOT / 'VLA/third_party/evo1/Evo_1/model/internvl3/internvl3_embedder.py'
    native = load_module('evo1_source_pixel_audit_r78', source)
    # Run the real source method without calling its weight-loading constructor.
    encoder = native.InternVL3Embedder.__new__(native.InternVL3Embedder)
    torch.nn.Module.__init__(encoder)
    encoder.device, encoder.image_size = 'cpu', 448
    data = Data(horizon=50)
    records, sheets = [], []
    for task in ('touch', 'pick', 'move', 'pick_place'):
        ei = next(i for i, r in enumerate(data.rows)
                  if r['task'] == task and r['source_kind'] == 'complete_original'
                  and 'gelatin' in r['pick_name'])
        row, num = data.rows[ei], data.numeric(ei)
        raw = num['scene64']
        near = int(np.linalg.norm(raw[:, 11:14], axis=1).argmin())
        held = np.flatnonzero((raw[:, 31] > .5) & (raw[:, 30] < .5))
        # For touch, no held state is expected: review the mid-trajectory instead.
        transport = int(held[len(held)//2]) if len(held) else len(raw)//2
        selected = [('reset', 0), ('closest_approach', near),
                    ('held_or_midtrajectory', transport), ('terminal', len(raw)-1)]
        canvas = Image.new('RGB', (3 * 448, 4 * 474), 'white')
        draw = ImageDraw.Draw(canvas)
        for ri, (phase, fi) in enumerate(selected):
            sample = data.at(ei, fi)
            rgb = [sample[k] for k in ('front', 'wrist', 'map')]
            if any(x.dtype != np.uint8 or x.shape[-1] != 3 for x in rgb):
                raise ValueError('Source must be actual uint8 RGB')
            batch = [[torch.from_numpy(x.copy()).permute(2, 0, 1).float()/255. for x in rgb]]
            pixels, tiles = encoder._preprocess_images(batch)
            if tuple(pixels.shape) != (3, 3, 448, 448) or tiles != [[1, 1, 1]]:
                raise ValueError('Unexpected actual source image boundary')
            if not torch.isfinite(pixels).all() or sample['numeric67'].shape != (67,):
                raise ValueError('Incomplete/non-finite input')
            mean = torch.tensor(native.IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(native.IMAGENET_STD).view(1, 3, 1, 1)
            decoded = ((pixels.float()*std+mean)*255).round().clamp(0, 255).byte()
            for col, image in enumerate(decoded):
                canvas.paste(Image.fromarray(image.permute(1, 2, 0).numpy()), (448*col, 474*ri+26))
            draw.text((4, 474*ri+4), f'{task} | {phase} | t={fi} | front / wrist / GT map', fill='black')
            records.append({'task': task, 'phase_for_audit_only': phase, 'episode': ei, 'frame': fi,
                            'target': row['pick_name'], 'source': row['source'],
                            'source_sha256': row['source_sha256'], 'source_hash_verified_by_Data': True,
                            'numeric67_finite': bool(np.isfinite(sample['numeric67']).all()),
                            'actual_pixel_shape': list(pixels.shape), 'actual_tiles': tiles,
                            'native_pixels_sha256': hashlib.sha256(pixels.float().numpy().tobytes()).hexdigest(),
                            'sufficient_for_full_dataset_or_VLM_qualification': False})
        path = OUT / ('EVO1_NATIVE_INPUTS_' + task + '.png')
        if path.exists():
            raise FileExistsError(path)
        canvas.save(path)
        sheets.append(str(path))
    save(OUT / 'EVO1_PIXEL_BOUNDARY_AUDIT.json', {
        'actual_source_pixel_method_executed': True, 'source_sha256': sha(source),
        'TRAIN_only': True, 'four_tasks_four_states_each': True,
        'records': records, 'sheets': sheets,
        'full_VLM_forward_OR_model_SR_measured': False,
        'image_audit_is_not_model_training_admission': True,
        'EvoDepth_DA3_requires_physical_views_only_not_GT_map': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--images', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        inventory()
    if args.images:
        image_audit()
    if not (args.inventory or args.images):
        parser.error('Choose --inventory and/or --images')
    print(json.dumps({'output': str(OUT), 'new_model_SR_claim': False}))
