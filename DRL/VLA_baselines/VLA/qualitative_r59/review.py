"""Extract NEW encoded-frame evidence without changing any R58 artifact."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'VLA/qualitative/ur5e_xvla_put_r58_complete_20261017'
OUT = ROOT/'VLA/qualitative/ur5e_xvla_put_r59_independent_review_20261017'
METHODS = ('Scripted_Oracle', 'XVLA_terminal_release')


def read(path):
    return json.loads(path.read_text())


def samples(frames, evidence):
    n = len(frames)
    if evidence['full_put_complete']:
        e = evidence['stable_completion_frame']
        raw = [0, n//4, n//2, max(0, e-45), e-30, e-21, e-12, e-6, e, e+6, e+15, n-1]
    else:
        event = max(evidence['events'], key=lambda r: r.get('object_displacement_within250ms_m', 0))
        mode = event['mode']
        e = event['event_frame']
        if 'failed_acquisition_empty_closure' in mode:
            contacts = [i for i, r in enumerate(frames) if r['target_contact_force_N'] > .005]
            e = contacts[0] if contacts else n//2
        if 'timeout' in mode:
            raw = np.linspace(0, n-1, 12).round().astype(int).tolist()
        else:
            raw = [e-15, e-9, e-3, e, e+2, e+4, e+7, e+12, e+24,
                   (e+n-1)//2, max(e+24, n-31), n-1]
    return [int(np.clip(i, 0, n-1)) for i in raw]


def decode_selected(path, indices):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot decode '+str(path))
    result = {}
    try:
        wanted = set(indices)
        for i in range(max(wanted)+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Incomplete decode: {path} frame{i}')
            if i in wanted:
                result[i] = frame
    finally:
        cap.release()
    return result


def fixed_crop(frames, indices):
    # A constant review ROI, never a moving camera or edited delivery video.
    boxes = [frames[i]['wide'][1]['object_box'] for i in indices[:9]
             if frames[i]['wide'][1].get('object_box')]
    if not boxes:
        return [0, 0, 1920, 1080]
    boxes = np.asarray(boxes)
    x0, y0 = np.min(boxes[:, :2], axis=0)
    x1, y1 = np.max(boxes[:, 2:], axis=0)
    x0, y0, x1, y1 = x0-140, y0-300, x1+140, y1+90
    w, h = max(512, x1-x0), max(288, y1-y0)
    w = max(w, h*16/9); h = w*9/16
    if w >= 1920 or h >= 1080:
        return [0, 0, 1920, 1080]
    cx, cy = (x0+x1)/2, (y0+y1)/2
    left, top = min(max(0, round(cx-w/2)), 1920-round(w)), min(max(0, round(cy-h/2)), 1080-round(h))
    return [left, top, left+round(w), top+round(h)]


def panel(frame, label, roi=None):
    if roi:
        x0, y0, x1, y1 = roi
        frame = frame[y0:y1, x0:x1]
    thumb = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA)
    # Labels only on review contact sheets, NEVER on source videos.
    canvas = np.full((314, 512, 3), 245, np.uint8)
    canvas[:288] = thumb
    cv2.putText(canvas, label, (8, 307), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = read(SOURCE/'produced_pairs.json')
    counts = Counter(p['group'] for p in pairs if not p['backup'])
    expected = dict(zip(('01_RL_succeed_VLA_succeed', '02_RL_fail_VLA_succeed',
                        '03_RL_succeed_VLA_fail', '04_RL_fail_VLA_fail'), (2, 3, 3, 3)))
    if counts != expected or len([p for p in pairs if p['backup']]) != 4:
        raise ValueError('Wrong scene coverage')
    records = []
    for p in pairs:
        pair = Path(p['paired_output'])
        label = ('backup' if p['backup'] else 'scene')+f"{p['scene_id']:02d}"
        dest = OUT/label
        dest.mkdir(exist_ok=True)
        for method in METHODS:
            directory = pair/method
            frames = read(directory/'frames.json')
            evidence = read(directory/'content_evidence.json')
            indices = samples(frames, evidence)
            roi = fixed_crop(frames, indices)
            wide = decode_selected(directory/'view_2.mp4', indices)
            wrist = decode_selected(directory/'handeye.mp4', indices)
            for name, decoded, crop in [('event_zoom', wide, roi), ('handeye', wrist, None)]:
                thumbs = [panel(decoded[i], f"f{i} / sim {frames[i]['simulation_time_s']:.3f}s", crop) for i in indices]
                grid = np.vstack([np.hstack(thumbs[k:k+4]) for k in range(0, 12, 4)])
                cv2.imwrite(str(dest/(method+'_'+name+'.jpg')), grid, [cv2.IMWRITE_JPEG_QUALITY, 97])
            full_indices = [0, indices[3], indices[8], len(frames)-1]
            # Decode reset separately when not part of an event window.
            full = decode_selected(directory/'view_2.mp4', full_indices)
            overview = np.hstack([panel(full[i], f"f{i} / sim {frames[i]['simulation_time_s']:.3f}s") for i in full_indices])
            cv2.imwrite(str(dest/(method+'_uncropped_overview.jpg')), overview, [cv2.IMWRITE_JPEG_QUALITY, 97])
            artifact_names = [method+'_'+suffix+'.jpg' for suffix in ('event_zoom', 'handeye', 'uncropped_overview')]
            record = {'scene': label, 'seed': p['seed'], 'method': method, 'source_pair': str(pair),
                'task_success': evidence['full_put_complete'], 'source_frame_indices': indices,
                'simulation_times_s': [frames[i]['simulation_time_s'] for i in indices],
                'encoded_PTS_s': [frames[i]['encoded_pts_s'] for i in indices],
                'review_only_fixed_crop_xyxy': roi, 'delivered_video_uncut_and_unchanged': True,
                'event_evidence': evidence,
                'source_video_sha256': {name: sha256_file(directory/(name+'.mp4')) for name in ('view_2', 'handeye')},
                'images': {name: sha256_file(dest/name) for name in artifact_names},
                'actual_manual_inspection_pending': True}
            atomic_json(dest/(method+'_review.json'), record)
            records.append(record)
            atomic_json(OUT/'extraction_progress.json', {'completed_rollouts': len(records), 'planned_rollouts': 30})
            print(label, method, indices, flush=True)
    atomic_json(OUT/'inventory.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'source': str(SOURCE), 'main_group_pair_counts': expected,
        'main_videos': 22, 'backup_videos': 8, 'camera_streams': 210, 'records': records})


def verify_manifest():
    expected = read(SOURCE/'MANIFEST.sha256.json')
    errors = []; inode_cache = {}
    for name, digest in expected.items():
        path = SOURCE/name
        try:
            st = path.stat(); key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            actual = inode_cache.get(key)
            if actual is None:
                actual = sha256_file(path); inode_cache[key] = actual
            if actual != digest: errors.append(name)
        except FileNotFoundError:
            errors.append(name)
    atomic_json(OUT/'independent_delivery_integrity.json', {'passed': not errors,
        'verified_manifest_entries': len(expected), 'unique_inodes_read': len(inode_cache),
        'errors': errors, 'original_manifest_sha256': sha256_file(SOURCE/'MANIFEST.sha256.json')})
    if errors: raise ValueError('Delivery modified: '+str(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify: verify_manifest()
    else: prepare()
