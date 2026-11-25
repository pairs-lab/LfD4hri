"""Decode every CASE_demon video; write MANIFEST.json, README.md and contact sheets."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from env_eval.case_demon.render import FPS, HEIGHT, VIEWS, WIDTH, rectangle
from env_eval.case_demon.robots import ROBOTS
from env_eval.case_demon.run import OUTPUT, WORK
from env_eval.case_demon.scenes import TASKS


def decode(path, inset):
    video = cv2.VideoCapture(str(path))
    size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    left, top, width, height = inset
    count, darkest, samples, border = 0, 255., [], []
    while True:
        ok, image = video.read()
        if not ok:
            break
        count += 1
        darkest = min(darkest, float(image.mean()))
        # The light panel frame must stay in place for the entire episode.
        border.append(float(image[top-2, left:left+width].mean()))
        samples.append(image) if count % 40 == 1 else None
    video.release()
    return {'size': size, 'fps': fps, 'frames': count, 'darkest_frame_mean': darkest,
            'inset_border_min': min(border), 'samples': samples}


def main():
    rows, problems, pairs = [], [], []
    (OUTPUT/'_review').mkdir(parents=True, exist_ok=True)
    for sid in range(1, len(TASKS)+1):
        report = json.loads((WORK/'qualify'/f'scene{sid:02d}.json').read_text())
        rendered = json.loads((WORK/'rendered'/f'scene{sid:02d}.json').read_text())
        scene = rendered['scene']
        pairs.append(json.dumps(scene['objects']))
        directory = OUTPUT/scene['robot']/f"env{scene['env']}"
        frames, sheet = {}, []
        for view in VIEWS:
            cfg = rendered['insets'][str(view)]
            path = directory/f'view_{view}.mp4'
            if not path.exists():
                problems.append(f'missing {path}')
                continue
            info = decode(path, rectangle(cfg['fraction'], cfg['side']))
            frames[view] = info['frames']
            if info['size'] != (WIDTH, HEIGHT) or abs(info['fps']-FPS) > 1e-6:
                problems.append(f'{path}: format {info["size"]} {info["fps"]}')
            if info['frames'] != rendered['frames']:
                problems.append(f'{path}: decoded {info["frames"]} != written {rendered["frames"]}')
            if info['darkest_frame_mean'] < 40:
                problems.append(f'{path}: dark frame {info["darkest_frame_mean"]:.1f}')
            if info['inset_border_min'] < 150:
                problems.append(f'{path}: inset frame moved')
            samples = info['samples']
            picks = [samples[int(t*(len(samples)-1))] for t in (0, .35, .55, .75, 1)]
            sheet.append(np.hstack([cv2.resize(s, (384, 216)) for s in picks]))
        if sheet:
            cv2.imwrite(str(OUTPUT/'_review'/f"{scene['robot']}_env{scene['env']}.jpg"), np.vstack(sheet))
        if len(set(frames.values())) != 1:
            problems.append(f'{directory}: view frame counts differ {frames}')
        if not rendered['overall_pass'] or not rendered['replay_matches_qualification']:
            problems.append(f'{directory}: physical result rejected')
        rows.append({'robot': scene['robot'], 'env': scene['env'], 'scene_id': sid, 'target': scene['target'],
                     'destination': scene['goal'], 'objects': scene['objects'],
                     'layout_candidate': scene['candidate'], 'frames': rendered['frames'], 'seconds': rendered['seconds'],
                     'final_xy_offset_m': rendered['final_xy_offset_m'], 'max_distractor_motion_m': max(rendered['distractor_motion_m'].values()),
                     'insets': rendered['insets'], 'rejected_attempts': [
                         {'candidate': a['scene']['candidate'], 'failure': a.get('failure'), 'unwanted_contacts': a['unwanted_contacts']}
                         for a in report['attempts'] if not a['overall_pass']]})
    videos = sorted(p.relative_to(OUTPUT).as_posix() for p in OUTPUT.glob('*/env*/view_*.mp4'))
    leftovers = sorted(str(p) for p in OUTPUT.rglob('*.partial.mp4'))
    if len(videos) != 45:
        problems.append(f'{len(videos)} videos, expected 45')
    if leftovers:
        problems.append(f'partial files: {leftovers}')
    if Counter(pairs).most_common(1)[0][1] != 1:
        problems.append('Two scenes share a layout')
    manifest = {'videos': len(videos), 'robots': list(ROBOTS), 'scenes': rows, 'problems': problems,
                'audit_pass': not problems, 'views': list(VIEWS), 'resolution': [WIDTH, HEIGHT], 'fps': FPS,
                'controller': 'scripted privileged suction Oracle with IK; NOT a trained RL policy, not success-rate evidence'}
    (OUTPUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    lines = ['# CASE_demon', '',
             '45 videos: 5 robots x 3 environments x views 3, 4, 6. Action: PUT with suction. 1920x1080, 30 FPS, real-time playback, no text.',
             'Scripted privileged Oracle (simulator poses, contacts, IK), not a trained RL policy. Every environment is different.', '',
             '| Robot | Env | Put | Into |', '|---|---:|---|---|']
    lines += [f"| {r['robot']} | {r['env']} | {r['target']} | {r['destination']} |" for r in rows]
    lines += ['', 'Layout: `Robot/envN/view_{3,4,6}.mp4`. `_review/` has first-to-last contact sheets; `MANIFEST.json` has physical results and rejected attempts.']
    (OUTPUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('videos', len(videos), 'audit_pass', not problems)
    for problem in problems:
        print(' -', problem)


if __name__ == '__main__':
    main()
