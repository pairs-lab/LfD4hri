"""Decode genuine development video; preserve failures in diagnostic sheets."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    d = json.loads(args.progress.read_text())
    seed = d['rows'][0]['seed']
    rows = [r for r in d['rows'] if r['seed'] == seed]
    if len(rows) != 4:
        raise ValueError('Require all four tasks for recorded seed')
    source = Path(rows[0]['trajectory']).parent/'rollout.mp4'
    events, offset = {}, 0
    for j, row in enumerate(rows):
        for k, control in enumerate(sorted(set([0, row['steps']//3, 2*row['steps']//3, row['steps']-1]))):
            events[offset+control] = (j, k, row, control)
        offset += row['steps']
    sheet = Image.new('RGB', (1536, len(rows)*222), '#eeeeee')
    drawing, captures = ImageDraw.Draw(sheet), []
    video = cv2.VideoCapture(str(source))
    fps, count = video.get(cv2.CAP_PROP_FPS), 0
    while True:
        okay, bgr = video.read()
        if not okay:
            break
        if bgr.shape != (384, 1152, 3):
            raise ValueError('Unexpected diagnostic source layout')
        if count in events:
            j, k, row, control = events[count]
            rgb = cv2.cvtColor(bgr[:, 384:], cv2.COLOR_BGR2RGB)
            sheet.paste(Image.fromarray(rgb).resize((384, 192)), (k*384, j*222+30))
            drawing.text((k*384+3, j*222+3), f"{row['task']} control{control+1}/{row['steps']} success={row['success']}", fill='#111111')
            captures.append({'task':row['task'],'control':control+1,'source_frame':count,'success':row['success']})
        count += 1
    video.release()
    if count != offset or abs(fps-10) > 1e-6 or len(captures) != len(events):
        raise ValueError('Frame count/cadence/events do not match actual controls')
    sheet.save(args.output/'actual_post_action_camera_review.jpg', quality=97)
    atomic_json(args.output/'audit.json', {'passed':True,'family':d['family'],'seed':seed,
        'video':str(source),'video_sha256':sha256_file(source),'decoded_frames':count,'fps':fps,
        'captures':captures,'all_failures_preserved':True,
        'scope':'diagnostic_encoded_POST_action_front_wrist_views_NOT_exact_pre_inference_bytes_or_paper_HD_video',
        'policy_actions_or_environment_modified':False})
    print(json.dumps({'family':d['family'],'decoded_frames':count,'sheet':str(args.output/'actual_post_action_camera_review.jpg')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
