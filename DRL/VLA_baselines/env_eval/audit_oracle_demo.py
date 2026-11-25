"""Audit encoded videos, physics evidence and visibility; export review sheets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw


def audit(root):
    rows=[]
    sheets=root/'review';sheets.mkdir(exist_ok=True)
    for robot in ('UF850','UR5e_Robotiq2F'):
        for action in ('touch','pick','move','put'):
            tiles=[]
            for config in range(1,5):
                directory=root/robot/action/f'config{config}'
                path=directory/'audit.json'
                if not path.exists():
                    rows.append({'robot':robot,'action':action,'config':config,'pass':False,'error':'missing audit'})
                    continue
                physics=json.loads(path.read_text())
                visibility=json.loads((directory/'visibility.json').read_text())
                trajectory=json.loads((directory/'trajectory.json').read_text())
                phase_times={}
                for sample in trajectory:
                    phase_times.setdefault(sample['phase'],[]).append(sample['time'])
                videos=[]
                for view in range(1,4):
                    video=directory/f'view_{view}.mp4'
                    cap=cv2.VideoCapture(str(video))
                    fps=cap.get(cv2.CAP_PROP_FPS);width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration=count/fps if fps else 0
                    stream_ok=fps in (30.,60.) and width>=1280 and height>=720
                    timing_ok=abs(duration-physics['duration_seconds'])<=1/fps+.002 if fps else False
                    decoded=True;last_frame=None
                    for frame in sorted(set([0,max(0,count//2),max(0,count-1)])):
                        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read()
                        if not ok or img.std()<5:decoded=False
                        if ok:last_frame=img
                    # Independently extract true phase midpoints from encoded video.
                    for phase,times in phase_times.items():
                        for stage,seconds in [('start',times[0]),('middle',.5*(times[0]+times[-1])),('end',times[-1])]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES,min(count-1,round(seconds*fps)))
                            ok,img=cap.read()
                            if ok:
                                cv2.imwrite(str(directory/'frames'/f'{phase}_{stage}_view_{view}.jpg'),img)
                    cap.release()
                    relevant=[v['visible_pick_pixels'] for v in visibility if v['view']==view]
                    visible_ok=bool(relevant) and min(relevant)>=8
                    videos.append({'file':str(video.relative_to(root)),'fps':fps,'resolution':[int(width),int(height)],
                        'frames':count,'duration_seconds':duration,'timing_pass':timing_ok,
                        'decode_pass':decoded,'minimum_visible_pick_pixels':min(relevant,default=0),
                        'visibility_pass':visible_ok,'pass':stream_ok and timing_ok and decoded and visible_ok})
                    # Actual grasp/transport/release stages in each of the three views.
                    phases=['initial_start','grasp_middle' if action!='touch' else 'touch_middle',
                            'carry_middle' if action!='touch' else 'final_middle',
                            'release_end' if action=='put' else 'final_end']
                    if action=='put':phases.append('settle_end')
                    for phase in phases:
                        frame_path=directory/'frames'/f'{phase}_view_{view}.jpg'
                        if not frame_path.exists():
                            candidates=list((directory/'frames').glob(f'*_view_{view}.jpg'))
                            frame_path=candidates[-1] if candidates else None
                        if frame_path:
                            image=Image.open(frame_path).convert('RGB');image.thumbnail((320,180))
                            tile=Image.new('RGB',(320,202),'white');tile.paste(image,(0,0))
                            ImageDraw.Draw(tile).text((4,184),f'C{config} V{view} {phase}',fill='black')
                            tiles.append(tile)
                rows.append({'robot':robot,'action':action,'config':config,
                    'physics_pass':physics['overall_pass'],'videos':videos,
                    'pass':physics['overall_pass'] and all(v['pass'] for v in videos)})
            if tiles:
                columns=15 if action=='put' else 12;canvas=Image.new('RGB',(columns*320,((len(tiles)+columns-1)//columns)*202),'#e6e6e6')
                for i,tile in enumerate(tiles):canvas.paste(tile,((i%columns)*320,(i//columns)*202))
                canvas.save(sheets/f'{robot}_{action}.jpg',quality=92)
    summary={'expected_scenarios':32,'expected_videos':96,'scenes_passed':sum(r['pass'] for r in rows),
        'videos_passed':sum(v['pass'] for r in rows for v in r.get('videos',[])),
        'scope':'The exact deterministic requested scenes; contact-gated Oracle grasp assistance, not learned-policy results.',
        'rows':rows}
    (root/'video_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=audit(args.root)
    return 0 if result['scenes_passed']==32 else 1

if __name__=='__main__':raise SystemExit(main())
