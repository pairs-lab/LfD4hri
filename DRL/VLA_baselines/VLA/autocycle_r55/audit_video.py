"""Decode every frame of the first-seed diagnostic video, export real events.

This is post-action recording, not exact pre-inference pixels or paper HD.
No failure is removed because its task failed or the policy left camera FOV.
"""
import argparse
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import read,TASKS


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    d=read(a.progress)
    if d['completed_trials']!=d['planned_trials'] or not d['rows']:raise ValueError('Complete actual cohort required')
    seed=d['rows'][0]['seed'];rows=[r for r in d['rows'] if r['seed']==seed]
    if len(rows)!=4 or {r['task'] for r in rows}!=set(TASKS):raise ValueError('All four recorded tasks required')
    source=Path(rows[0]['trajectory']).parent/'rollout.mp4';events={};offset=0
    for j,row in enumerate(rows):
        for k,control in enumerate(sorted(set([0,row['steps']//3,2*row['steps']//3,row['steps']-1]))):
            events[offset+control]=(j,k,row,control)
        offset+=row['steps']
    sheet=Image.new('RGB',(1536,4*222),'#eeeeee');draw=ImageDraw.Draw(sheet)
    stream=cv2.VideoCapture(str(source));fps=stream.get(cv2.CAP_PROP_FPS);count=0;captures=[]
    while True:
        okay,bgr=stream.read()
        if not okay:break
        if bgr.shape!=(384,1152,3):raise ValueError('Unexpected native diagnostic camera layout')
        if count in events:
            j,k,row,control=events[count]
            rgb=cv2.cvtColor(bgr[:,384:],cv2.COLOR_BGR2RGB)
            sheet.paste(Image.fromarray(rgb).resize((384,192)),(384*k,222*j+30))
            draw.text((384*k+3,222*j+3),f"{row['task']} control{control+1}/{row['steps']} success={row['success']}",fill='#111111')
            captures.append({'task':row['task'],'control':control+1,'video_frame':count,'task_success':row['success']})
        count+=1
    stream.release()
    if count!=offset or abs(fps-10)>1e-6 or len(captures)!=len(events):
        raise ValueError('Actual decode/frame-count/cadence does not match controls')
    sheet.save(a.output/'actual_post_action_camera_review.jpg',quality=97)
    atomic_json(a.output/'audit.json',{'passed':True,'model':a.model,'video':str(source),'seed':seed,
        'video_sha256':sha256_file(source),'all_stream_frames_decoded':count,'fps':fps,'captures':captures,
        'manual_visual_review_done':False,'automatic_decode_and_event_export_only':True,
        'task_failures_preserved_NOT_camera_failure_filters':True,'policy_actions_or_physics_changed':False,
        'scope':'encoded_POST_action_diagnostic_NOT_exact_pre_inference_RGB_or_paper_HD',
        'safety_rollback_contact_cause_requires_pre_rollback_trace_NOT_just_this_video':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--progress',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
