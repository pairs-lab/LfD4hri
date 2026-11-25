"""Twelve actual encoded event frames, plus uncut-view and hand-eye context."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_r59.review import decode_selected,panel,samples
from VLA.qualitative_r57.common import read
from utils.camera import CINEMATIC_VIEWS


def project(point):
    cfg=CINEMATIC_VIEWS[1]
    view=np.array(p.computeViewMatrix(cfg['eye'],cfg['target'],cfg['up'])).reshape(4,4,order='F')
    proj=np.array(p.computeProjectionMatrixFOV(cfg['fov'],1920/1080,.015,4.)).reshape(4,4,order='F')
    clip=proj@view@np.r_[point,1.];ndc=clip[:3]/clip[3]
    return np.array([(ndc[0]+1)*960,(1-ndc[1])*540])


def roi_for(frames,indices,ground):
    points=[project(frames[i]['pad_center']) for i in indices]
    if not ground:
        for i in indices:
            box=frames[i]['wide'][1].get('object_box')
            if box:points.extend([box[:2],box[2:]])
    points=np.array(points)
    lo=points.min(axis=0)-[140,250];hi=points.max(axis=0)+[140,150]
    w=max(640,hi[0]-lo[0],(hi[1]-lo[1])*16/9);h=w*9/16
    if w>=1920 or h>=1080:return [0,0,1920,1080]
    center=(lo+hi)/2
    x=int(np.clip(center[0]-w/2,0,1920-w));y=int(np.clip(center[1]-h/2,0,1080-h))
    return [x,y,x+round(w),y+round(h)]


def main(pair):
    out=pair/'review/independent';out.mkdir(exist_ok=False)
    results=[]
    for method in ('Scripted_Oracle','XVLA_terminal_release'):
        src=pair/method;frames=read(src/'frames.json');evidence=read(src/'content_evidence.json')
        indices=samples(frames,evidence)
        ground=any(e['mode']=='STAGED_Oracle_Ground_Slam' for e in evidence['events'])
        roi=roi_for(frames,indices[:9],ground)
        wide=decode_selected(src/'view_2.mp4',indices);handeye=decode_selected(src/'handeye.mp4',indices)
        names=[]
        for suffix,decoded,crop in [('event_zoom',wide,roi),('handeye',handeye,None),('wide_context',wide,None)]:
            thumbs=[panel(decoded[i],f'f{i} / sim {frames[i]["simulation_time_s"]:.3f}s',crop) for i in indices]
            grid=np.vstack([np.hstack(thumbs[k:k+4]) for k in range(0,12,4)])
            name=method+'_'+suffix+'.jpg';names.append(name)
            cv2.imwrite(str(out/name),grid,[cv2.IMWRITE_JPEG_QUALITY,97])
        if read(pair/'PAIR_AUDIT.json')['scene_id']==13:
            view1=decode_selected(src/'view_1.mp4',indices)
            thumbs=[panel(view1[i],f'f{i} / sim {frames[i]["simulation_time_s"]:.3f}s') for i in indices]
            grid=np.vstack([np.hstack(thumbs[k:k+4]) for k in range(0,12,4)])
            name=method+'_recommended_view1.jpg';names.append(name)
            cv2.imwrite(str(out/name),grid,[cv2.IMWRITE_JPEG_QUALITY,97])
        results.append({'method':method,'frames':indices,'review_only_fixed_roi':roi,
            'source_video_sha256':sha256_file(src/'view_2.mp4'),
            'actual_encoded_frames_not_animation':True,
            'images':{name:sha256_file(out/name) for name in names},
            'actual_visual_review_pending':True})
    atomic_json(out/'extraction.json',results)
    print(out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pair',type=Path,required=True)
    a.add_argument('--approve',action='store_true');a.add_argument('--note')
    opts=a.parse_args()
    if opts.approve:
        if not opts.note:raise ValueError('Actual manually observed findings required')
        out=opts.pair/'review/independent';records=read(out/'extraction.json')
        for row in records:
            for name,sha in row['images'].items():
                if sha256_file(out/name)!=sha:raise ValueError('Review image changed')
            row.update(actual_visual_review_pending=False,actual_agent_findings=opts.note,
                source_scope='12 sampled decoded event/aftermath frames per stream; NOT continuous every-frame viewing')
        atomic_json(out/'actual_visual_review.json',{'passed':True,'reviewer':'Codex after view_image inspection',
            'opened_images':{name:sha for row in records for name,sha in row['images'].items()},
            'findings':opts.note,'records':records})
    else:main(opts.pair)
