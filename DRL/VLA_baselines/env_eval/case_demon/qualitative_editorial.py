"""Publication-layout revision using immutable audited Oracle/X-VLA pixels.

No simulation, model calls, generated imagery, photometric editing or source
overwrites. Tight workspace crops are shared by the two executors and every
timepoint. Native hand-eye frames remain full 16:9. Labels stay vector.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from .qualitative_figure import read,write,digest,project,camera,detail_label

SOURCE=Path('/home/uet/Downloads/qualitative_comparison_redesign')
OUTPUT=Path('/home/uet/Downloads/qualitative_comparison_editorial_v2')
INK='#202C38';MUTED='#65707B';RULE='#C8CED5'
TARGET='#147DB3';GOAL='#B48A25';EVENT='#CD4B35';SUCCESS='#17805D'
GROUPS={1:('(a)','Both succeed'),2:('(b)','Oracle fails / X-VLA succeeds'),
        3:('(c)','Oracle succeeds / X-VLA fails'),4:('(d)','Both fail')}
SHORT={'Ground contact':'Contact','Off-target close':'Missed close','Empty closure':'Closed empty',
       'Grasp attempt':'Attempt','Still unplaced':'Unplaced','Empty motion':'Empty motion',
       'Not acquired':'Not acquired','Initial scene':'Initial','Recorded state':'Approach'}
OUTCOMES={'Success':'Placed','Recovered success':'Recovered','Transit drop':'Transit drop',
          'Ground collision':'Collision','Object ejection':'Ejection','Target miss':'Target miss',
          'Empty lift':'Empty lift','Grasp miss':'Grasp miss','Timeout':'Timeout'}


def configure():
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white',
        'axes.facecolor':'white','path.simplify':False})


def text(fig,label,x,y,W,H,*,size=8,weight='normal',color=INK,ha='left'):
    return fig.text(x/W,y/H,label,ha=ha,va='center',fontsize=size,fontweight=weight,
                    color=color,transform=fig.transFigure)


def line(fig,x1,y1,x2,y2,W,H,color=RULE,lw=.45):
    fig.add_artist(Line2D([x1/W,x2/W],[y1/H,y2/H],transform=fig.transFigure,color=color,linewidth=lw))


def crop_for(case,frames):
    points=[];cfg=camera(case['view'])
    for x in (-.35,.35):
        for y in (-.85,-.20):points.append(project([x,y,0],cfg,1920,1080))
    points.append(project(case['task']['place_position'],cfg,1920,1080))
    # Every displayed target/goal is retained, including the ejected object.
    # The initial elevated arm is not a reason to shrink the whole workspace.
    for method in case['methods']:
        fs=frames[method['name']]
        for i in method['selection']['detail_indices']:
            info=fs[i]['wide'][case['view']-1]
            for key in ('object_box','projected_pick_box','projected_place_box'):
                b=info.get(key)
                if b:points.extend((np.asarray(b[:2]),np.asarray(b[2:])))
        for mark in method['selection']['main']:
            p=project(fs[mark['index']]['pad_center'],cfg,1920,1080)
            points.extend((p+[-48,-78],p+[48,48]))
    pts=np.asarray([p for p in points if p is not None and np.isfinite(p).all()])
    lo=pts.min(0)-[25,23];hi=pts.max(0)+[25,23]
    ratio=2.15;w=max(hi[0]-lo[0],ratio*(hi[1]-lo[1]));h=w/ratio
    if w>1920 or h>1080:raise ValueError('Cannot crop all evidence without losing source coverage')
    c=(lo+hi)/2;x=int(np.clip(c[0]-w/2,0,1920-np.ceil(w)));y=int(np.clip(c[1]-h/2,0,1080-np.ceil(h)))
    crop=[x,y,x+int(np.ceil(w)),y+int(np.ceil(h))]
    if np.any(pts.min(0)<np.asarray(crop[:2])) or np.any(pts.max(0)>np.asarray(crop[2:])):
        raise ValueError('Crop excludes target, goal or selected gripper event')
    return crop


def load(out):
    manifest=read(SOURCE/'manifest.json')
    if not read(SOURCE/'status.json')['completed_delivery']:raise ValueError('Need audited source delivery')
    cases=copy.deepcopy(manifest['cases']);selection=[]
    for case in cases:
        frames={m['name']:read(Path(m['clean'])/'frames.json') for m in case['methods']}
        old=case['crop'];case['crop']=crop_for(case,frames);case['editorial_frames']=frames
        for method in case['methods']:
            for panel in method['panels']:
                i=panel['frame_index_0_based'];dest=out/'panels'/f"scene{case['scene_id']:02d}"/method['name']
                dest.mkdir(parents=True,exist_ok=True)
                for role in ('wide','handeye'):
                    item=panel['files'][role];source=SOURCE/item['file']
                    if digest(str(source))!=item['sha256']:raise ValueError('Source panel changed')
                    target=dest/source.name
                    if not target.exists():shutil.copy2(source,target)
                    if digest(str(target))!=item['sha256']:raise ValueError('Panel copy differs')
                    panel['files'][role]['source_file']=str(source)
                    panel['files'][role]['file']=str(target.relative_to(out))
                source=out/panel['files']['wide']['file'];path=dest/f'frame{i:05d}_workspace.png'
                with Image.open(source) as image:
                    cropped=image.crop(case['crop']);cropped.save(path)
                panel['files']['workspace_crop']={'file':str(path.relative_to(out)),
                    'sha256':digest(str(path)),'source_file':str(source),'crop_xyxy':case['crop']}
                panel['common_pair_crop_xyxy']=case['crop']
                selection.append({'scene':case['scene_id'],'method':method['name'],'frame':i,
                    'simulation_time_s':panel['simulation_time_s'],'view':case['view'],
                    'crop_xyxy':case['crop'],'previous_crop_xyxy':old,
                    'source_wide':panel['files']['wide']['source_file'],
                    'source_wide_sha256':panel['files']['wide']['sha256'],
                    'handeye':panel['files']['handeye'],'workspace':panel['files']['workspace_crop']})
    write(out/'frame_manifest.json',{'source_manifest':str(SOURCE/'manifest.json'),
        'source_manifest_sha256':digest(str(SOURCE/'manifest.json')),'same_events_and_outcomes_as_source':True,
        'no_new_rollouts':True,'no_photometric_changes':True,'handeye_full_frame_16_9':True,
        'all_pair_crops_constant':True,'records':selection})
    return cases,manifest


def corners(ax,box,crop,color=TARGET,pad=8,lw=.8):
    a,b,c,d=np.asarray(box,float);x0,y0,_,_=crop
    a-=x0+pad;b-=y0+pad;c+=pad-x0;d+=pad-y0
    length=min(38,max(9,min(c-a,d-b)*.25))
    for x,y,sx,sy in ((a,b,1,1),(c,b,-1,1),(a,d,1,-1),(c,d,-1,-1)):
        ax.plot([x,x+sx*length],[y,y],color=color,lw=lw,solid_capstyle='butt')
        ax.plot([x,x],[y,y+sy*length],color=color,lw=lw,solid_capstyle='butt')


def image_axis(fig,case,method,i,role,box,out):
    panel=next(p for p in method['panels'] if p['frame_index_0_based']==i)
    path=out/panel['files']['workspace_crop' if role=='wide' else 'handeye']['file']
    with Image.open(path) as source:pixels=np.asarray(source.convert('RGB')).copy()
    ax=fig.add_axes(box);ax.imshow(pixels,interpolation='none',aspect='equal')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#D9DDE1');spine.set_linewidth(.35)
    f=case['editorial_frames'][method['name']][i]
    crop=case['crop'] if role=='wide' else [0,0,1280,720]
    info=f['wide'][case['view']-1] if role=='wide' else f['handeye']
    target=info.get('object_box')
    if target:corners(ax,target,crop)
    elif role=='wide' and info.get('projected_pick_box'):
        corners(ax,info['projected_pick_box'],crop,lw=.6)
    if role=='wide':
        goal=info.get('projected_place_box')
        if goal:
            a,b,c,d=goal
            ax.add_patch(Rectangle((a-crop[0]-5,b-crop[1]-5),c-a+10,d-b+10,
                fill=False,edgecolor=GOAL,lw=.65,linestyle=(0,(2.5,1.5))))
        else:
            p=project(case['task']['place_position'],camera(case['view']),1920,1080)-np.asarray(crop[:2])
            ax.add_patch(Rectangle(p-[17,17],34,34,fill=False,edgecolor=GOAL,lw=.65,linestyle=(0,(2.5,1.5))))
    labels={x['index']:x['label'] for x in method['selection']['main']}
    if labels.get(i) in ('Impact','Ground contact','Off-target close','Empty closure'):
        cfg=camera(case['view']) if role=='wide' else f['handeye']['camera']
        p=project(f['pad_center'],cfg,1920 if role=='wide' else 1280,1080 if role=='wide' else 720)
        if p is not None:
            radius=27 if role=='wide' else 72
            corners(ax,[p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius],crop,EVENT,pad=0,lw=.9)
    return ax


def status_icon(fig,x,y,W,H,success):
    if success:
        line(fig,x,y-.1,x+.55,y-.65,W,H,SUCCESS,.9)
        line(fig,x+.55,y-.65,x+1.65,y+.75,W,H,SUCCESS,.9)
    else:
        line(fig,x,y-.6,x+1.2,y+.6,W,H,EVENT,.9)
        line(fig,x,y+.6,x+1.2,y-.6,W,H,EVENT,.9)


def cell(fig,case,method,i,label,x,top,w,W,H,out,*,final=False):
    wide_h=w/2.15;eye_h=w*9/16;bar=3.65
    if final:
        success=method['selection']['success'];color=SUCCESS if success else EVENT
        value=OUTCOMES[method['selection']['outcome'].rstrip('*')]
        if method['name']=='Scripted_Oracle' and not success:value+='*'
        status_icon(fig,x+.55,top-bar/2,W,H,success)
        label_x=x+2.7
    else:
        color=EVENT if label in ('Detached','Dropped','Impact','Ground contact','Off-target close','Empty lift','Empty closure') else INK
        value=SHORT.get(label,label);label_x=x+.5
    # One compact bar conveys event AND time, rather than two duplicate rows.
    text(fig,value,label_x,top-bar/2,W,H,size=8,weight='bold' if final else 'normal',color=color)
    t=next(p['simulation_time_s'] for p in method['panels'] if p['frame_index_0_based']==i)
    text(fig,f'{t:.1f} s',x+w-.45,top-bar/2,W,H,size=8,color=MUTED,ha='right')
    line(fig,x,top-bar,x+w,top-bar,W,H,RULE,.4)
    image_axis(fig,case,method,i,'wide',[x/W,(top-bar-wide_h)/H,w/W,wide_h/H],out)
    image_axis(fig,case,method,i,'handeye',[x/W,(top-bar-wide_h-.35-eye_h)/H,w/W,eye_h/H],out)
    return top-bar-wide_h-.35-eye_h


def legend(fig,W,H):
    text(fig,'Workspace above · HD hand-eye below (not actor input)',2,7,W,H,size=8,color=MUTED)
    text(fig,'* Scripted Oracle fault',W-2,7,W,H,size=8,color=MUTED,ha='right')
    for x,color,name,dashed in [(2,TARGET,'Target',False),(25,GOAL,'Destination',True),(57,EVENT,'Contact / miss',False)]:
        artist=Line2D([(x)/W,(x+4)/W],[2.5/H,2.5/H],transform=fig.transFigure,color=color,lw=1)
        if dashed:artist.set_linestyle((0,(2.5,1.5)))
        fig.add_artist(artist);text(fig,name,x+5.5,2.5,W,H,size=8,color=MUTED)
    text(fig,'Times are simulation seconds; event-aligned, not synchronized.',W-2,2.5,W,H,size=8,color=MUTED,ha='right')


def sheet(cases,stem,out,*,preview=False):
    configure();W=180.;margin=2.;gutter=4.;gap=1.2
    mw=(W-2*margin-gutter)/2;w=(mw-2*gap)/3
    block=3.65+w/2.15+.35+w*9/16
    row_h=4.5+block+1.5;H=10+len(cases)*row_h+11
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    text(fig,'Scripted Oracle',margin+mw/2,H-3,W,H,size=10.5,weight='bold',ha='center')
    text(fig,'X-VLA',margin+mw+gutter+mw/2,H-3,W,H,size=10.5,weight='bold',ha='center')
    line(fig,margin,H-6.5,W-margin,H-6.5,W,H,INK,.75)
    y=H-10
    for case in cases:
        group=GROUPS[case['group']]
        text(fig,group[0],margin,y-1.35,W,H,size=8.5,weight='bold')
        text(fig,f"S{case['scene_id']:02d}",margin+5.3,y-1.35,W,H,size=8.5,weight='bold')
        instruction=case['short_instruction'].replace('bottom right corner','lower-right region')
        text(fig,instruction,margin+13,y-1.35,W,H,size=8.5)
        text(fig,group[1],W-margin,y-1.35,W,H,size=8,color=MUTED,ha='right')
        for k,method in enumerate(case['methods']):
            x=margin+k*(mw+gutter)
            for j,q in enumerate(method['selection']['main']):
                cell(fig,case,method,q['index'],q['label'],x+j*(w+gap),y-4.5,w,W,H,out,final=j==2)
            if k==0:line(fig,W/2,y-4.5,W/2,y-4.5-block,W,H,'#E0E4E8',.45)
        y-=row_h
        line(fig,margin,y+.65,W-margin,y+.65,W,H,'#DDE2E7',.4)
    legend(fig,W,H)
    return export(fig,stem,out,preview=preview)


def case_plate(case,stem,out,*,extended=False,preview=False):
    configure();W=180.;n=6 if extended else 3;margin=2.;gap=1.15
    w=(W-2*margin-(n-1)*gap)/n;block=3.65+w/2.15+.35+w*9/16
    H=18+2*(block+6)+12
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    task=case['instruction'].replace('orange orange','orange')
    text(fig,f"S{case['scene_id']:02d}   {task}",margin,H-4,W,H,size=10,weight='bold')
    text(fig,GROUPS[case['group']][0]+' '+GROUPS[case['group']][1],margin,H-9,W,H,size=8.5,color=MUTED)
    y=H-15
    for method in case['methods']:
        text(fig,'Scripted Oracle' if method['name']=='Scripted_Oracle' else 'X-VLA',margin,y,W,H,size=9.5,weight='bold')
        marks=method['selection']['detail_indices'] if extended else [q['index'] for q in method['selection']['main']]
        main_labels={q['index']:q['label'] for q in method['selection']['main']}
        for j,i in enumerate(marks):
            lab=main_labels.get(i,detail_label(method,i))
            if lab=='Recorded state':lab='Recorded'
            cell(fig,case,method,i,lab,margin+j*(w+gap),y-3.5,w,W,H,out,final=j==n-1)
        y-=block+6
    legend(fig,W,H)
    return export(fig,stem,out,preview=preview)


def export(fig,stem,out,*,preview=False):
    path=out/stem;path.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.get_size_inches()*fig.dpi
    over=[];collisions=[]
    boxes=[(t,t.get_window_extent(renderer)) for t in fig.texts]
    for t,b in boxes:
        if b.x0<-.1 or b.y0<-.1 or b.x1>W+.1 or b.y1>H+.1:over.append(t.get_text())
    for j,(t,b) in enumerate(boxes):
        for u,c in boxes[j+1:]:
            # These figure-level texts are intentionally outside image areas.
            if min(b.x1,c.x1)-max(b.x0,c.x0)>.4 and min(b.y1,c.y1)-max(b.y0,c.y0)>.4:
                collisions.append([t.get_text(),u.get_text()])
    if over or collisions:
        plt.close(fig);raise ValueError(json.dumps({'outside':over,'text_collisions':collisions}))
    files=[]
    if not preview:
        for ext in ('pdf','svg'):
            target=path.with_suffix('.'+ext);temp=target.with_suffix('.'+ext+'.tmp')
            fig.savefig(temp,format=ext,dpi=600);temp.replace(target);files.append(str(target.relative_to(out)))
    for ax in fig.axes:
        for im in ax.images:im.set_interpolation('lanczos')
    if not preview:
        target=path.with_suffix('.png');temp=target.with_suffix('.png.tmp')
        fig.savefig(temp,format='png',dpi=600);temp.replace(target);files.append(str(target.relative_to(out)))
    target=path.with_name(path.name+'_preview').with_suffix('.jpg')
    fig.savefig(target,dpi=180,pil_kwargs={'quality':96});files.append(str(target.relative_to(out)))
    result={'stem':stem,'files':files,'size_mm':(fig.get_size_inches()*25.4).tolist(),
        'font':'Liberation Sans','minimum_text_pt':min(t.get_fontsize() for t in fig.texts),
        'text_overlap_pairs':collisions,'text_outside_canvas':over,
        'vector_images_use_original_native_pixels':True,'PNG_dpi':600}
    plt.close(fig);write(out/'layout_audits'/(stem.replace('/','__')+'.json'),result)
    return result


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    marker=out/'ownership.json'
    if marker.exists() and read(marker).get('schema')!='editorial_v2':raise ValueError('Unrelated folder')
    write(marker,{'schema':'editorial_v2','preserve_previous_delivery':str(SOURCE)})
    cases,source=load(out);byid={c['scene_id']:c for c in cases}
    if args.stage=='pilot':
        audit=sheet([byid[i] for i in (3,6,13)],'pilot/event_triptych',out,preview=args.preview)
        case_plate(byid[6],'pilot/large_ejection_case',out,preview=args.preview)
        case_plate(byid[3],'pilot/extended_drop_evidence',out,extended=True,preview=args.preview)
        write(out/'pilot.json',audit);return
    products=[('main_6_cases',lambda:sheet([byid[i] for i in source['main_ids']],'main_6_cases',out)),
        ('main_partA',lambda:sheet([byid[i] for i in source['main_ids'][:3]],'main_partA',out)),
        ('main_partB',lambda:sheet([byid[i] for i in source['main_ids'][3:]],'main_partB',out))]
    for group in GROUPS:
        stem=f'groups/group{group:02d}'
        products.append((stem,lambda g=group,s=stem:sheet([c for c in cases if c['group']==g],s,out)))
    for case in cases:
        sid=case['scene_id']
        products.append((f'cases/scene{sid:02d}',lambda c=case,s=f'cases/scene{sid:02d}':case_plate(c,s,out)))
        products.append((f'timelines/scene{sid:02d}',lambda c=case,s=f'timelines/scene{sid:02d}':case_plate(c,s,out,extended=True)))
    for stem,make in products:
        if args.only and stem not in args.only.split(','):continue
        print(json.dumps({'exporting':stem}),flush=True);make()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--stage',choices=('pilot','full'),default='pilot')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--only')
    main(parser.parse_args())
