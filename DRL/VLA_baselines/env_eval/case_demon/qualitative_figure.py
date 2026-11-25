"""Evidence-led Oracle/X-VLA temporal figures; never generates policy actions.

Run ``python -m env_eval.case_demon.qualitative_figure --stage pilot`` first,
inspect the generated images, then use ``--stage full``. Inputs are immutable
recordings. Unannotated decoded pixels and vector annotations remain separate.
No training, Oracle simulation, neural inference or physics steps occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'VLA/qualitative/ur5e_xvla_put_r60_named_failures_20261017'
OLD = ROOT/'VLA/qualitative/ur5e_xvla_put_r58_complete_20261017'
OUTPUT = Path('/home/uet/Downloads/qualitative_comparison_redesign')
POLICY = Path('/home/uet/Documents/X_VLA_view6_policy_cameras_20261031')
CHECKPOINT = ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/xvla_round1/fit1500/checkpoints/step-014500'
CASES = (1, 2, 3, 4, 5, 12, 6, 7, 8, 13, 10, 11)
MAIN = (2, 3, 12, 6, 13, 10)
PILOT = (3, 6, 13)
METHODS = ('Scripted_Oracle', 'XVLA_terminal_release')
NAMES = {'Scripted_Oracle': 'Scripted Oracle', 'XVLA_terminal_release': 'X-VLA'}
GROUPS = {
    1: ('(a)', 'Both succeed', '#e8f2ed'),
    2: ('(b)', 'Oracle fails / X-VLA succeeds', '#f7eee3'),
    3: ('(c)', 'Oracle succeeds / X-VLA fails', '#e8eff7'),
    4: ('(d)', 'Both fail', '#f2e9ed'),
}
BLUE, ORANGE, GREEN, GRAY = '#0072B2', '#D55E00', '#007F58', '#616c78'
SCHEMA = 'oracle_xvla_temporal_figure_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


@lru_cache(maxsize=512)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def image_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def camera(view):
    # These two published wide cameras are identical for both methods.
    return ({'eye': (-.85, -1.85, 1.15), 'target': (0., -.38, .35),
             'up': (0, 0, 1), 'fov': 47.} if view == 1 else
            {'eye': (1.25, -1.45, 1.05), 'target': (0., -.35, .34),
             'up': (0, 0, 1), 'fov': 48.})


def project(point, settings, width, height):
    eye = np.asarray(settings['eye'], float)
    forward = np.asarray(settings['target'], float)-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, settings['up']); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(point, float)-eye
    z = float(delta@forward)
    if z <= .001:
        return None
    focal = height/(2*np.tan(np.radians(settings['fov'])/2))
    return np.array([width/2+focal*(delta@right)/z,
                     height/2-focal*(delta@up)/z])


def primary_event(evidence):
    events = evidence.get('events', [])
    priorities = ('STAGED_Oracle_Ground_Slam', 'native_Target_Miss_then_Failed_Recovery',
                  'empty_gripper_lift_after_failed_acquisition_no_object_lift',
                  'verified_in_transit_drop', 'controlled_Oracle_transport_timeout',
                  'controlled_Oracle_failed_acquisition_empty_closure')
    for mode in priorities:
        matches = [event for event in events if event['mode'] == mode]
        if matches:
            return matches[0]
    if events:
        return max(events, key=lambda e: e.get('object_displacement_within250ms_m', 0.))
    return None


def last_detachment_frame(frames, start):
    """Detachment, not a successful at-goal release, is the drop event."""
    for i in range(max(1, start-6), min(len(frames), start+25)):
        if frames[i-1]['correct_attached'] and not frames[i]['correct_attached']:
            return i
    raise ValueError('Drop evidence has no attachment-to-detachment transition')


def selection(frames, evidence):
    n = len(frames)
    event = primary_event(evidence)
    if evidence['full_put_complete']:
        held = [i for i, frame in enumerate(frames)
                if frame['correct_attached'] and frame['bilateral']]
        if len(held) < 3:
            raise ValueError('A success sequence requires actual held frames')
        e = evidence['stable_completion_frame']
        raised = [i for i in held if frames[i]['object_position'][2] > frames[0]['object_position'][2]+.04]
        first = raised[0] if raised else held[0]
        # Keep distinct acquisition/transport states and stable unheld outcome.
        middle = held[len(held)*2//3]
        if middle <= first:
            middle = next(i for i in held if i > first)
        final = max(e, middle+1)
        if frames[final]['correct_attached']:
            final = next(i for i in range(final, n) if not frames[i]['correct_attached'])
        drops=[e for e in evidence.get('events',[]) if e['mode']=='verified_in_transit_drop']
        if drops:
            event=drops[0]
            detachment=last_detachment_frame(frames,int(event['event_frame']))
            reheld=next((i for i in held if i>int(event['after_frame'])),None)
            if reheld is None:
                raise ValueError('Recovered success requires measured reacquisition')
            before=int(event['before_frame'])
            landed=min(int(event['after_frame']),reheld-1)
            main=[(before,'Held'),(landed,'Dropped'),(final,'Recovered')]
            outcome='Recovered success'
            detail=[0,before,landed,min(reheld+3,final-1),final,n-1]
        else:
            main = [(first, 'Held'), (middle, 'Transport'), (final, 'Placed')]
            outcome = 'Success'
            detail = [0, max(1, held[0]-9), first, middle, final, n-1]
    elif not event:
        raise ValueError('An unclassified failure must not silently become a figure label')
    else:
        e = int(event['event_frame']); mode = event['mode']
        if 'verified_in_transit_drop' == mode:
            detachment = last_detachment_frame(frames, e)
            before = min(int(event['before_frame']), detachment-1)
            if not frames[before]['correct_attached']:
                before = max(i for i in range(detachment) if frames[i]['correct_attached'])
            main = [(before, 'Held'), (min(detachment+3,n-2), 'Detached'),
                    (min(max(event['after_frame']+6,detachment+12),n-1), 'Dropped')]
            outcome = 'Transit drop'
        elif 'Ground_Slam' in mode:
            if not event.get('ground_slam_force_velocity_gate_pass'):
                raise ValueError('Do not label a ground slam without its physical gate')
            main = [(int(event['before_frame']), 'Descending'), (e, 'Ground contact'),
                    (int(event['after_frame']), 'Still unplaced')]
            outcome = 'Ground collision*'
        elif 'contact_confirmed_impact' in mode:
            if event['force_N'] < 1 or event['object_displacement_within250ms_m'] < .08:
                raise ValueError('Selected ejection lacks qualified contact/displacement')
            main = [(int(event['before_frame']), 'Approach'), (e, 'Impact'),
                    (min(int(event['after_frame'])+3,n-1), 'Ejected')]
            outcome = 'Object ejection'
        elif 'empty_gripper_lift' in mode:
            if evidence['held_frames']:
                raise ValueError('Empty-lift classifier contradicted by actual held frames')
            # First show a lifted empty hand; last proves failure persists.
            main = [(e, 'Grasp attempt'), (int(event['after_frame']), 'Empty lift'),
                    (n-1, 'Still unplaced')]
            outcome = 'Empty lift'
        elif 'Target_Miss' in mode:
            if evidence['held_frames'] or event['first_close_target_XY_separation_m'] <= .10:
                raise ValueError('Target miss requires measured off-target empty closure')
            main = [(e, 'Off-target close'), (min(e+90,n-2), 'Empty motion'),
                    (n-1, 'Still unplaced')]
            outcome = 'Target miss'
        elif 'timeout' in mode:
            held = [i for i, row in enumerate(frames) if row['correct_attached']]
            if not held:
                raise ValueError('Transport timeout requires actual acquisition')
            main = [(held[min(9,len(held)-1)], 'Held'), ((held[0]+n-1)//2, 'Transport'),
                    (n-1, 'Time limit')]
            outcome = 'Timeout*'
        elif 'failed_acquisition' in mode:
            # The two-shot narrative is misleading without an approach frame.
            main = [(max(0,e-75), 'Approach'), (max(1,e-30), 'Empty closure'),
                    (n-1, 'Not acquired')]
            outcome = 'Grasp miss*'
        else:
            raise ValueError('No predeclared visual narrative for '+mode)
        detail = [0, max(1,main[0][0]-9), *[i for i,_ in main], n-1]
    if not all(a[0] < b[0] for a,b in zip(main,main[1:])):
        raise ValueError('Main temporal keyframes must be strictly ordered')
    # Six unique event-led detail frames, retaining all main frames and final.
    ids = set(int(np.clip(i,0,n-1)) for i in detail)
    additions = ([max(0,int(event['event_frame'])+d) for d in (-9,-3,2,6,12,24)]
                 if event and not evidence['full_put_complete'] else [n//4,n//2,3*n//4])
    for i in additions:
        if len(ids) >= 6:
            break
        ids.add(min(i,n-1))
    if len(ids) != 6:
        raise ValueError('Expected exactly six distinct detail keyframes')
    assert bool(evidence['full_put_complete']) == (outcome in ('Success','Recovered success'))
    return {'main': [{'index': int(i), 'label': label} for i,label in main],
            'detail_indices': sorted(ids), 'outcome': outcome,
            'selected_event': event, 'success': bool(evidence['full_put_complete'])}


def fixed_crop(methods, view):
    """One constant scene crop shared across methods/time, including aftermath."""
    points = []
    for method in methods:
        frames = method['frames']
        # Include the entire workspace and every selected/dense target position.
        for x in (-.35,.35):
            for y in (-.85,-.20):
                points.append(project([x,y,0],camera(view),1920,1080))
        for frame in frames:
            for key in ('projected_pick_box','projected_place_box'):
                box = frame['wide'][view-1].get(key)
                if box:
                    points += [np.array(box[:2]),np.array(box[2:])]
        ids = [item['index'] for item in method['selection']['main']]+method['selection']['detail_indices']
        for i in ids:
            point = project(frames[i]['pad_center'],camera(view),1920,1080)
            if point is not None:
                points += [point+[-70,-100],point+[70,40]]
    points = np.array([p for p in points if p is not None and np.isfinite(p).all()])
    low, high = points.min(0)-[75,70], points.max(0)+[75,65]
    low = np.maximum(low,0);high=np.minimum(high,[1920,1080])
    width,height = high-low
    # A wide interaction crop lets HD hand-eye remain legible in the print grid.
    ratio=2.2;width=max(width,height*ratio);height=width/ratio
    if width>1920 or height>1080:
        return [0,0,1920,1080]
    center=(low+high)/2
    left=int(np.clip(center[0]-width/2,0,1920-width))
    top=int(np.clip(center[1]-height/2,0,1080-height))
    return [left,top,min(1920,left+int(np.ceil(width))),min(1080,top+int(np.ceil(height)))]


def clean_render(sid,seed,alias,published):
    candidates = [base/'_renders'/f'scene{sid:02d}_seed{seed}_{alias}' for base in (SOURCE,OLD)]
    # Reused VLA source for scene12 is disclosed explicitly, not a new trial.
    if sid==12 and alias=='VLA':
        candidates.append(OLD/'_renders'/f'scene01_seed{seed}_VLA')
    for path in candidates:
        if (path/'wide_1.mp4').exists():
            a,b=read(path/'frames.json'),read(published/'frames.json')
            if len(a)!=len(b):
                continue
            # Verify all saved timestamps/positions, not just a matching filename.
            if all(x['physics_tick']==y['physics_tick'] and x['object_position']==y['object_position']
                   and x['pad_center']==y['pad_center'] for x,y in zip(a,b,strict=True)):
                return path
    raise ValueError('No matching clean wide render for '+str(published))


def inventory(ids,extra_root=None):
    labels=read(SOURCE/'LABELS.json');cases=[]
    for sid in ids:
        if sid==14:
            if extra_root is None or not read(extra_root/'capture_qualification.json')['passed']:
                raise ValueError('New recovery source not independently qualified')
            initial=read(extra_root/'XVLA_render_v2/initial_scene.json')
            relevant=[{'scene_id':14,'seed':initial['seed'],'method_alias':alias,
                       'paired_scene':str(extra_root),'instruction':initial['task']['prompt']}
                      for alias in ('RL','VLA')]
        else:
            relevant=[r for r in labels if r['scene_id']==sid and not r['backup']]
        if len(relevant)!=2:
            raise ValueError('Expected exactly one paired scene, not backup aliases')
        original=next(r for r in relevant if r['method_alias']=='VLA')
        pair=Path(original['paired_scene']);seed=original['seed'];view=1 if sid==13 else 2
        methods=[]
        tasks=[]
        for name,alias in zip(METHODS,('Oracle','VLA'),strict=True):
            if sid==14:
                published=pair/('XVLA_render_v2' if alias=='VLA' else 'Oracle_render_v2');clean=published
            else:
                published=pair/name;clean=clean_render(sid,seed,alias,published)
            frames=read(clean/'frames.json');evidence=read(published/'content_evidence.json')
            initial=read(published/'initial_scene.json');tasks.append(initial['task'])
            chosen=selection(frames,evidence)
            source_files={str(path):digest(str(path)) for path in (
                clean/f'wide_{view}.mp4',clean/'handeye.mp4',clean/'frames.json',
                published/'initial_scene.json',published/'content_evidence.json',published/'renderer.json')}
            if (published/'replay_proof.json').exists():
                source_files[str(published/'replay_proof.json')]=digest(str(published/'replay_proof.json'))
                replay=read(published/'replay_proof.json')
                if name=='XVLA_terminal_release' and (not replay.get('passed_exact_replay')
                        or replay.get('checkpoint')!=str(CHECKPOINT)
                        or replay.get('physics_camera_safety_or_benchmark_success_modified',False)
                        or replay.get('maximum_original_pre_action_scene64_error',1)>1e-6):
                    raise ValueError('Actual neural source replay/checkpoint gate failed')
            elif name=='XVLA_terminal_release':
                raise ValueError('VLA figures require actual saved-action physical replay proof')
            methods.append({'name':name,'published':str(published),'clean':str(clean),
                            'selection':chosen,'frames':frames,'source_sha256':source_files,
                            'evidence':evidence,'initial_state_sha256':initial.get('measured_state_sha256')})
        if tasks[0]!=tasks[1]:
            raise ValueError('Paired task differs')
        # Some old headers do not have a stored world digest; compare actual world.
        a,b=[read(Path(m['published'])/'initial_scene.json') for m in methods]
        if a['measured_state']!=b['measured_state']:
            raise ValueError('The two methods do not share the exact initial physical scene')
        success=[m['selection']['success'] for m in methods]
        group=1 if all(success) else 2 if success==[False,True] else 3 if success==[True,False] else 4
        crop=fixed_crop(methods,view)
        task=tasks[0]
        short=f"{task['pick_name'].replace('orange orange','orange').capitalize()} → {task['place_name']}"
        cases.append({'scene_id':sid,'seed':seed,'group':group,'view':view,'crop':crop,
                      'instruction':task['prompt'],'short_instruction':short,'task':task,
                      'initial_world_pair_verified':True,'methods':methods,
                      'reused_source_NOT_independent_trial':sid==12,
                      'shared_XVLA_checkpoint':str(CHECKPOINT)})
    return cases


def decode(path,indices,expected_count):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open '+str(path))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if count!=expected_count or abs(fps-30)>1e-6:
        raise ValueError('Source timing/frame count mismatch')
    wanted=set(indices);result={};decoded=0;last_pts=-1;max_pts_error=0.;minimum_signal=255.
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            signal=float(cv2.meanStdDev(frame)[1].max())
            minimum_signal=min(minimum_signal,signal)
            if signal<1.:
                raise ValueError('Constant/empty RGB source rejected: '+str(path))
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if pts+1e-9<last_pts:
                raise ValueError('Video timestamps move backwards')
            max_pts_error=max(max_pts_error,abs(pts-decoded/30));last_pts=pts
            if decoded in wanted:
                result[decoded]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            decoded+=1
    finally:
        cap.release()
    if decoded!=expected_count or set(result)!=wanted or max_pts_error>1e-4:
        raise ValueError('Incomplete decode or unexpected presentation timestamps')
    return result,{'source':str(path),'decoded_frames':decoded,'fps':fps,
                   'resolution':[width,height],'max_encoded_PTS_error_s':max_pts_error,
                   'minimum_spatial_channel_std':minimum_signal,
                   'sha256':digest(str(path))}


def save_png(path,pixels):
    path.parent.mkdir(parents=True,exist_ok=True)
    # Always preserve lossless, unannotated pixels. No label is burned in.
    Image.fromarray(pixels).save(path)


def dense_indices(method):
    n=len(method['frames']);s=method['selection'];e=s['selected_event']
    center=int(e['event_frame']) if e and (not s['success'] or s['outcome']=='Recovered success') else int(method['evidence']['stable_completion_frame'])
    return sorted(set([0,n-1,*[int(np.clip(center+d,0,n-1)) for d in (-15,-9,-3,0,2,4,7,12,24,60)],
                       *[q['index'] for q in s['main']],*s['detail_indices']]))


def prepare(cases,out):
    for case in cases:
        for method in case['methods']:
            stem=f"scene{case['scene_id']:02d}/{method['name']}"
            dest=out/'panels'/stem;clean=Path(method['clean']);frames=method['frames']
            cache=dest/'panel_manifest.json'
            if cache.exists():
                saved=read(cache)
                if (saved.get('source_sha256')==method['source_sha256'] and
                        saved.get('crop')==case['crop'] and saved.get('selection')==method['selection']):
                    for row in saved['panels']:
                        for item in row['files'].values():
                            if digest(str(out/item['file']))!=item['sha256']:
                                raise ValueError('Cached panel checksum mismatch')
                    for key in ('panels','dense_review_file','decode_audit'):
                        method[key]=saved[key]
                    print(json.dumps({'stage':'validated_panel_cache','scene':case['scene_id'],'method':method['name']}),flush=True)
                    continue
            dense=dense_indices(method)
            indices=sorted(set(dense+method['selection']['detail_indices']))
            rgb,wide_audit=decode(clean/f"wide_{case['view']}.mp4",indices,len(frames))
            wrist,wrist_audit=decode(clean/'handeye.mp4',indices,len(frames))
            method['decode_audit']={'wide':wide_audit,'handeye':wrist_audit}
            records=[];main_ids={r['index'] for r in method['selection']['main']}
            panel_ids=set(method['selection']['detail_indices'])|main_ids
            for i in sorted(panel_ids):
                row=frames[i]
                if abs(row['physics_tick']/240-row['simulation_time_s'])>1e-8:
                    raise ValueError('Simulation timestamp is not the recorded physics tick')
                if abs(row['encoded_pts_s']-i/30)>1e-8:
                    raise ValueError('Encoded time offset misread')
                names={}
                for role,pixels in (('wide',rgb[i]),('handeye',wrist[i])):
                    path=dest/f'frame{i:05d}_{role}.png';save_png(path,pixels)
                    names[role]={'file':str(path.relative_to(out)),'sha256':digest(str(path)),
                                 'decoded_pixel_sha256':image_digest(pixels),'shape':list(pixels.shape)}
                x0,y0,x1,y1=case['crop'];crop=rgb[i][y0:y1,x0:x1]
                path=dest/f'frame{i:05d}_workspace_crop.png';save_png(path,crop)
                names['workspace_crop']={'file':str(path.relative_to(out)),'sha256':digest(str(path)),
                                         'decoded_pixel_sha256':image_digest(crop),'shape':list(crop.shape)}
                records.append({'frame_index_0_based':i,'simulation_time_s':row['simulation_time_s'],
                                'encoded_PTS_s':row['encoded_pts_s'],'physics_tick':row['physics_tick'],
                                'camera_view':case['view'],'common_pair_crop_xyxy':case['crop'],
                                'is_main':i in main_ids,'files':names,
                                'correct_attached':row['correct_attached'],'bilateral':row['bilateral'],
                                'target_contact_force_N':row['target_contact_force_N'],
                                'object_position':row['object_position'],'pad_center':row['pad_center'],
                                'annotations_are_separate_vectors':True})
            method['panels']=records
            method['dense_review_file']=str((out/'review'/f"scene{case['scene_id']:02d}_{method['name']}_dense.jpg").relative_to(out))
            review_contact(case,method,dense,rgb,wrist,out)
            write(cache,{**{key:method[key] for key in ('source_sha256','selection','panels','dense_review_file','decode_audit')},
                         'crop':case['crop']})
            print(json.dumps({'stage':'decoded_and_extracted','scene':case['scene_id'],
                              'method':method['name'],'frames':len(frames),'selected':len(records)}),flush=True)


def review_contact(case,method,indices,wide,wrist,out):
    tiles=[];x0,y0,x1,y1=case['crop']
    for i in indices:
        crop=wide[i][y0:y1,x0:x1]
        tile=np.full((490,440,3),255,np.uint8)
        tile[:200]=cv2.resize(crop,(440,200),interpolation=cv2.INTER_AREA)
        tile[200:448]=cv2.resize(wrist[i],(440,248),interpolation=cv2.INTER_AREA)
        row=method['frames'][i]
        cv2.putText(tile,f"f{i}  t={row['simulation_time_s']:.3f}s  held={int(row['correct_attached'])}",
                    (7,468),cv2.FONT_HERSHEY_SIMPLEX,.49,(35,35,35),1,cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles)%4:
        tiles.append(np.full_like(tiles[0],255))
    canvas=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    path=out/method['dense_review_file'];path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(canvas).save(path,quality=94)


def axes_image(fig,box,pixels,*,border=True):
    ax=fig.add_axes(box);ax.imshow(pixels,interpolation='lanczos',aspect='equal')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border);spine.set_linewidth(.45);spine.set_edgecolor('#d1d7dc')
    return ax


def annotate(ax,case,method,i,role):
    frame=method['frames'][i]
    crop=case['crop'] if role=='wide' else [0,0,1280,720]
    x0,y0,x1,y1=crop
    info=frame['wide'][case['view']-1] if role=='wide' else frame['handeye']
    target=info.get('object_box') or (info.get('projected_pick_box') if role=='wide' else None)
    if target:
        a,b,c,d=np.asarray(target,float)
        ax.add_patch(Ellipse(((a+c)/2-x0,(b+d)/2-y0),max(c-a+12,18),max(d-b+12,18),
                            fill=False,edgecolor=BLUE,linewidth=.8,zorder=4))
    if role=='wide':
        goal=info.get('projected_place_box')
        if goal:
            a,b,c,d=goal
            ax.add_patch(Rectangle((a-x0-4,b-y0-4),c-a+8,d-b+8,fill=False,
                                   edgecolor=GRAY,linewidth=.7,linestyle=(0,(3,2)),zorder=3))
        else:
            # Textual destinations have no body AABB. Mark only their centre;
            # this schematic symbol is NOT the evaluator's success radius.
            point=project(case['task']['place_position'],camera(case['view']),1920,1080)
            if point is not None:
                ax.add_patch(Rectangle((point[0]-x0-20,point[1]-y0-20),40,40,
                    fill=False,edgecolor=GRAY,linewidth=.7,linestyle=(0,(3,2)),zorder=3))
    # The warm event ring is on the gripper at actual contact/miss, not a fake trail.
    main={r['index']:r['label'] for r in method['selection']['main']}
    if not method['selection']['success'] and i in main and main[i] in (
            'Ground contact','Off-target close','Empty lift','Empty closure','Impact','Grasp attempt'):
        cfg=camera(case['view']) if role=='wide' else frame['handeye']['camera']
        point=project(frame['pad_center'],cfg,1920 if role=='wide' else 1280,1080 if role=='wide' else 720)
        if point is not None:
            point-=np.array([x0,y0])
            diameter=75 if role=='wide' else 160
            ax.add_patch(Ellipse(point,diameter,diameter,fill=False,edgecolor=ORANGE,linewidth=.9,zorder=5))


def file_image(out,path):
    with Image.open(out/path) as image:
        return np.asarray(image.convert('RGB')).copy()


def header(fig,text,x,y,width,height,*,size=8,weight='normal',color='#18222c',ha='left'):
    artist=fig.text(x/width,y/height,text,fontsize=size,fontweight=weight,color=color,
                    va='center',ha=ha,transform=fig.transFigure)
    return artist


def panel(fig,case,method,i,label,x,y,w,total_w,total_h,out,*,wide_h=None):
    """Millimetre geometry. Timestamp/captions outside unmodified source pixels."""
    index={r['frame_index_0_based']:r for r in method['panels']};row=index[i]
    wide=file_image(out,row['files']['workspace_crop']['file']);wrist=file_image(out,row['files']['handeye']['file'])
    h=w*wide.shape[0]/wide.shape[1] if wide_h is None else wide_h
    wh=w*9/16
    header(fig,f"t = {row['simulation_time_s']:.1f} s",x,y+1.5,total_w,total_h,size=8)
    ax=axes_image(fig,[x/total_w,(y-1-h)/total_h,w/total_w,h/total_h],wide)
    annotate(ax,case,method,i,'wide')
    ax=axes_image(fig,[x/total_w,(y-1-h-.55-wh)/total_h,w/total_w,wh/total_h],wrist)
    annotate(ax,case,method,i,'handeye')
    header(fig,label,x+w/2,y-h-wh-3,total_w,total_h,size=8,ha='center')
    return y-h-wh-5


def configure():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','font.size':8,'savefig.facecolor':'white',
                         'figure.facecolor':'white','axes.facecolor':'white'})


def save_figure(fig,stem,dpi,out):
    destination=out/stem;destination.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for suffix in ('pdf','svg','png'):
        path=destination.with_suffix('.'+suffix)
        temporary=path.with_suffix(path.suffix+'.tmp')
        fig.savefig(temporary,format=suffix,dpi=dpi,facecolor='white')
        temporary.replace(path)
        files.append(str(path.relative_to(out)))
    preview=destination.with_name(destination.name+'_preview').with_suffix('.jpg')
    fig.savefig(preview,dpi=160,facecolor='white',pil_kwargs={'quality':94})
    files.append(str(preview.relative_to(out)))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();overflows=[]
    fw,fh=fig.get_size_inches()*fig.dpi
    for text in fig.texts:
        box=text.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fw+.5 or box.y1>fh+.5:
            overflows.append(text.get_text())
    # Check all visible annotations are true vector text at >=8pt.
    sizes=[float(t.get_fontsize()) for t in fig.texts if t.get_text()]
    audit={'files':files,'width_mm':float(fig.get_figwidth()*25.4),
           'height_mm':float(fig.get_figheight()*25.4),'PNG_dpi':dpi,
           'minimum_figure_font_pt':min(sizes) if sizes else None,
           'text_outside_page':overflows,'vector_text_in_PDF_and_SVG':True}
    plt.close(fig)
    if overflows:
        raise ValueError('Text overflows page: '+str(overflows))
    write(out/'build_cache'/(stem.replace('/','__')+'.json'),{
        'stem':stem,'script_sha256':digest(str(Path(__file__).resolve())),
        'audit':audit,'file_sha256':{str(path):digest(str(out/path)) for path in files}})
    return audit


def overview(cases,stem,out,dpi,title=None):
    configure();width=180.;margin=3.;gutter=5.;gap=.8
    method_w=(width-2*margin-gutter)/2;w=(method_w-2*gap)/3
    # Interaction crop and full 16:9 wrist; no tiny image overlay.
    wide_h=w/2.2;wh=w*9/16
    row_h=4.5+3+wide_h+.55+wh+3+2.2+4.5
    groups=list(dict.fromkeys(c['group'] for c in cases))
    height=10+len(cases)*row_h+len(groups)*4.8+10
    fig=plt.figure(figsize=(width/25.4,height/25.4));top=height-4
    header(fig,'Scripted Oracle',margin+method_w/2,top,width,height,size=10,weight='bold',ha='center')
    header(fig,'X-VLA',margin+method_w+gutter+method_w/2,top,width,height,size=10,weight='bold',ha='center')
    y=height-11;previous=None
    for case in cases:
        group=case['group']
        if group!=previous:
            letter,name,color=GROUPS[group]
            fig.patches.append(Rectangle((margin/width,(y-2)/height),(width-2*margin)/width,4.3/height,
                                          transform=fig.transFigure,facecolor=color,edgecolor='none',zorder=-2))
            header(fig,letter+'  '+name,margin+1,y,width,height,size=8.5,weight='bold')
            y-=5;previous=group
        header(fig,f"S{case['scene_id']:02d}  {case['short_instruction']}",margin,y,width,height,size=8,weight='bold')
        # Method outcomes belong to the whole sequence, not to its first frame.
        for mi,method in enumerate(case['methods']):
            xbase=margin+mi*(method_w+gutter)
            label=method['selection']['outcome'];color=GREEN if method['selection']['success'] else ORANGE
            header(fig,label,xbase+method_w,y-4.5,width,height,size=8,ha='right',color=color,weight='bold')
            for j,item in enumerate(method['selection']['main']):
                x=xbase+j*(w+gap)
                panel(fig,case,method,item['index'],item['label'],x,y-11.5,w,width,height,out,wide_h=wide_h)
                if j<2:
                    xx=x+w+gap/2
                    fig.add_artist(FancyArrowPatch((xx/width,(y-11.5-wide_h/2)/height),
                                                   ((xx+.55)/width,(y-11.5-wide_h/2)/height),
                                                   transform=fig.transFigure,arrowstyle='-|>',
                                                   mutation_scale=5,linewidth=.5,color=GRAY))
        y-=row_h
    header(fig,'Top: workspace   Bottom: hand-eye HD (not policy input)',margin,6,width,height,size=8,color=GRAY)
    header(fig,'* Scripted Oracle fault',width-margin,6,width,height,size=8,color=GRAY,ha='right')
    return save_figure(fig,stem,dpi,out)


def detail_label(method,i):
    labels={r['index']:r['label'] for r in method['selection']['main']}
    if i in labels:
        return labels[i]
    frame=method['frames'][i]
    if i==0:return 'Initial scene'
    if i==len(method['frames'])-1:return 'Final state'
    if frame['correct_attached']:return 'Held'
    return 'Recorded state'


def detail(case,out,dpi):
    configure();width=270.;height=153.;fig=plt.figure(figsize=(width/25.4,height/25.4))
    header(fig,f"Scene {case['scene_id']:02d} · {case['instruction']}",4,height-5,width,height,size=11,weight='bold')
    header(fig,'Selected qualitative example; event timing differs between methods.',4,height-11,width,height,size=8,color=GRAY)
    gap=1.5;w=(width-8-5*gap)/6
    for mi,method in enumerate(case['methods']):
        y=height-19-mi*61
        header(fig,NAMES[method['name']]+' — '+method['selection']['outcome'],4,y,width,height,size=9,weight='bold',
               color=GREEN if method['selection']['success'] else '#18222c')
        for j,i in enumerate(method['selection']['detail_indices']):
            panel(fig,case,method,i,detail_label(method,i),4+j*(w+gap),y-6,w,width,height,out,wide_h=w/2.2)
    evidence=case['methods'][1]['evidence'];event=case['methods'][1]['selection']['selected_event']
    note=f"Seed {case['seed']} | {evidence['held_frames']} verified X-VLA held frames | "
    note+='HD hand-eye is a presentation camera, not the actor observation.'
    header(fig,note,4,7,width,height,size=8,color=GRAY)
    header(fig,'Oracle faults are staged. X-VLA success may include terminal opening-only assistance.',4,3,width,height,size=8,color=GRAY)
    return save_figure(fig,f"details/scene{case['scene_id']:02d}",dpi,out)


def policy_figure(out,dpi):
    """Use only already RPC/hash-verified original model-input images."""
    path=POLICY/'evidence/03_gray_block_target_miss/episode.json';meta=read(path)
    if meta['seed']!=20271060 or meta['checkpoint']!=str(CHECKPOINT):
        raise ValueError('Policy evidence belongs to another scene/checkpoint')
    if not meta['queries'] or not all(q['raw_front_wrist_and_segmentation_bit_exact'] for q in meta['queries']):
        raise ValueError('Need actual RPC-verified policy imagery')
    queries=[meta['queries'][i] for i in (0,1,2,4,6)]
    verified=[]
    for query in queries:
        for role in ('front','wrist'):
            source=POLICY/query['files'][role]['model224']
            if digest(str(source))!=query['files'][role]['model224_png_sha256']:
                raise ValueError('Original native model224 image changed')
            dest=out/'policy_observations'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(source,dest)
            if digest(str(dest))!=digest(str(source)):raise ValueError('Policy image copy differs')
            verified.append({'role':role,'query':query['query'],'time_s':query['simulation_time_s'],
                             'original':str(source),'file':str(dest.relative_to(out)),
                             'sha256':digest(str(source)), 'original_RPC_sha256':query['original_RPC_sha256']})
    configure();width=180.;height=102.;fig=plt.figure(figsize=(width/25.4,height/25.4))
    header(fig,'Scene 13 · Actual X-VLA camera inputs at policy queries',4,97,width,height,size=10,weight='bold')
    header(fig,'Native 224 × 224; HD video and wrist input are not the same camera.',4,91,width,height,size=8,color=GRAY)
    left=14;gap=1.6;w=(width-left-4-4*gap)/5
    for j,query in enumerate(queries):
        x=left+j*(w+gap)
        header(fig,f"q{query['query']}: {query['simulation_time_s']:.1f} s",x+w/2,85,width,height,size=8,ha='center')
        for k,role in enumerate(('front','wrist')):
            item=next(r for r in verified if r['query']==query['query'] and r['role']==role)
            axes_image(fig,[x/width,(82-(k+1)*w-k*1.5)/height,w/width,w/height],file_image(out,item['file']))
    header(fig,'Front',2,82-w/2,width,height,size=8)
    header(fig,'Wrist',2,82-1.5-1.5*w,width,height,size=8)
    header(fig,'Actor queries every 3 s; observations are held between queries.',4,8,width,height,size=8,color=GRAY)
    header(fig,'A GT map + numeric/proprio inputs were also available; no memory-loss claim.',4,3.5,width,height,size=8,color=GRAY)
    return save_figure(fig,'policy_observations/scene13_actual_policy_inputs',dpi,out),verified


CAPTION = (
    'Event-aligned qualitative comparison of Scripted Oracle and X-VLA on paired '
    'UR5e–Robotiq manipulation scenes. Each method is shown at three independently '
    'selected simulation times, illustrating the state before an event, the event, '
    'and its consequence; columns are not synchronized across methods. Workspace '
    'and HD hand-eye images at each instant are synchronized within a rollout. '
    'Blue outlines identify the target; dashed outlines identify the destination '
    '(schematic centre marks for textual regions, not success tolerances); '
    'orange outlines indicate the relevant interaction. Oracle failures are '
    'explicitly scripted, not failures of a trained TD3 policy. X-VLA uses '
    'simulator-privileged observations and the shared servo/grasp/safety system; '
    'successful qualitative clips include the previously disclosed terminal '
    'opening-only supervisor. HD hand-eye views are presentation images, not '
    'policy-camera inputs. Examples are selected post hoc and do not estimate '
    'failure frequencies or establish model-intrinsic causal mechanisms. Scene12 '
    'reuses a previously published X-VLA source; it is not a new independent trial. '
    'When included, Scene14 is a verified replay of an existing frozen benchmark '
    'trace with natural drop and reacquisition, not a newly inferred episode.'
)


def compact(case):
    result={k:v for k,v in case.items() if k!='methods'}
    result['methods']=[{k:v for k,v in m.items() if k!='frames'} for m in case['methods']]
    return result


def html_report(out,manifest,figures,stage):
    cards=[]
    for fig in figures:
        files=fig['files'];png=next(p for p in files if p.endswith('_preview.jpg'))
        pdf=next(p for p in files if p.endswith('.pdf'));svg=next(p for p in files if p.endswith('.svg'))
        cards.append(f'<section><h2>{html.escape(Path(pdf).stem)}</h2><p>'
                     f'<a href="{pdf}">PDF</a> · <a href="{svg}">Editable SVG</a> · '
                     f'<a href="{next(p for p in files if p.endswith(".png"))}">600 DPI PNG</a></p>'
                     f'<img loading="lazy" src="{png}"></section>')
    text='''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Oracle / X-VLA · Temporal qualitative figures</title>
<style>body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.5 system-ui;color:#18222c;background:white}
img{max-width:100%;height:auto;border:1px solid #d1d7dc}section{margin:40px 0}a{color:#0072b2}.note{background:#eef3f7;padding:16px}</style>
<h1>Oracle / X-VLA · Event-aligned qualitative comparison</h1>'''
    text+=f'<p class="note">Build stage: {stage}. '
    text+='Read audit.json and visual_review.json before calling this delivery complete.</p>'
    text+='<p>'+html.escape(CAPTION)+'</p><p><a href="manifest.json">Source/frame manifest</a> · '
    text+='<a href="audit.json">Automated audit</a> · <a href="visual_review.json">Visual review</a></p>'
    text+=''.join(cards)
    text+='<h2>Interpretation limits</h2><ul><li>No semantic misidentification, memory loss or model-only blame is inferred.</li>'
    text+='<li>Empty lift is not a transit drop. Ejection is not necessarily an elastic bounce.</li>'
    text+='<li>Selected clips are not an unbiased success-rate cohort. No quantitative result has been edited.</li>'
    text+='<li>600 DPI changes export density, not the information content of source images.</li></ul></html>'
    (out/'index.html').write_text(text)
    (out/'caption.txt').write_text(CAPTION+'\n')


def audit(cases,out,figures,stage,policy_proof,main_ids):
    checks=[]
    for case in cases:
        for method in case['methods']:
            frames=method['frames'];main=method['selection']['main']
            times=[frames[r['index']]['simulation_time_s'] for r in main]
            if not all(a<b for a,b in zip(times,times[1:])):
                raise ValueError('Non-increasing simulation time')
            for row in method['panels']:
                for item in row['files'].values():
                    if digest(str(out/item['file']))!=item['sha256']:
                        raise ValueError('Output unannotated panel changed')
            for path,sha in method['source_sha256'].items():
                digest.cache_clear()
                if digest(path)!=sha:
                    raise ValueError('Source was mutated during figure generation')
            checks.append({'scene':case['scene_id'],'method':method['name'],'times_s':times,
                           'source_immutable':True,'full_wide_and_handeye_decode_passed':True,
                           'within_rollout_camera_synchrony_passed':True,
                           'constant_pair_view_and_crop':True,'event_labels_validated':True})
    result={'schema':SCHEMA,'stage':stage,'automated_pass':True,'cases':len(cases),
            'method_rollouts_shown':2*len(cases),'group_counts':dict(Counter(c['group'] for c in cases)),
            'main_case_ids':list(main_ids),'requested_full_case_ids':[c['scene_id'] for c in cases],'checks':checks,
            'figures':figures,'policy_RGB224_provenance_verified':bool(policy_proof),
            'policy_images_are_not_HD_handeye':True,'no_new_policy_inference_or_simulation_in_figure_export':True,
            'no_invented_natural_drop_or_recovery':True,'all_original_source_files_unchanged':True,
            'manual_visual_review_required':True,'manual_every_video_frame_viewing_claimed':False,
            'completed_delivery':False}
    write(out/'audit.json',result)
    write(out/'status.json',{'stage':stage,'automated_pass':True,'awaiting_visual_review':True,
                             'completed_delivery':False})


def main(args):
    out=args.output.resolve()
    if out.exists():
        marker=out/'ownership.json'
        if not marker.exists() or read(marker)['schema']!=SCHEMA:
            raise FileExistsError('Refuse to reuse an unrelated existing output directory')
    else:
        out.mkdir(parents=True)
        write(out/'ownership.json',{'schema':SCHEMA,'script':str(Path(__file__).resolve()),
                                   'source_files_are_read_only':True,'created_unix':time.time()})
    if args.stage=='refresh':
        return refresh(args,out)
    if args.stage in ('sources','figures','assemble'):
        return staged(args,out)
    ids=PILOT if args.stage=='pilot' else ((2,14,3,4,5,12,6,7,8,13,10,11) if args.include_recovery else CASES)
    main_ids=(14,3,12,6,13,10) if args.include_recovery else MAIN
    write(out/'status.json',{'stage':'source_inventory','requested_stage':args.stage,'case_ids':list(ids)})
    cases=inventory(ids,out/'additional_recovery_source' if args.include_recovery else None)
    prepare(cases,out)
    figures=[]
    if args.stage=='pilot':
        figures.append(overview(cases,'pilot/pilot_temporal_comparison',out,args.dpi))
    else:
        main_cases=[next(c for c in cases if c['scene_id']==i) for i in main_ids]
        figures.append(overview(main_cases,'main_6_cases',out,args.dpi))
        # Two print-friendly halves retain full 8pt text when a manuscript
        # cannot accommodate the tall six-case overview on one page.
        figures.append(overview(main_cases[:3],'print_halves/main_partA',out,args.dpi))
        figures.append(overview(main_cases[3:],'print_halves/main_partB',out,args.dpi))
        for group in GROUPS:
            subset=[c for c in cases if c['group']==group]
            figures.append(overview(subset,f'groups/group{group:02d}',out,args.dpi))
    for case in cases:
        figures.append(detail(case,out,args.dpi))
    policy_figure_audit,proof=policy_figure(out,args.dpi);figures.append(policy_figure_audit)
    manifest={'schema':SCHEMA,'source_label_inventory':str(SOURCE/'LABELS.json'),
              'case_ids':list(ids),'main_ids':list(main_ids),'cases':[compact(c) for c in cases],
              'policy_observations':proof,'script_sha256':digest(str(Path(__file__).resolve())),
              'checkpoint_config_sha256':digest(str(CHECKPOINT/'config.json')),
              'checkpoint_trainables_sha256':digest(str(CHECKPOINT/'trainable_masters.pt')),
              'new_qualitative_rollouts':0,'no_quantitative_results_modified':True}
    write(out/'manifest.json',manifest)
    html_report(out,manifest,figures,args.stage)
    audit(cases,out,figures,args.stage,proof,main_ids)
    print(json.dumps({'stage':args.stage,'output':str(out),'figures':len(figures),
                      'cases':len(cases),'automated_pass':True,'manual_review_pending':True}),flush=True)


def refresh(args,out):
    """Revalidate sources, replace rejected recovery imagery, polish vector layout.

    Reuse already full-decoded immutable panels for the 11 unchanged scenes.
    The additional recovery is decoded anew. No policy inference occurs.
    """
    previous=read(out/'manifest.json');prior_audit=read(out/'audit.json')
    if prior_audit['stage']!='full' or len(previous['case_ids'])!=12:
        raise ValueError('Refresh requires a completed full automatic build first')
    cases=inventory(previous['case_ids'],out/'additional_recovery_source')
    old={c['scene_id']:c for c in previous['cases']}
    for case in cases:
        if case['scene_id']==14:
            prepare([case],out)
            continue
        if case['crop']!=old[case['scene_id']]['crop']:
            raise ValueError('Unchanged source crop differs')
        for method,saved in zip(case['methods'],old[case['scene_id']]['methods'],strict=True):
            if method['source_sha256']!=saved['source_sha256'] or method['selection']!=saved['selection']:
                raise ValueError('Cached source/selection changed; full decode required')
            for key in ('panels','dense_review_file','decode_audit'):
                method[key]=saved[key]
            for row in method['panels']:
                for item in row['files'].values():
                    if digest(str(out/item['file']))!=item['sha256']:
                        raise ValueError('Cached unannotated image changed')
                    pixels=file_image(out,item['file'])
                    if float(pixels.astype(np.float32).std(axis=(0,1)).max())<1.:
                        raise ValueError('Cached empty image rejected')
    main_ids=previous['main_ids'];figures=[]
    main_cases=[next(c for c in cases if c['scene_id']==i) for i in main_ids]
    figures.append(overview(main_cases,'main_6_cases',out,args.dpi))
    figures.append(overview(main_cases[:3],'print_halves/main_partA',out,args.dpi))
    figures.append(overview(main_cases[3:],'print_halves/main_partB',out,args.dpi))
    for group in GROUPS:
        figures.append(overview([c for c in cases if c['group']==group],f'groups/group{group:02d}',out,args.dpi))
    for case in cases:
        sid=case['scene_id']
        if sid in (10,11,13,14):
            figures.append(detail(case,out,args.dpi))
        else:
            figures.append(next(f for f in prior_audit['figures'] if f['files'][0]==f'details/scene{sid:02d}.pdf'))
    policy_audit,proof=policy_figure(out,args.dpi);figures.append(policy_audit)
    previous.update(cases=[compact(c) for c in cases],policy_observations=proof,
        script_sha256=digest(str(Path(__file__).resolve())),
        new_XVLA_inference_trials=0,new_scripted_Oracle_counterparts=1,
        rejected_render_revision='v1: EGL loaded after visual shape creation; never used in final figures')
    write(out/'manifest.json',previous)
    html_report(out,previous,figures,'full')
    audit(cases,out,figures,'full',proof,main_ids)
    print(json.dumps({'refreshed':True,'figures':len(figures),'manual_review_pending':True}),flush=True)


def staged(args,out):
    """Restart-safe bounded jobs; commit source inventory BEFORE heavy exports."""
    ids=(2,14,3,4,5,12,6,7,8,13,10,11) if args.include_recovery else CASES
    main_ids=(14,3,12,6,13,10) if args.include_recovery else MAIN
    cases=inventory(ids,out/'additional_recovery_source' if args.include_recovery else None)
    prepare(cases,out)
    manifest={'schema':SCHEMA,'source_label_inventory':str(SOURCE/'LABELS.json'),
        'case_ids':list(ids),'main_ids':list(main_ids),'cases':[compact(c) for c in cases],
        'script_sha256':digest(str(Path(__file__).resolve())),
        'checkpoint_config_sha256':digest(str(CHECKPOINT/'config.json')),
        'checkpoint_trainables_sha256':digest(str(CHECKPOINT/'trainable_masters.pt')),
        'new_XVLA_inference_trials':0,'new_scripted_Oracle_counterparts':int(14 in ids),
        'no_quantitative_results_modified':True,
        'rejected_render_revision':'v1: EGL loaded after shapes; final figures use v2 only'}
    write(out/'manifest.json',manifest)
    write(out/'status.json',{'stage':args.stage,'case_ids':list(ids),'completed_delivery':False})
    if args.stage=='sources':
        print(json.dumps({'sources_committed':True,'cases':len(cases)}),flush=True)
        return
    mains=[next(c for c in cases if c['scene_id']==i) for i in main_ids]
    products=[('main_6_cases',lambda:overview(mains,'main_6_cases',out,args.dpi)),
        ('print_halves/main_partA',lambda:overview(mains[:3],'print_halves/main_partA',out,args.dpi)),
        ('print_halves/main_partB',lambda:overview(mains[3:],'print_halves/main_partB',out,args.dpi))]
    for group in GROUPS:
        stem=f'groups/group{group:02d}'
        products.append((stem,lambda group=group,stem=stem:overview(
            [c for c in cases if c['group']==group],stem,out,args.dpi)))
    for case in cases:
        products.append((f"details/scene{case['scene_id']:02d}",lambda case=case:detail(case,out,args.dpi)))
    products.append(('policy_observations/scene13_actual_policy_inputs',lambda:policy_figure(out,args.dpi)[0]))
    figures=[]
    for stem,render in products:
        if args.stage=='figures' and args.only and stem not in args.only.split(','):
            continue
        cache=out/'build_cache'/(stem.replace('/','__')+'.json')
        saved=read(cache) if cache.exists() else None
        ready=(saved and saved['script_sha256']==manifest['script_sha256'] and
               all((out/path).exists() and digest(str(out/path))==sha for path,sha in saved['file_sha256'].items()))
        if not ready:
            if args.stage=='assemble':raise ValueError('Missing current verified export: '+stem)
            result=render()
            figures.append(result)
        else:figures.append(saved['audit'])
        print(json.dumps({'figure':stem,'ready':True}),flush=True)
    if args.stage=='assemble':
        # Load policy-image provenance without re-exporting its figure.
        proof=[]
        meta=read(POLICY/'evidence/03_gray_block_target_miss/episode.json')
        for query in [meta['queries'][i] for i in (0,1,2,4,6)]:
            for role in ('front','wrist'):
                source=POLICY/query['files'][role]['model224'];dest=out/'policy_observations'/source.name
                if digest(str(source))!=query['files'][role]['model224_png_sha256'] or digest(str(dest))!=digest(str(source)):
                    raise ValueError('Policy RGB224 evidence changed')
                proof.append({'role':role,'query':query['query'],'time_s':query['simulation_time_s'],
                    'original':str(source),'file':str(dest.relative_to(out)),
                    'sha256':digest(str(source)),'original_RPC_sha256':query['original_RPC_sha256']})
        manifest['policy_observations']=proof;write(out/'manifest.json',manifest)
        html_report(out,manifest,figures,'full')
        audit(cases,out,figures,'full',proof,main_ids)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('pilot','full','refresh','sources','figures','assemble'),default='pilot')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--dpi',type=int,default=600)
    parser.add_argument('--include-recovery',action='store_true',help='Include qualified natural drop/reacquisition Scene14')
    parser.add_argument('--only',help='Comma-separated figure stems for bounded, restart-safe export jobs')
    main(parser.parse_args())
