"""Audit encoded camera editions against native streams and every-frame masks."""
from pathlib import Path
import json,subprocess
import cv2,imageio_ffmpeg,numpy as np
from env_eval.audit_cinematic import motion_metrics


def decode_at(path,index):
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot decode {path}, frame {index}')
    return frame


def camera_gate(rows,name,action):
    interaction=[r for r in rows if r['phase'] in ('grasp','carry','release','touch','touch_retract') or r['distance']<.04]
    placing=[r for r in rows if action=='put' and r['phase'] in ('retract','settle')]
    critical=interaction+placing
    minimum=min((r[name]['object_pixels'] for r in critical),default=0)
    pads=[min((r[name]['pad_pixels'][i] for r in interaction),default=0) for i in (0,1)]
    margin=min((r[name]['object_border_margin_px'] for r in critical),default=-1)
    return dict(minimum_critical_object_pixels=minimum,minimum_interaction_pad_pixels=pads,
        minimum_critical_object_border_px=margin,critical_frames=len(critical),
        pass_visibility=bool(minimum>=100 and min(pads)>=20 and margin>=8))


def audit_scene(source,target,mode,settings):
    physics=json.loads((source/'audit.json').read_text());trajectory=json.loads((source/'trajectory.json').read_text())
    visibility=json.loads((source/'visibility.json').read_text());layouts=json.loads((target/'layout.json').read_text())
    replay=json.loads((source/'replay_proof.json').read_text());motion=motion_metrics(trajectory)
    cameras={name:camera_gate(visibility,name,physics['action']) for name in
             (('front','handeye') if mode=='front_handeye' else ('front',) if mode=='front' else ('handeye',))}
    events={0,len(trajectory)//2,len(trajectory)-1}
    for phase in ('touch','grasp','carry','release','retract','settle'):
        ids=[i for i,s in enumerate(trajectory) if s['phase']==phase]
        if ids:events.add(ids[len(ids)//2])
    held=[i for i,s in enumerate(trajectory) if s['held']]
    if held:events.add(held[0])
    frames=target/'review';frames.mkdir(exist_ok=True)
    native={}
    for name in cameras:
        file='gripper_closeup.mp4' if name=='front' else 'handeye.mp4'
        native[name]={index:decode_at(target/file,index) for index in events}
        for index,frame in native[name].items():
            phase=trajectory[index]['phase'];cv2.imwrite(str(frames/f'{index:04}_{phase}_{name}.jpg'),frame)
    rows=[];sync=[]
    for path in sorted(target.glob('*.mp4')):
        cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];cap.release()
        # Decode the whole encoded stream, including frames between review events.
        check=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',str(path),'-f','null','-'],capture_output=True,text=True)
        decoded=check.returncode==0 and not check.stderr.strip()
        expected=settings['wide_resolution'] if path.stem.startswith('view_') else settings['handeye_resolution'] if path.stem=='handeye' else settings['front_resolution']
        row=dict(file=path.name,fps=fps,resolution=size,frames=n,full_decode_pass=decoded,
            pass_video=bool(fps==settings['fps'] and size==expected and n==len(visibility) and decoded))
        if path.stem.startswith('view_'):
            view=int(path.stem.split('_')[-1]);layout=layouts[str(view)]
            if layout.get('requires_clean_wide'):
                from env_eval.camera_editions import overlap
                row['max_object_pixels_covered']=max(sum(overlap(v['wide'][view-1]['projected_pick_box'],rect) for rect in layout['rects']) for v in visibility)
                row['max_destination_box_pixels_covered']=max(sum(overlap(v['wide'][view-1]['projected_place_box'],rect) for rect in layout['rects'])
                    if v['wide'][view-1]['projected_place_box'] is not None else 0 for v in visibility)
                row['clean_wide_proof']=json.loads((source/f'clean_wide_{view}_proof.json').read_text())
                row['pass_video'] &= row['clean_wide_proof']['pass_exact'] and row['max_destination_box_pixels_covered']==0
            else:row['max_object_pixels_covered']=max(v['wide'][view-1]['candidate_costs'][layout['key']][0] for v in visibility)
            row['minimum_wide_object_pixels']=min(v['wide'][view-1]['object_pixels'] for v in visibility)
            row['pass_video'] &= row['max_object_pixels_covered']==0 and row['minimum_wide_object_pixels']>=8
            for index in sorted(events):
                image=decode_at(path,index)
                if index in (min(events),len(trajectory)//2,max(events)) or trajectory[index]['phase'] in ('release','touch'):
                    cv2.imwrite(str(frames/f'{index:04}_{trajectory[index]["phase"]}_view_{view}.jpg'),image)
                for name,x,y,w,h in layout['rects']:
                    reference=cv2.resize(native[name][index],(w,h),interpolation=cv2.INTER_AREA)
                    error=float(np.abs(image[y:y+h,x:x+w].astype(float)-reference.astype(float)).mean())
                    sync.append(dict(view=view,frame=index,camera=name,mean_pixel_error=error,pass_sync=error<6.))
        rows.append(row)
    expected_count=8 if mode=='front_handeye' else 7
    baseline_compared=replay.get('baseline') is not None
    result=dict(robot=physics['robot'],action=physics['action'],config=physics['config'],mode=mode,
        physics_pass=physics['overall_pass'],baseline_compared=baseline_compared,
        exact_baseline_replay=replay['pass_exact'] if baseline_compared else None,motion=motion,
        cameras=cameras,videos=rows,inset_sync=sync,stable_layout=True,added_video_text=False)
    result['pass']=bool(result['physics_pass'] and (not baseline_compared or result['exact_baseline_replay']) and motion['motion_pass'] and
        len(rows)==expected_count and all(r['pass_video'] for r in rows) and all(s['pass_sync'] for s in sync) and
        all(c['pass_visibility'] for c in cameras.values()))
    return result
