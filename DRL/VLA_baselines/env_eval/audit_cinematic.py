"""Independent review of encoded, synchronized, text-free Oracle/PiP videos."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from utils.camera import CINEMATIC_VIEWS


def motion_metrics(samples):
    t=np.array([s['time'] for s in samples]);xyz=np.array([s['tcp'] for s in samples])
    q=np.array([s['joints'] for s in samples]);dt=float(np.median(np.diff(t)))
    velocity=np.diff(xyz,axis=0)/dt
    step=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    acceleration=np.diff(np.diff(q,axis=0)/dt,axis=0)/dt
    carry=[s for s in samples if s['phase']=='carry']
    transport={}
    if len(carry)>15:
        points=np.array([s['tcp'] for s in carry])
        window=min(11,(len(points)//2)*2-1)
        filtered=savgol_filter(points,window,3,axis=0)
        speed=np.linalg.norm(np.diff(filtered,axis=0),axis=1)/dt
        centre=speed[int(len(speed)*.2):int(len(speed)*.8)]
        rots=Rotation.from_quat([s['mount_quaternion'] for s in carry])
        transport={'carry_internal_stall_fraction':float(np.mean(centre<.008)),
                   'carry_wrist_excursion_degrees':float(np.degrees((rots[0].inv()*rots).magnitude()).max()),
                   'carry_peak_height_m':float(points[:,2].max()),'carry_duration_seconds':carry[-1]['time']-carry[0]['time']}
    alignment=sum(s['phase']=='align' for s in samples)*dt
    place_wait=sum(s['phase'] in ('place','hover') for s in samples)*dt
    grasp=sum(s['phase']=='grasp' for s in samples)*dt
    quiet=[s for s in samples if s['time']>=samples[-1]['time']-.25]
    mount_xyz=np.array([s['relative_mount_position'] for s in quiet])
    mount_rot=Rotation.from_quat([s['relative_mount_quaternion'] for s in quiet])
    quiet_translation=float(np.linalg.norm(mount_xyz-mount_xyz[0],axis=1).max())
    quiet_rotation=float(np.degrees((mount_rot[0].inv()*mount_rot).magnitude()).max())
    return {'max_tcp_frame_step_m':float(step.max()),'max_tcp_speed_m_s':float(np.linalg.norm(velocity,axis=1).max()),
            'max_joint_frame_acceleration_rad_s2':float(abs(acceleration).max()),
            'alignment_pause_seconds':alignment,'pre_release_pause_seconds':place_wait,
            'stationary_grasp_phase_seconds':grasp,
            'final_mount_translation_range_m':quiet_translation,
            'final_mount_rotation_range_deg':quiet_rotation,
            'motion_pass':bool(step.max()<.015 and abs(acceleration).max()<12 and
                transport.get('carry_internal_stall_fraction',0)<.08),**transport}


def audit_scene(directory):
    physics=json.loads((directory/'audit.json').read_text())
    samples=json.loads((directory/'trajectory.json').read_text())
    visibility=json.loads((directory/'visibility.json').read_text())
    metrics=motion_metrics(samples)
    interaction=[v for v in visibility if v['view']=='gripper_closeup' and
                 (v['phase'] in ('grasp','carry','release','touch','touch_retract') or v['tcp_object_distance']<.04)]
    post_release=[v for v in visibility if v['view']=='gripper_closeup' and
                  physics['action']=='put' and v['phase'] in ('retract','settle')]
    critical=interaction+post_release
    macro={'minimum_interaction_object_pixels':min((v['visible_pick_pixels'] for v in critical),default=0),
           'minimum_interaction_pad_pixels':[min((v['pad_pixels'][i] for v in interaction),default=0) for i in (0,1)]}
    macro['pass']=bool(macro['minimum_interaction_object_pixels']>=100 and min(macro['minimum_interaction_pad_pixels'])>=20)
    if physics['action']=='put':
        macro['minimum_object_border_margin_px']=min((v.get('object_border_margin_px',-1) for v in critical),default=-1)
        macro['minimum_post_release_object_pixels']=min((v['visible_pick_pixels'] for v in post_release),default=0)
        macro['pass'] &= macro['minimum_object_border_margin_px']>=8
    rows=[];timing=[]
    event_times={'initial':0.}
    for phase in dict.fromkeys(s['phase'] for s in samples):
        selected=[s for s in samples if s['phase']==phase]
        event_times[phase]=.5*(selected[0]['time']+selected[-1]['time'])
    first_held=next((s['time'] for s in samples if s['held'] and s['bilateral']),None)
    if first_held is not None:event_times['verified_grasp']=first_held
    if physics['action']=='touch':
        contact=next((s['time'] for s in samples if s.get('contact_force_n',sum(s['pad_forces']))>.02),None)
        if contact is not None:event_times['verified_touch']=contact
    if any(s['phase']=='carry' for s in samples):
        selected=[s for s in samples if s['phase']=='carry']
        event_times['carry_peak']=max(selected,key=lambda s:s['tcp'][2])['time']
    frames=directory/'audit_frames';frames.mkdir(exist_ok=True)
    for name,view in [(f'view_{i}',i) for i in range(1,len(CINEMATIC_VIEWS)+1)]+[('gripper_closeup','gripper_closeup')]:
        path=directory/f'{name}.mp4';cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration=count/fps if fps else 0
        valid=fps==30 and w>=(1280 if view=='gripper_closeup' else 1920) and h>=(720 if view=='gripper_closeup' else 1080)
        decoded=True
        for event,seconds in event_times.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES,min(count-1,max(0,round(seconds*fps))))
            ok,image=cap.read()
            decoded &= bool(ok and image.std()>5)
            if ok:cv2.imwrite(str(frames/f'{event}_{name}.jpg'),image)
        cap.release()
        time_error=abs(duration-physics['duration_seconds']);timing.append(count)
        selected=[v for v in visibility if v['view']==view]
        covered=max((v.get('pick_pixels_covered_by_inset',0) for v in selected),default=0)
        min_visible=min((v['visible_pick_pixels'] for v in selected),default=0)
        row={'file':path.name,'fps':fps,'resolution':[w,h],'frames':count,'duration_seconds':duration,
             'duration_error_seconds':time_error,'decode_pass':decoded,'max_main_pick_pixels_covered_by_inset':covered,
             'minimum_pick_pixels':min_visible,'pass':bool(valid and decoded and time_error<=1/30+.002 and
                 (macro['pass'] if view=='gripper_closeup' else min_visible>=8 and covered==0))}
        rows.append(row)
    # Compare the embedded macro against the same numbered native macro frame.
    macro_cap=cv2.VideoCapture(str(directory/'gripper_closeup.mp4'));sync=[]
    seconds=event_times.get('verified_grasp',event_times.get('verified_touch',event_times.get('touch',event_times.get('carry',0))))
    frame_id=round(seconds*30);macro_cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,macro_frame=macro_cap.read();macro_cap.release()
    if ok:
        for i in range(1,len(CINEMATIC_VIEWS)+1):
            meta=min((v for v in visibility if v['view']==i),key=lambda v:abs(v['time']-seconds))
            cap=cv2.VideoCapture(str(directory/f'view_{i}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,film=cap.read();cap.release()
            if ok:
                x,y,w,h=meta['inset_rect'];target=cv2.resize(macro_frame,(w,h),interpolation=cv2.INTER_AREA)
                error=float(np.abs(film[y:y+h,x:x+w].astype(float)-target.astype(float)).mean())
                sync.append({'view':i,'mean_pixel_error':error,'pass':error<6.})
    sync_pass=len(set(timing))==1 and len(sync)==len(CINEMATIC_VIEWS) and all(s['pass'] for s in sync)
    # A separate articulated body uses a numerical fixed constraint, not a
    # merged URDF. Require sub-visible error during contact and wrist motion.
    # UR keeps its original numerical attachment/dynamics. Allow at most
    # ~3 close-up pixels of transient compliance; resting motion is bounded
    # separately below. The UF root attachment uses the tighter 1 mm bound.
    translation_bound=.001 if physics['robot']=='uf850' else .0012
    mount_pass=bool(physics['max_mount_translation_error_m']<translation_bound and physics['max_mount_rotation_error_deg']<1.5
                    and metrics['final_mount_translation_range_m']<.0002 and metrics['final_mount_rotation_range_deg']<.2)
    gentle_pass=physics['action']!='touch' or physics['max_touch_force_n']<5.
    result={'robot':physics['robot'],'action':physics['action'],'config':physics['config'],
            'physics_pass':physics['overall_pass'],'motion':metrics,'closeup':macro,'videos':rows,
            'inset_sync':sync,'inset_sync_pass':sync_pass,'mount_pass':mount_pass,'gentle_touch_pass':gentle_pass,
            'mount_limits':{'loaded_translation_m':translation_bound,'loaded_rotation_deg':1.5,
                            'final_translation_range_m':.0002,'final_rotation_range_deg':.2},
            'pass':bool(physics['overall_pass'] and metrics['motion_pass'] and macro['pass'] and sync_pass and mount_pass
                        and gentle_pass and all(r['pass'] for r in rows))}
    (directory/'presentation_audit.json').write_text(json.dumps(result,indent=2))
    return result


def review_sheets(root):
    review=root/'review';review.mkdir(exist_ok=True)
    for robot in ('UF850','UR5e_Robotiq2F'):
        for action in ('touch','pick','move','put'):
            tiles=[]
            for config in range(1,5):
                directory=root/robot/action/f'config{config}'/'audit_frames'
                if not directory.exists():continue
                event_list=['initial','verified_touch' if action=='touch' else 'verified_grasp',
                            'touch_retract' if action=='touch' else 'carry_peak',
                            'settle' if action=='put' else 'final']
                for event in event_list:
                    for view in tuple(f'view_{i}' for i in range(1,len(CINEMATIC_VIEWS)+1))+('gripper_closeup',):
                        path=directory/f'{event}_{view}.jpg'
                        if not path.exists():continue
                        image=Image.open(path).convert('RGB');image.thumbnail((400,225))
                        tile=Image.new('RGB',(400,246),'white');tile.paste(image,(0,0))
                        ImageDraw.Draw(tile).text((5,228),f'C{config} {event} {view}',fill='black');tiles.append(tile)
            if tiles:
                columns=len(CINEMATIC_VIEWS)+1
                canvas=Image.new('RGB',(400*columns,((len(tiles)+columns-1)//columns)*246),'white')
                for i,t in enumerate(tiles):canvas.paste(t,((i%columns)*400,(i//columns)*246))
                canvas.save(review/f'{robot}_{action}.jpg',quality=93)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    rows=[]
    for f in sorted(args.root.glob('*/*/*/audit.json')):
        stored=f.parent/'presentation_audit.json'
        result=json.loads(stored.read_text()) if args.resume and stored.exists() else audit_scene(f.parent)
        rows.append(result);print(result['robot'],result['action'],result['config'],result['pass'],flush=True)
    review_sheets(args.root)
    summary={'scenes':len(rows),'scenes_passed':sum(r['pass'] for r in rows),
             'videos':sum(len(r['videos']) for r in rows),'videos_passed':sum(v['pass'] for r in rows for v in r['videos']),
             'rows':rows}
    (args.root/'presentation_audit.json').write_text(json.dumps(summary,indent=2))
    print({k:v for k,v in summary.items() if k!='rows'})
    return 0 if len(rows)==32 and all(r['pass'] for r in rows) else 1

if __name__=='__main__':raise SystemExit(main())
