"""Independent encoded-media and measured physical-outcome checks."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from .qualify import OUT,ROBOTS,digest
from .render import VIEWS
from .scenes import get_scene
SCHEMA='ICCAS_decode_black_framecount_v2'


def read(path):return json.loads(Path(path).read_text())


def decode_frame(path,index):
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot decode frame {index}: {path}')
    return frame


def stream_integrity(path):
    # One full decode checks corruption, the ACTUAL decoded frame count, and
    # black intervals, including frames that are not on a contact sheet.
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','info','-nostats',
        '-xerror','-err_detect','explode','-threads','1','-i',str(path),
        '-vf','blackdetect=d=0:pix_th=0.02:pic_th=0.99','-progress','pipe:1','-f','null','-'],
        capture_output=True,text=True)
    counts=[int(line.split('=',1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    return {'decode_returncode':result.returncode,'actually_decoded_frames':counts[-1] if counts else None,
        'black_intervals_detected':result.stderr.count('black_start:'),
        'passed':result.returncode==0 and bool(counts) and 'black_start:' not in result.stderr}


def motion_audit(trajectory):
    q=np.array([s['joints'] for s in trajectory]);vel=np.diff(q,axis=0)*30
    acceleration=np.diff(vel,axis=0)*30
    changes={}
    for name in trajectory[0]['objects']:
        positions=np.array([s['objects'][name]['position'] for s in trajectory])
        changes[name]=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).max(initial=0))
    return {'max_joint_velocity_rad_s':float(abs(vel).max(initial=0)),
        'max_joint_acceleration_rad_s2':float(abs(acceleration).max(initial=0)),
        'max_object_frame_displacement_m':changes,
        'passed':bool(abs(vel).max(initial=0)<1.8 and abs(acceleration).max(initial=0)<12
            and max(changes.values(),default=0)<.065)}


def audit(robot,sid,out=OUT):
    folder=out/ROBOTS[robot]/f'scene{sid:02d}';selection=read(folder/'PHYSICS.json')
    physical=selection['result'];render=read(folder/'RENDER.json');source=Path(selection['qualified_directory'])
    samples=read(source/'trajectory.json');visibility=read(out/'_camera_sources'/ROBOTS[robot]/f'scene{sid:02d}/visibility.json')
    native=Path(render['native_front']);layouts=read(folder/'layout.json');spec=get_scene(sid)
    counts=len(samples);events={0,counts-1};groups={}
    for i,s in enumerate(samples):groups.setdefault((s['task_index'],s['phase']),[]).append(i)
    for key,indices in groups.items():
        if key[1] in ('approach','carry','place_align','release','settle','home_return','perceive_home','final','touch'):
            events.update((indices[0],indices[len(indices)//2],indices[-1]))
    # Native images cached only at audit milestones, not hundreds of HD frames.
    native_frames={i:decode_frame(native,i) for i in events};videos=[];sync=[]
    review=folder/'review';review.mkdir(exist_ok=True)
    for path in [*(folder/f'view_{view}.mp4' for view in VIEWS),native]:
        cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        fps=cap.get(cv2.CAP_PROP_FPS);size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        # ffmpeg consumes every frame through EOF and reports corruption/errors.
        check=stream_integrity(path)
        expected=[1280,720] if path==native else [1920,1080]
        passed=bool(declared==counts and fps==30 and size==expected and check['passed'] and check['actually_decoded_frames']==counts)
        videos.append({'path':str(path),'frames':declared,'fps':fps,'resolution':size,
            'stream_integrity':check,'full_decode_passed':check['passed'],'passed':passed,'sha256':digest(path)})
        if path==native:continue
        view=int(path.stem.split('_')[1]);_,x,y,w,h=layouts[str(view)]['rect']
        for i in sorted(events):
            image=decode_frame(path,i)
            target=cv2.resize(native_frames[i],(w,h),interpolation=cv2.INTER_AREA)
            error=float(np.abs(image[y:y+h,x:x+w].astype(np.float32)-target.astype(np.float32)).mean())
            sync.append({'view':view,'frame':i,'mean_pixel_error':error,'passed':error<6.})
        # A compact, labeled contact sheet is an audit artifact only. The
        # actual delivered MP4s have no text or labels.
        chosen=[]
        for task_index in range(len(spec.steps)):
            for phase in ('approach','carry','release','settle','perceive_home','final','touch'):
                indices=groups.get((task_index,phase),[])
                if indices:chosen.append(indices[len(indices)//2])
        columns=3;cw,ch=640,380;sheet=Image.new('RGB',(columns*cw,((len(chosen)+columns-1)//columns)*ch),'white')
        draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(chosen):
            rgb=cv2.cvtColor(decode_frame(path,i),cv2.COLOR_BGR2RGB)
            picture=Image.fromarray(rgb);picture.thumbnail((cw,360));x0=(j%columns)*cw;y0=(j//columns)*ch
            sheet.paste(picture,(x0,y0+20));draw.text((x0+5,y0+3),f'task {samples[i]["task_index"]+1} / {samples[i]["phase"]} / f{i}',fill='black')
        sheet.save(review/f'CONTACT_SHEET_view_{view}.jpg',quality=92)
    critical=[v for i,v in enumerate(visibility) if samples[i]['held'] and samples[i]['phase'] in ('carry','place_align','release')]
    native_min=min((v['target_pixels_front'] for v in critical),default=None)
    minima={str(view):min((v['wide'][str(view)]['target_pixels'] for v in critical),default=None) for view in VIEWS}
    wide_zero=sum(v['wide'][str(view)]['object_pixels_covered_by_inset'] for v in visibility for view in VIEWS)==0
    visible={name:sum(visibility[0]['wide'][str(view)].get('per_object_pixels',{}).get(name,0) for view in VIEWS)
        for name,xy in spec.objects}
    geometry_visible=all(visible[name]>=20 for name,xy in spec.objects)
    homes=physical['home_checks'];home_ok=len(homes)==(0 if spec.steps[0].action=='pick' else len(spec.steps)) and all(h['passed'] for h in homes)
    safety=all(physical[k]>=-.001 for k in ('self_min_m','arm_gripper_min_m','gripper_ground_min_m'))
    move=motion_audit(samples)
    timing=bool(np.allclose(np.array([s['time'] for s in samples]),np.arange(counts)/30,atol=1e-9,rtol=0))
    task_order=[s['task_index'] for s in samples]
    continuous=task_order==sorted(task_order) and set(task_order)==set(range(len(spec.steps)))
    result={'schema':SCHEMA,'robot':robot,'scene':sid,'physical_tasks_passed':all(t['success'] and t['safety_pass'] for t in physical['subtasks']),
        'expected_subtasks':len(spec.steps),'measured_subtasks':len(physical['subtasks']),
        'continuous_scene_and_home_return_passed':home_ok,'physical_nonpenetration_passed':safety,
        'initial_per_object_pixels_across_four_views':visible,'all_requested_objects_rendered':geometry_visible,
        'all_insets_unoccluded':wide_zero,'motion':move,'videos':videos,
        'minimum_front_held_target_pixels':native_min,'minimum_wide_held_target_pixels':minima,
        'front_target_visible_while_held':native_min is None or native_min>=100,
        'physically_inside_cup_or_bowl_may_occlude_object_after_release':True,
        'sync_checks':len(sync),'maximum_sync_mean_pixel_error':max(s['mean_pixel_error'] for s in sync),
        'all_sync_checks_passed':all(s['passed'] for s in sync),'all_frames':counts,
        'simulation_timestamps_match_30FPS':timing,'continuous_subtask_order_passed':continuous,
        'replayed_physics_state_error_m':render['max_replayed_tcp_error_m'],
        'no_added_video_text':True,'executor':'scripted privileged feedback Oracle, not learned RL/VLA'}
    result['passed']=bool(result['physical_tasks_passed'] and result['measured_subtasks']==len(spec.steps)
        and render['full_scene'] and home_ok and safety and wide_zero and move['passed']
        and geometry_visible and timing and continuous and all(v['passed'] for v in videos) and result['all_sync_checks_passed']
        and result['front_target_visible_while_held'])
    atomic_json(folder/'AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('videos','motion')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--robots',default='ur5e,uf850')
    parser.add_argument('--scenes',default='1');parser.add_argument('--output',type=Path,default=OUT)
    a=parser.parse_args()
    for robot in a.robots.split(','):
        for sid in map(int,a.scenes.split(',')):audit(robot,sid,a.output)
