"""Presentation-only replay of measured 30 Hz physical states, with no extra steps."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import cv2
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from VLA.common.artifact_io import atomic_json
from env_eval.camera_editions import writer,rectangles,projected_box,overlap
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.feedback_motion import MotionProfile
from utils.camera import CINEMATIC_VIEWS
from config import COLORS
from .assets import ICCASEnvironment
from .motion import RequestedRollout
from .scenes import get_scene
from .qualify import ROOT,OUT,ROBOTS,digest

VIEWS=(2,3,4,6)
SETTINGS=json.loads((ROOT/'env_eval/camera_editions.json').read_text())


def apply_frame(env,schema,arrays,index):
    for (name,body),pose in zip(schema['tracked'],arrays['base'][index]):
        p.resetBasePositionAndOrientation(body,pose[:3],pose[3:])
    for body,key in ((env.robot_id,'robot_joints'),(env.gripper.body,'gripper_joints')):
        for joint,value in enumerate(arrays[key][index]):p.resetJointState(body,joint,float(value),0.)
    p.performCollisionDetection()
    # This Bullet/EGL build loses a mesh's initial material when that mesh
    # receives visual-shape ID zero (the brush when it is loaded first). Refresh
    # ONLY the explicitly requested procedural prop palette, never robot/native
    # materials or any pose/dynamics. This is presentation, not policy input.
    for name,color in (('green chalk_brush','green'),('blue cup','blue'),
                       ('brown bottle','brown'),('red pen','red')):
        if name in env.object_ids:
            p.changeVisualShape(env.object_ids[name],-1,rgbaColor=COLORS[color],textureUniqueId=-1)


def choose_layouts(env,schema,arrays,samples):
    candidates=[]
    for fraction in (.31,.28,.25,.22,.20):
        for side in ('left','right'):
            rect=rectangles('front',fraction,side,SETTINGS)[0]
            candidates.append({'fraction':fraction,'side':side,'rect':rect})
    feasible={view:list(candidates) for view in VIEWS}
    for index in range(len(samples)):
        apply_frame(env,schema,arrays,index)
        for view in VIEWS:
            boxes=[projected_box(env,body,CINEMATIC_VIEWS[view-1],1920,1080) for body in env.object_ids.values()]
            feasible[view]=[candidate for candidate in feasible[view]
                if not any(overlap(box,candidate['rect'])>0 for box in boxes)]
            if not feasible[view]:raise ValueError(f'No stable unoccluded inset, view {view}, frame {index}')
    return {str(view):feasible[view][0] for view in VIEWS}


def render(robot,sid,out=OUT,limit=None):
    destination=out/ROBOTS[robot]/f'scene{sid:02d}'
    if (destination/'RENDER.json').exists():return json.loads((destination/'RENDER.json').read_text())
    selection=json.loads((destination/'PHYSICS.json').read_text());source=Path(selection['qualified_directory'])
    if digest(source/'frame_states.npz')!=selection['frame_states_sha256'] or digest(source/'trajectory.json')!=selection['trajectory_sha256']:
        raise ValueError('Qualified actual physical trajectory changed')
    if digest(Path(__file__).with_name('assets.py'))!=selection['asset_source_sha256']:
        raise ValueError('Asset geometry changed since physical qualification')
    schema=json.loads((source/'REPLAY_SCHEMA.json').read_text());samples=json.loads((source/'trajectory.json').read_text())
    arrays=np.load(source/'frame_states.npz');spec=get_scene(sid);started=time.monotonic()
    cache=out/'_camera_sources'/ROBOTS[robot]/f'scene{sid:02d}';cache.mkdir(parents=True,exist_ok=True)
    if any(destination.glob('*.mp4')) or (cache/'gripper_closeup.mp4').exists():
        raise FileExistsError('Partial media preserved; choose a separate attempt folder instead of overwriting')
    with ICCASEnvironment(robot,out/'_derived_assets'/ROBOTS[robot],True) as env:
        if env.demo_renderer!=p.ER_BULLET_HARDWARE_OPENGL:raise RuntimeError('EGL required, no silent renderer substitution')
        env.reset(spec.task());rollout=RequestedRollout(env,spec,cache,MotionProfile(**schema['profile']))
        if [(n,int(b)) for n,b in schema['tracked']]!=rollout.tracked:raise ValueError('Replay body IDs or object list changed')
        preflight=destination/'LAYOUT_PREFLIGHT.json'
        qualified=json.loads(preflight.read_text()) if preflight.exists() else None
        if qualified and qualified['passed'] and qualified['frame_states_sha256']==selection['frame_states_sha256']:
            layouts=qualified['layouts']
        else:layouts=choose_layouts(env,schema,arrays,samples)
        atomic_json(destination/'layout.json',layouts)
        writers={view:writer(destination/f'view_{view}.mp4',(1920,1080),30) for view in VIEWS}
        front_writer=writer(cache/'gripper_closeup.mp4',(1280,720),30)
        review=destination/'review';review.mkdir(exist_ok=True)
        visibility=[];max_tcp=0.;max_mount=0.;events=set();last=None
        for i,s in enumerate(samples):
            key=s['task_index'],s['phase']
            if key!=last:events.add(i);last=key
        events.update((0,len(samples)-1))
        try:
            for index,s in enumerate(samples[:limit] if limit else samples):
                apply_frame(env,schema,arrays,index)
                step=spec.steps[s['task_index']];env.pick_id=env.object_ids[step.pick];env.place_id=env.object_ids.get(step.place)
                rollout.scene=SimpleNamespace(action=step.action,place=step.place,config=sid)
                rollout.phase=s['phase'];rollout.phase_progress=s['phase_progress']
                plan=schema['subtasks'][s['task_index']]['plan'];q1=Rotation.from_euler('xyz',[0,np.pi,0])
                if plan.get('jaw_axis_world') is not None:
                    axis=plan['jaw_axis_world'];yaw=(np.arctan2(axis[1],axis[0])+np.pi/2)%np.pi-np.pi/2
                    q1=Rotation.from_euler('z',yaw)*q1
                jaw=q1.apply([-1.,0.,0.]);jaw[2]=0;jaw/=np.linalg.norm(jaw)
                rollout.macro_axis=jaw;rollout.macro_normal=np.array([-jaw[1],jaw[0],0.])
                if rollout.macro_normal[1]>0:rollout.macro_normal=-rollout.macro_normal
                tcp=env.gripper.grasp_center();max_tcp=max(max_tcp,float(np.max(abs(tcp-np.array(s['tcp'])))))
                max_mount=max(max_mount,float(np.max(abs(env.pose()[0]-np.asarray(s['mount_position'])))))
                if max_tcp>2e-6 or max_mount>2e-6:raise ValueError('Presentation replay FK differs from measured physical trajectory')
                macro,cseg=rollout.frame(rollout.macro_geometry(),1280,720);front_writer.send(macro)
                ids=cseg&((1<<24)-1);links=(cseg>>24)-1
                row={'frame':index,'simulation_time_s':s['time'],'task_index':s['task_index'],'phase':s['phase'],
                    'target_pixels_front':int((ids==env.pick_id).sum()),
                    'pad_pixels_front':[int(((ids==env.gripper.body)&(links==link)).sum()) for link in env.gripper.PAD_LINKS],
                    'tcp_object_distance':s['tcp_object_distance'],'wide':{}}
                for view in VIEWS:
                    rgb,seg=rollout.frame(CINEMATIC_VIEWS[view-1],1920,1080)
                    _,x,y,w,h=layouts[str(view)]['rect'];all_ids=seg&((1<<24)-1)
                    covered=sum(int((all_ids[y-3:y+h+3,x-3:x+w+3]==body).sum()) for body in env.object_ids.values())
                    if covered:raise ValueError(f'Inset covers physical object, frame {index}, view {view}')
                    rgb[y-3:y+h+3,x-3:x+w+3]=(221,227,232)
                    rgb[y:y+h,x:x+w]=cv2.resize(macro,(w,h),interpolation=cv2.INTER_AREA)
                    writers[view].send(rgb)
                    row['wide'][str(view)]={'target_pixels':int((all_ids==env.pick_id).sum()),
                        'goal_pixels':int((all_ids==env.place_id).sum()) if env.place_id is not None else None,
                        'per_object_pixels':{name:int((all_ids==body).sum()) for name,body in env.object_ids.items()},
                        'object_pixels_covered_by_inset':covered}
                    if index in events:cv2.imwrite(str(review/f'{index:05d}_task{s["task_index"]+1}_{s["phase"]}_view_{view}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if index in events:cv2.imwrite(str(review/f'{index:05d}_task{s["task_index"]+1}_{s["phase"]}_front.jpg'),cv2.cvtColor(macro,cv2.COLOR_RGB2BGR))
                visibility.append(row)
                if index%30==0:
                    atomic_json(destination/'RENDER_PROGRESS.json',{'frame':index+1,'total':len(samples),'robot':robot,'scene':sid,
                        'wall_s':time.monotonic()-started})
        finally:
            front_writer.close()
            for w in writers.values():w.close()
        atomic_json(cache/'visibility.json',visibility)
        result={'robot':robot,'scene':sid,'frames':len(visibility),'expected_frames':len(samples),
            'full_scene':len(visibility)==len(samples),'renderer':'PyBullet EGL hardware OpenGL',
            'frame_state_replay_only_no_extra_physics_steps':True,'max_replayed_tcp_error_m':max_tcp,
            'procedural_prop_palette_explicitly_refreshed_for_EGL':True,
            'max_replayed_mount_error_m':max_mount,'fps':30,'resolution':[1920,1080],
            'views':list(VIEWS),'front_only':True,'handeye':False,'wall_s':time.monotonic()-started,
            'actual_physical_source':str(source),'native_front':str(cache/'gripper_closeup.mp4')}
        atomic_json(destination/'RENDER.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--robots',default='ur5e,uf850')
    parser.add_argument('--scenes',default='1');parser.add_argument('--output',type=Path,default=OUT)
    a=parser.parse_args()
    for robot in a.robots.split(','):
        for sid in map(int,a.scenes.split(',')):render(robot,sid,a.output)
