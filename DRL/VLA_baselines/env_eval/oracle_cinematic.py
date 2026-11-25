"""Second, presentation-oriented Oracle edition with a stabilized gripper inset.

The existing oracle_20261015 videos are read-only. This is still a privileged
Oracle, not a learned policy. Each scene produces three wide/inset films and
an independent HD close-up stream, all synchronized to 240 Hz physics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pybullet as p
import imageio_ffmpeg
from scipy.spatial.transform import Rotation
from VLA.collision_grasp_planner import plan_body_grasp, wrap_axis_angle
from env_eval.oracle_demo import Rollout, scenes, parse_config
from env_eval.oracle_trajectory import PoseCurve, minimum_jerk
from env_eval.oracle_style import styled_approach, styled_carry, styled_retract, styled_lift
from utils.camera import CINEMATIC_VIEWS as DEMO_VIEWS

OLD_OUTPUT=Path(__file__).resolve().parent/'demos/oracle_20261015'
class CinematicRollout(Rollout):
    def __init__(self,env,scene,directory,video=False,width=1920,height=1080,fps=30):
        self.initializing=True
        super().__init__(env,scene,directory,False,width,height,fps)
        self.initializing=False
        self.camera_focus=None
        self.macro_axis=np.array([-1.,0.,0.])
        self.macro_normal=np.array([0.,-1.,0.])
        self.macro_size=(1280,720)
        self.controlled_variation={}
        self.current_command=-1.
        self.phase_visits={}
        self.inset_corners={}
        self.max_touch_force=0.
        self.grasp_target=None
        self.approach_close_elapsed=None
        self.directory.mkdir(parents=True,exist_ok=True)
        if video:
            for filename,size in [(f'view_{i}.mp4',(width,height)) for i in range(1,len(DEMO_VIEWS)+1)]+[('gripper_closeup.mp4',self.macro_size)]:
                writer=imageio_ffmpeg.write_frames(str(directory/filename),size,fps=fps,
                    codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error',
                    output_params=['-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart'])
                writer.send(None);self.writers.append(writer)

    def frame(self,cfg,width,height):
        view=p.computeViewMatrix(cfg['eye'],cfg['target'],cfg.get('up',(0,0,1)))
        projection=p.computeProjectionMatrixFOV(cfg['fov'],width/height,.015,4.)
        _,_,rgba,_,seg=p.getCameraImage(width,height,viewMatrix=view,projectionMatrix=projection,
            renderer=self.renderer,shadow=1,lightDirection=[-1,-.6,2],lightColor=[1,1,.98],
            lightAmbientCoeff=.55,lightDiffuseCoeff=.65,lightSpecularCoeff=.15,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return np.asarray(rgba,dtype=np.uint8).reshape(height,width,4)[:,:,:3].copy(), np.asarray(seg,dtype=np.int64).reshape(height,width)

    def macro_geometry(self):
        e=self.env;tcp=e.gripper.grasp_center()
        low,high=map(np.asarray,p.getAABB(e.pick_id));obj=(low+high)/2
        separation=float(np.linalg.norm(obj-tcp))
        blend=.4*float(np.clip((.17-separation)/.10,0,1))
        placing=self.scene.action=='put' and self.phase in ('release','retract','settle')
        if placing:blend=.65
        focus=(1-blend)*tcp+blend*obj+[0,0,.025 if placing else .04]
        # A world-stabilized director camera; it never rolls with the wrist.
        if self.camera_focus is None:self.camera_focus=focus.copy()
        else:self.camera_focus += .28*(focus-self.camera_focus)
        distance=.32+max(0.,separation-.07)*(.8 if blend else .25)
        side_offset=.055
        eye=self.camera_focus+self.macro_normal*distance+self.macro_axis*side_offset+[0,0,.105]
        if self.scene.action=='put':
            # Keep the whole placed object and attachment visible, including
            # a long spoon handle after the arm retracts. Fit in camera space.
            corners=lambda lo,hi:np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            if placing:
                context=np.vstack([corners(low,high),*e.gripper.world_pad_vertices()])
            else:
                a,b=map(np.asarray,p.getAABB(e.gripper.body,0))
                context=np.vstack([corners(low,high),corners(a,b)])
            ray=eye-self.camera_focus;length=np.linalg.norm(ray);direction=ray/length
            safe=length
            for _ in range(10):
                candidate=self.camera_focus+direction*safe
                forward=-direction;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
                up=np.cross(right,forward);relative=context-candidate;depth=relative@forward
                if depth.min()<=.015:safe*=1.3;continue
                tan=np.tan(np.radians(43.)/2)
                required=max(float(np.max(abs(relative@right)/(depth*tan*(16/9))/.84)),
                             float(np.max(abs(relative@up)/(depth*tan)/.84)))
                if required<=1:break
                safe*=max(1.02,required)
            weight=1. if self.phase in ('grasp','carry','release','retract','settle') else float(np.clip((.18-separation)/.10,0,1))
            desired=length+weight*(safe-length)
            previous=getattr(self,'framing_distance',desired)
            self.framing_distance=max(desired,previous+.12*(desired-previous))
            eye=self.camera_focus+direction*self.framing_distance
        return {'eye':eye.tolist(),'target':self.camera_focus.tolist(),'up':(0,0,1),'fov':43.}

    def inset(self,wide,macro,seg,view_index):
        width=int(self.width*.31)//2*2;height=round(width*9/16)//2*2
        margin=round(self.width*.018);top=margin+2
        ids=seg&((1<<24)-1)
        robot_mask=(ids==self.env.robot_id)|(ids==self.env.gripper.body)
        if view_index not in self.inset_corners:
            candidates=[margin,self.width-width-margin]
            costs=[int(robot_mask[top-3:top+height,left:left+width].sum())+
                   100*int((ids[top-3:top+height,left:left+width]==self.env.pick_id).sum())
                   for left in candidates]
            self.inset_corners[view_index]=candidates[int(np.argmin(costs))]
        left=self.inset_corners[view_index]
        resized=cv2.resize(macro,(width,height),interpolation=cv2.INTER_AREA)
        # The wide scene remains visible; the inset occupies empty upper-right space.
        cv2.rectangle(wide,(left-3,top-3),(left+width+3,top+height+3),(28,32,38),-1)
        cv2.rectangle(wide,(left-2,top-2),(left+width+2,top+height+2),(226,231,236),2)
        wide[top:top+height,left:left+width]=resized
        # The user's videos must contain no text, titles, or phase labels.
        return wide,(left,top,width,height)

    def capture(self):
        if self.initializing:return
        e=self.env;position,orientation=e.pose()
        obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
        states=p.getJointStates(e.robot_id,e.robot_spec.arm_joint_ids)
        contact=p.getContactPoints(e.gripper.body,e.pick_id)
        sample={'time':self.tick_count/e.physics_hz,'phase':self.phase,'phase_progress':self.phase_progress,
            'tcp':e.gripper.grasp_center().tolist(),'object':obj.tolist(),'mount_quaternion':orientation.tolist(),
            'mount_position':position.tolist(),
            'joints':[s[0] for s in states],'joint_velocities':[s[1] for s in states],
            'relative_mount_position':getattr(e.gripper,'relative_mount_position',e.gripper.mount_offset.tolist()),
            'relative_mount_quaternion':getattr(e.gripper,'relative_mount_quaternion',list(e.gripper.mount_rotation)),
            'gripper_width':e.gripper.width(),'gripper_command':self.current_command,
            'finger_joints':[p.getJointState(e.gripper.body,j)[0] for j in e.gripper.JOINTS],
            'contact_force_n':float(sum(c[9] for c in contact)),
            'physical_contact':bool(contact),
            'bilateral':e.gripper._bilateral(e.pick_id),'held':e.gripper.grasped_object==e.pick_id,
            'pad_forces':[sum(c[9] for c in contact if c[3]==link) for link in e.gripper.PAD_LINKS]}
        low,high=map(np.asarray,p.getAABB(e.pick_id))
        sample['tcp_object_distance']=float(np.linalg.norm(e.gripper.grasp_center()-(low+high)/2))
        self.samples.append(sample)
        if not self.writers:return
        macro_cfg=self.macro_geometry()
        macro,macro_seg=self.frame(macro_cfg,*self.macro_size)
        self.writers[-1].send(macro)
        macro_body=(macro_seg&((1<<24)-1))==e.gripper.body
        macro_links=(macro_seg>>24)-1
        visibility={'time':sample['time'],'phase':self.phase,'view':'gripper_closeup',
            'visible_pick_pixels':int(((macro_seg&((1<<24)-1))==e.pick_id).sum()),
            'pad_pixels':[int((macro_body&(macro_links==link)).sum()) for link in e.gripper.PAD_LINKS],
            'camera':macro_cfg,'tcp_object_distance':sample['tcp_object_distance']}
        ys,xs=np.where((macro_seg&((1<<24)-1))==e.pick_id)
        visibility['object_border_margin_px']=int(min(xs.min(),self.macro_size[0]-1-xs.max(),ys.min(),self.macro_size[1]-1-ys.max())) if len(xs) else -1
        self.visibility.append(visibility)
        stage=('start' if self.phase_progress<.05 else 'middle' if .48<=self.phase_progress<=.53
               else 'end' if self.phase_progress>=.98 else None)
        if self.phase in ('initial','final','settle') and stage is None:stage='middle'
        frames=self.directory/'frames';frames.mkdir(exist_ok=True)
        key=(self.phase,stage,'gripper_closeup')
        if stage is not None and key not in self.phase_frames:
            cv2.imwrite(str(frames/f'{self.phase}_{stage}_gripper_closeup.jpg'),cv2.cvtColor(macro,cv2.COLOR_RGB2BGR))
            self.phase_frames[key]=True
        for i,cfg in enumerate(DEMO_VIEWS):
            wide,seg=self.frame(cfg,self.width,self.height)
            raw=wide.copy()
            wide,rect=self.inset(wide,macro,seg,i)
            self.writers[i].send(wide)
            pick_mask=(seg&((1<<24)-1))==e.pick_id
            left,top,w,h=rect
            hidden=int(pick_mask[top:top+h,left:left+w].sum())
            self.visibility.append({'time':sample['time'],'phase':self.phase,'view':i+1,
                'visible_pick_pixels':int(pick_mask.sum()),'pick_pixels_covered_by_inset':hidden,
                'robot_pixels_covered_by_inset':int((((seg&((1<<24)-1))==e.robot_id)[top:top+h,left:left+w]).sum()),
                'inset_rect':[int(x) for x in rect]})
            key=(self.phase,stage,i)
            if stage is not None and key not in self.phase_frames:
                cv2.imwrite(str(frames/f'{self.phase}_{stage}_view_{i+1}.jpg'),cv2.cvtColor(wide,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(frames/f'{self.phase}_{stage}_wide_{i+1}.jpg'),cv2.cvtColor(raw,cv2.COLOR_RGB2BGR))
                self.phase_frames[key]=True

    def step(self,center,q,command):
        if self.phase=='approach' and self.scene.action!='touch' and self.grasp_target is not None:
            tcp=self.env.gripper.grasp_center()
            # Establish a body-level pinch rather than latch at the upper edge.
            self.env.gripper.assist_grasp=np.linalg.norm(tcp-self.grasp_target)<.0018
            if self.approach_close_elapsed is None and np.linalg.norm(tcp-self.grasp_target)<.008:
                self.approach_close_elapsed=0.
                self.approach_close_start=float(command)
            if self.approach_close_elapsed is not None:
                self.approach_close_elapsed+=1/self.env.physics_hz
                command=self.approach_close_start+(1-self.approach_close_start)*minimum_jerk(self.approach_close_elapsed/.55)
        else:
            self.env.gripper.assist_grasp=True
        self.current_command=float(command)
        super().step(center,q,command)
        if self.scene.action=='touch':
            self.max_touch_force=max(self.max_touch_force,
                sum(c[9] for c in p.getContactPoints(self.env.gripper.body,self.env.pick_id)))

    def curve(self,curve,phase,command_start=-1.,command_end=None):
        self.phase=phase
        if command_end is None:command_end=command_start
        for step in range(1,round(curve.duration*self.env.physics_hz)+1):
            elapsed=step/self.env.physics_hz;self.phase_progress=elapsed/curve.duration
            center,q=curve.sample(elapsed)
            clock=minimum_jerk(elapsed/curve.duration)
            self.step(center,q,command_start+(command_end-command_start)*clock)
            if self.scene.action=='touch' and self.touch and phase in ('approach','touch_search'):break
            if phase in ('approach','grasp') and self.env.gripper.grasped_object==self.env.pick_id and self.env.gripper._bilateral(self.env.pick_id):break

    def corridor_clearance(self,start,goal):
        e=self.env
        low,high=map(np.asarray,p.getAABB(e.pick_id))
        carried_bottom_offset=float(e.gripper.grasp_center()[2]-low[2])
        radius=max(.075,float(np.linalg.norm((high-low)[:2])/2)+.025)
        start,goal=np.asarray(start),np.asarray(goal)
        line=goal[:2]-start[:2];length2=float(line@line)
        clearance=max(start[2]+.06,goal[2]+.015)
        for body in e.object_ids.values():
            if body in (e.pick_id,e.place_id):continue
            lo,hi=map(np.asarray,p.getAABB(body));centre=(lo+hi)/2
            t=np.clip(float((centre[:2]-start[:2])@line)/max(length2,1e-8),0,1)
            distance=np.linalg.norm(centre[:2]-(start[:2]+t*line))
            if distance<radius+np.linalg.norm((hi-lo)[:2])/2:
                clearance=max(clearance,float(hi[2])+carried_bottom_offset+.035)
        return float(clearance)

    def run(self):
        e=self.env
        try:
            body_plan=plan_body_grasp(e,e.pick_id) if self.scene.action!='touch' else None
        except ValueError:
            from VLA.collision_grasp_planner import minimum_caliper_grasp
            pos,quat=p.getBasePositionAndOrientation(e.pick_id)
            vertices=np.array(p.getMeshData(e.pick_id,-1,flags=p.MESH_DATA_SIMULATION_MESH)[1])
            body_plan=minimum_caliper_grasp(Rotation.from_quat(quat).apply(vertices)+pos)
            if body_plan['cross_section_width_m']>.084:
                raise ValueError('Collision footprint exceeds open gripper capacity')
        plan=({'point_world':(np.array(p.getAABB(e.pick_id)[0])+p.getAABB(e.pick_id)[1])/2, 'jaw_axis_world':None, 'method':'surface_raycast'}
              if self.scene.action=='touch' else body_plan)
        center=e.gripper.grasp_center(); _,q0=e.pose()
        # Top-down tool frame, yaw selected from collision principal axis.
        q1=np.array(p.getQuaternionFromEuler([0,np.pi,0]))
        axis=plan.get('jaw_axis_world')
        if axis is not None:
            # Open Robotiq jaw axis is local tool X under the mount transform.
            yaw=wrap_axis_angle(np.arctan2(axis[1],axis[0]))
            q1=(Rotation.from_euler('z',yaw)*Rotation.from_quat(q1)).as_quat()
        target=plan['point_world'].copy()
        table_top=p.getAABB(e.table_id)[1][2]
        target[2]=max(target[2],table_top+.021)
        rng=np.random.default_rng(20261015+self.scene.config*71+e.robot_spec.name.__len__()*13+e.task.task_id*101)
        bend=float(rng.uniform(.009,.019))
        tempo=float(rng.uniform(.91,1.08))
        lean=float(rng.uniform(.020,.035))
        self.controlled_variation={'bend_m':bend,'tempo_multiplier':tempo,'wrist_lean_rad':lean}
        jaw=Rotation.from_quat(q1).apply([-1.,0.,0.]);jaw[2]=0.;jaw/=np.linalg.norm(jaw)
        normal=np.array([-jaw[1],jaw[0],0.])
        if normal[1]>0:normal=-normal
        self.macro_axis=jaw;self.macro_normal=normal
        self.capture()
        if self.scene.action=='touch':
            # One open fingertip, rather than the empty gap, touches the surface.
            local=Rotation.from_quat(q0).inv().apply(e.gripper.world_pad_vertices()[0]-center)
            vertices=Rotation.from_quat(q1).apply(local)
            tip=vertices[vertices[:,2] <= vertices[:,2].min()+.001].mean(axis=0)
            surface=plan.get('touch_surface_world',e._pick_top()).copy()
            low,high=p.getAABB(e.pick_id)
            hit=p.rayTest([surface[0],surface[1],high[2]+.02],[surface[0],surface[1],low[2]-.01])[0]
            if hit[0]==e.pick_id:surface=np.array(hit[3])
            target=surface-tip+[0,0,-.0004]
            self.curve(styled_approach(center,target,q0,q1,duration=3.25*tempo,side=bend,lean=lean),'approach')
            if self.touch:
                target=e.gripper.grasp_center();q1=e.pose()[1]
                e.touch_reference={'mount':e.pose()[0],'tcp':target.copy()}
            self.hold(target,q1,.25,'touch',-1.)
            if not self.touch:
                self.curve(PoseCurve([target,target-[0,0,.003]],q1,q1,.5),'touch_search')
                target=e.gripper.grasp_center();q1=e.pose()[1]
                e.touch_reference={'mount':e.pose()[0],'tcp':target.copy()}
                self.hold(target,q1,.20,'touch',-1.)
            self.curve(PoseCurve([target,target+[0,0,.006]],q1,q1,.30),'touch_retract',-1.)
            self.hold(target+[0,0,.006],q1,.3,'final',-1.)
            success=self.touch and np.linalg.norm(np.array(p.getBasePositionAndOrientation(e.pick_id)[0])-e.initial_pick_position)<.025
        else:
            width=plan.get('cross_section_width_m',p.getAABB(e.pick_id)[1][0]-p.getAABB(e.pick_id)[0][0])
            desired_gap=np.clip(width+.022,.050,.0885)
            preshape=float(np.clip(2*(1-(desired_gap-.0064)/.0822)-1,-1.,.1))
            if e.task.pick_name.endswith(('knife','spoon','spatula')):
                preshape=-1.
            self.grasp_target=target.copy()
            self.curve(styled_approach(center,target,q0,q1,duration=3.25*tempo,side=bend,lean=lean),'approach',-1.,preshape)
            if e.gripper.grasped_object!=e.pick_id:
                self.hold(target,q1,.70*tempo,'grasp',self.current_command,1.)
            if e.gripper.grasped_object!=e.pick_id:
                raise RuntimeError(f'No verified bilateral grasp: width={e.gripper.width():.4f}, plan={plan}')
            obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
            center=e.gripper.grasp_center()
            if self.scene.action=='pick':
                goal=e.initial_pick_position+[0,0,.15]
            else:
                if e.place_id is None:
                    goal=np.array(e.task.place_position)+[0,0,.15]
                else:
                    goal=e._body_top(e.place_id)+[0,0,.15]
                if self.scene.action=='put':
                    lo,hi=map(np.asarray,p.getAABB(e.pick_id))
                    bottom_offset=obj[2]-lo[2]
                    if 'bowl' in self.scene.place:
                        bowl_lo,bowl_hi=p.getAABB(e.place_id)
                        goal[:2]=(np.array(bowl_lo[:2])+bowl_hi[:2])/2
                        goal[2]=bowl_hi[2]+bottom_offset+.006
                    else:goal[2]=e._body_top(e.place_id)[2]+bottom_offset+.002
            target=center+goal-obj
            clearance=self.corridor_clearance(center,target)
            duration=(2.7 if self.scene.action=='pick' else 3.65)*tempo
            # Begin transport at the measured contact pose, including wrist
            # orientation, rather than abruptly demanding the nominal pose.
            q_contact=e.pose()[1]
            path=(styled_lift(center,target,q_contact,duration=duration,side=bend,lean=lean,goal_quaternion=q1)
                  if self.scene.action=='pick' else
                  styled_carry(center,target,q_contact,clearance=clearance,duration=duration,side=bend,lean=lean,goal_quaternion=q1))
            self.curve(path,'carry',1.)
            if self.scene.action=='put':
                self.hold(target,q1,.68*tempo,'release',1.,-1.)
                retract=target+[0,0,.15]
                self.curve(styled_retract(target,retract,q1,duration=1.4*tempo),'retract',-1.)
                self.hold(retract,q1,1.2,'settle',-1.)
                obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
                support=bool(p.getContactPoints(bodyA=e.pick_id,bodyB=e.place_id))
                velocity=np.linalg.norm(p.getBaseVelocity(e.pick_id)[0])
                success=e.gripper.grasped_object is None and support and np.linalg.norm(obj[:2]-goal[:2])<.045 and velocity<.025
            else:
                self.hold(target,q1,.55,'final',1.)
                obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
                success=e.gripper.grasped_object==e.pick_id and e.gripper._bilateral(e.pick_id) and np.linalg.norm(obj-goal)<.02
        final=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
        distractor_motion={name:float(np.linalg.norm(np.array(p.getBasePositionAndOrientation(body)[0])-e.initial_objects[name]))
            for name,body in e.object_ids.items() if body not in (e.pick_id,e.place_id)}
        safety=not self.contacts and self.self_min>=-.001 and self.arm_gripper_min>=-.001 and self.ground_min>=-.001 and max(distractor_motion.values(),default=0)<.02
        joints=np.array([s['joints'] for s in self.samples]);dt=1/self.fps
        v=np.diff(joints,axis=0)/dt;a=np.diff(v,axis=0)/dt
        return {'robot':e.robot_spec.name,'action':self.scene.action,'config':self.scene.config,
            'success':bool(success),'safety_pass':bool(safety),'overall_pass':bool(success and safety),
            'physical_touch':bool(self.touch),'verified_bilateral_grasp':bool(self.verified),
            'grasp_assist':'fixed constraint only after 3 physics ticks of bilateral force-bearing pad contact',
            'final_object_position':final.tolist(),'self_min_distance':self.self_min,'arm_gripper_min_distance':self.arm_gripper_min,
            'gripper_table_min_distance':self.ground_min,'unexpected_contacts':self.contacts,
            'distractor_displacement':distractor_motion,'max_tracking_error':self.max_tracking_error,
            'max_joint_velocity':float(abs(v).max()),'max_joint_acceleration':float(abs(a).max()),
            'duration_seconds':self.tick_count/e.physics_hz,'fps':self.fps,'resolution':[self.width,self.height],
            'plan':{k:x.tolist() if isinstance(x,np.ndarray) else x for k,x in plan.items()},
            'task':e.task.to_dict(),'controlled_variation':self.controlled_variation,
            'max_touch_force_n':self.max_touch_force,
            'gripper_class':type(e.gripper).__module__+'.'+type(e.gripper).__name__,
            'mount_kind':('UF850 user root-constraint fix, zero offset at broad seating face' if e.robot_spec.name=='uf850'
                          else 'original UR5e child-link-0 mount and original dynamics'),
            'max_mount_translation_error_m':e.max_mount_translation_error,
            'max_mount_rotation_error_deg':e.max_mount_rotation_error,
            'added_video_text':False,'gripper_appearance':'native original URDF/OBJ/MTL materials; no color override',
            'controller':'privileged Oracle; presentation-oriented, not learned',
            'camera_views':[dict(c) for c in DEMO_VIEWS],
            'videos':[f'view_{i}.mp4' for i in range(1,len(DEMO_VIEWS)+1)]+['gripper_closeup.mp4']}


def main():
    from env_eval.legacy_gripper_environment import LegacyGripperEnvironment
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robots',default='uf850,ur5e')
    parser.add_argument('--actions',default='touch,pick,move,put')
    parser.add_argument('--configs',default='1,2,3,4')
    parser.add_argument('--config-file',type=Path)
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--fps',type=int,choices=[30,60],default=30)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.output.resolve()==OLD_OUTPUT or OLD_OUTPUT in args.output.resolve().parents:
        raise ValueError('The original video folder is protected; use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    selected=[parse_config(args.config_file)] if args.config_file else scenes()
    results=[]
    for robot in args.robots.split(','):
        with LegacyGripperEnvironment(robot,args.output/'assets',not args.no_video) as env:
            for scene in selected:
                if scene.action not in args.actions.split(',') or str(scene.config) not in args.configs.split(','):continue
                directory=args.output/('UF850' if robot=='uf850' else 'UR5e_Robotiq2F')/scene.action/f'config{scene.config}'
                if (directory/'audit.json').exists():
                    previous=json.loads((directory/'audit.json').read_text())
                    if args.resume and previous.get('overall_pass'):
                        results.append(previous);continue
                    raise FileExistsError(f'{directory} already contains an attempt; use another folder')
                directory.mkdir(parents=True,exist_ok=True)
                rollout=None
                try:
                    env.reset(scene.task())
                    rollout=CinematicRollout(env,scene,directory,not args.no_video,args.width,args.height,args.fps)
                    result=rollout.run()
                except Exception as exc:
                    result={'robot':robot,'action':scene.action,'config':scene.config,
                        'success':False,'overall_pass':False,'error':str(exc)}
                finally:
                    if rollout:
                        rollout.close()
                        (directory/'trajectory.json').write_text(json.dumps(rollout.samples))
                        (directory/'visibility.json').write_text(json.dumps(rollout.visibility))
                (directory/'audit.json').write_text(json.dumps(result,indent=2))
                results.append(result)
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({k:v for k,v in result.items() if k in
                    ('robot','action','config','success','overall_pass','error','duration_seconds','max_tracking_error','unexpected_contacts')}),flush=True)
    if not results:raise ValueError('Selection matched no scenes')
    return 0 if all(r['overall_pass'] for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
