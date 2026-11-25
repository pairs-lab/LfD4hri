"""Research-informed feedback Oracle with original robot/gripper appearance.

Run in a new folder. The two prior video editions are protected. This is not
ACT, Diffusion Policy or a trained RL network; see research/SURVEY.md.
"""
from __future__ import annotations
import argparse
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pybullet as p
import imageio_ffmpeg
from scipy.spatial.transform import Rotation
from VLA.collision_grasp_planner import plan_body_grasp,wrap_axis_angle
from env_eval.oracle_demo import Rollout,scenes,parse_config
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.oracle_trajectory import PoseCurve,minimum_jerk
from env_eval.feedback_motion import MotionProfile,CorrectionEnsemble,approach_chunk,transport_chunk,retract_chunk
from env_eval.feedback_environment import FeedbackEnvironment
from utils.camera import CINEMATIC_VIEWS as DEMO_VIEWS


class FeedbackRollout(CinematicRollout):
    def __init__(self,env,scene,directory,video=False,width=1920,height=1080,fps=30,*,profile=None):
        self.profile=profile or MotionProfile()
        self.target_feedback=np.zeros(3)
        self.feedback_initial_object=None
        super().__init__(env,scene,directory,False,width,height,fps)
        if video:
            for filename,size in [(f'view_{i}.mp4',(width,height)) for i in range(1,len(DEMO_VIEWS)+1)]+[('gripper_closeup.mp4',self.macro_size)]:
                writer=imageio_ffmpeg.write_frames(str(directory/filename),size,fps=fps,
                    codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error',
                    output_params=['-preset','fast','-tune','zerolatency','-rc-lookahead','0',
                                   '-bf','0','-refs','1','-threads','2','-pix_fmt','yuv420p',
                                   '-movflags','+faststart'])
                writer.send(None);self.writers.append(writer)

    def capture(self):
        super().capture()
        if self.initializing:return
        if hasattr(self.env,'reference_acceleration'):
            self.samples[-1].update(reference_joints=self.env.q.tolist(),
                reference_velocity=self.env.q_velocity.tolist(),
                reference_acceleration=self.env.reference_acceleration.tolist(),
                inverse_kinematics_target=self.env.last_ik.tolist(),
                table_guard_correction_m=getattr(self.env,'table_guard_correction',0.),
                commanded_tcp=getattr(self,'commanded_tcp',self.env.gripper.grasp_center()).tolist())

    def step(self,center,q,command):
        e=self.env
        e.gripper.pregrasp_force_enabled=self.scene.action!='touch' and self.phase in ('approach','grasp')
        if self.phase=='approach' and self.scene.action!='touch' and self.grasp_target is not None:
            distance=np.linalg.norm(e.gripper.grasp_center()-(self.grasp_target+self.target_feedback))
            e.gripper.assist_grasp=distance<.0020
            if self.approach_close_elapsed is None and distance<self.profile.contact_distance:
                self.approach_close_elapsed=0.;self.approach_close_start=float(command)
            if self.approach_close_elapsed is not None:
                self.approach_close_elapsed+=1/e.physics_hz
                command=self.approach_close_start+(1-self.approach_close_start)*minimum_jerk(
                    self.approach_close_elapsed/self.profile.contact_close_seconds)
        else:e.gripper.assist_grasp=True
        self.current_command=float(command)
        self.commanded_tcp=np.asarray(center).copy()
        Rollout.step(self,center,q,command)
        if self.scene.action=='touch':
            self.max_touch_force=max(self.max_touch_force,
                sum(c[9] for c in p.getContactPoints(e.gripper.body,e.pick_id)))

    def curve(self,curve,phase,command_start=-1.,command_end=None):
        self.phase=phase
        if command_end is None:command_end=command_start
        if phase=='approach':
            self.feedback_initial_object=np.array(p.getBasePositionAndOrientation(self.env.pick_id)[0])
            ensemble=CorrectionEnsemble();previous_displacement=np.zeros(3);observed_velocity=np.zeros(3)
        carry_offset=None
        if phase=='carry':
            tcp=self.env.gripper.grasp_center();obj=np.array(p.getBasePositionAndOrientation(self.env.pick_id)[0])
            q0=curve.sample(0)[1];carry_offset=Rotation.from_quat(q0).inv().apply(obj-tcp)
        for tick in range(1,round(curve.duration*self.env.physics_hz)+1):
            elapsed=tick/self.env.physics_hz;self.phase_progress=elapsed/curve.duration
            center,q=curve.sample(elapsed)
            if phase=='approach' and self.env.gripper.grasped_object is None:
                displacement=np.array(p.getBasePositionAndOrientation(self.env.pick_id)[0])-self.feedback_initial_object
                # Limit corrections caused by object settling/contact. The
                # same feedback also handles genuine small target movements.
                displacement=np.clip(displacement,-.025,.025)
                if tick==1 or tick%8==0:
                    observed_velocity+=.25*(np.clip((displacement-previous_displacement)*30,-.03,.03)-observed_velocity)
                    previous_displacement=displacement.copy()
                    ensemble.add(elapsed,displacement,observed_velocity)
                self.target_feedback=ensemble.sample(elapsed)
                center=center+self.target_feedback*minimum_jerk(min(1.,self.phase_progress/.45))
            if carry_offset is not None:
                center=center+Rotation.from_quat(q0).apply(carry_offset)-Rotation.from_quat(q).apply(carry_offset)
            def preview(ahead):
                point,orientation=curve.sample(elapsed+ahead)
                if phase=='approach':
                    point=point+self.target_feedback*minimum_jerk(min(1.,(elapsed+ahead)/curve.duration/.45))
                if carry_offset is not None:
                    point=point+Rotation.from_quat(q0).apply(carry_offset)-Rotation.from_quat(orientation).apply(carry_offset)
                return point,orientation
            self.env.preview_sampler=preview
            clock=minimum_jerk(self.phase_progress)
            self.step(center,q,command_start+(command_end-command_start)*clock)
            if self.scene.action=='touch' and self.touch and phase in ('approach','touch_search'):break
            if phase in ('approach','grasp') and self.env.gripper.grasped_object==self.env.pick_id and self.env.gripper._bilateral(self.env.pick_id):break
        self.env.preview_sampler=None

    def hold(self,center,q,duration,phase,command_start,command_end=None):
        if phase=='grasp':center=np.asarray(center)+self.target_feedback
        if phase!='release':
            return super().hold(center,q,duration,phase,command_start,command_end)
        # Once the grasp releases, unload the fingertips while completing the
        # opening command. The subsequent retract starts from the measured pose.
        self.phase=phase
        for tick in range(1,round(duration*self.env.physics_hz)+1):
            self.phase_progress=tick/(duration*self.env.physics_hz)
            command=command_start+(command_end-command_start)*minimum_jerk(self.phase_progress)
            unloading=.006*minimum_jerk(np.clip((self.phase_progress-.56)/.44,0,1))
            elapsed=tick/self.env.physics_hz
            def preview(ahead):
                progress=np.clip((elapsed+ahead)/duration,0,1)
                lift=.006*minimum_jerk(np.clip((progress-.56)/.44,0,1))
                return np.asarray(center)+[0,0,lift],q
            self.env.preview_sampler=preview
            self.step(np.asarray(center)+[0,0,unloading],q,command)
        self.env.preview_sampler=None

    def run(self):
        result=self._execute()
        commands=self.env.physics_commands
        a=np.array([s['a'] for s in commands]);jerk=np.diff(a,axis=0)*self.env.physics_hz
        result.update(motion_profile=self.profile.to_dict(),
            reference_max_jerk_rad_s3=float(abs(jerk).max()),
            reference_jerk_limit_rad_s3=self.profile.joint_jerk,
            closed_loop_target_correction_m=float(np.linalg.norm(self.target_feedback)),
            original_materials_preserved=True)
        return result

    def _execute(self):
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
        tempo=self.profile.tempo
        approach_seconds=(5.5 if e.robot_spec.name=='uf850' else 3.5)*tempo*self.profile.approach_scale
        lean=self.profile.wrist_compensation
        self.controlled_variation={'profile':self.profile.to_dict(),
                                   'geometry':'task-conditioned C2 chunks; no seeded ornaments'}
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
            self.curve(approach_chunk(center,target,q0,q1,duration=approach_seconds),'approach')
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
            self.curve(approach_chunk(center,target,q0,q1,duration=approach_seconds),'approach',-1.,preshape)
            if e.gripper.grasped_object!=e.pick_id:
                self.hold(target,q1,.60*tempo,'grasp',self.current_command,1.)
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
            clearance=self.corridor_clearance(center,target)+self.profile.corridor_margin-.025
            duration=(2.7 if self.scene.action=='pick' else 3.65)*tempo
            # Begin transport at the measured contact pose, including wrist
            # orientation, rather than abruptly demanding the nominal pose.
            q_contact=e.pose()[1]
            path=transport_chunk(center,target,q_contact,q1,clearance,duration,
                                 lift=self.scene.action=='pick',wrist=lean)
            self.curve(path,'carry',1.)
            if self.scene.action=='put':
                self.hold(target,q1,.55*tempo,'release',1.,-1.)
                target=e.gripper.grasp_center();q1=e.pose()[1]
                retract=target+[0,0,.15]
                self.curve(retract_chunk(target,retract,q1,duration=1.45*tempo),'retract',-1.)
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
            'controller':'privileged state-feedback Oracle with jerk-bounded execution; not a trained policy',
            'camera_views':[dict(c) for c in DEMO_VIEWS],
            'videos':[f'view_{i}.mp4' for i in range(1,len(DEMO_VIEWS)+1)]+['gripper_closeup.mp4']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robots',default='uf850,ur5e')
    parser.add_argument('--actions',default='touch,pick,move,put')
    parser.add_argument('--configs',default='1,2,3,4')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--config-file',type=Path)
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    protected=[Path(__file__).resolve().parent/'demos'/name for name in ('oracle_20261015','oracle_policy_style_v2_20261015')]
    if any(args.output.resolve()==old or old in args.output.resolve().parents for old in protected):
        raise ValueError('Use a new directory; previous video editions are protected')
    args.output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    names=('env_eval/oracle_feedback.py','env_eval/feedback_environment.py','env_eval/feedback_motion.py',
           'env_eval/oracle_cinematic.py','env_eval/legacy_gripper_environment.py','env_eval/oracle_demo.py',
           'env_eval/oracle_style.py','utils/Robotiq2F85.py','utils/Robotiq2F85UF850.py','utils/camera.py')
    (args.output/'run_sources.json').write_text(json.dumps({name:hashlib.sha256((project/name).read_bytes()).hexdigest()
                                                         for name in names},indent=2))
    profile_data=json.loads(args.profile.read_text()) if args.profile else {}
    selected=[parse_config(args.config_file)] if args.config_file else scenes();results=[]
    for robot in args.robots.split(','):
        profile=MotionProfile(**profile_data.get(robot,profile_data))
        with FeedbackEnvironment(robot,args.output/'assets',not args.no_video) as env:
            env.joint_jerk=profile.joint_jerk
            for scene in selected:
                if scene.action not in args.actions.split(',') or str(scene.config) not in args.configs.split(','):continue
                directory=args.output/('UF850' if robot=='uf850' else 'UR5e_Robotiq2F')/scene.action/f'config{scene.config}'
                if (directory/'audit.json').exists():
                    result=json.loads((directory/'audit.json').read_text())
                    if args.resume and result.get('overall_pass'):results.append(result);continue
                    raise FileExistsError(f'{directory} already contains an attempt; use a new directory')
                directory.mkdir(parents=True,exist_ok=True);rollout=None
                if any(directory.glob('*.mp4')):
                    raise FileExistsError(f'Partial videos exist in {directory}; use another directory')
                try:
                    env.reset(scene.task());rollout=FeedbackRollout(env,scene,directory,not args.no_video,profile=profile)
                    result=rollout.run()
                except Exception as exc:
                    result={'robot':robot,'action':scene.action,'config':scene.config,'overall_pass':False,'error':str(exc)}
                finally:
                    if rollout:
                        rollout.close()
                        (directory/'trajectory.json').write_text(json.dumps(rollout.samples))
                        (directory/'visibility.json').write_text(json.dumps(rollout.visibility))
                        commands=env.physics_commands
                        np.savez_compressed(directory/'joint_reference.npz',q=np.array([s['q'] for s in commands]),
                            v=np.array([s['v'] for s in commands]),a=np.array([s['a'] for s in commands]),dt=1/env.physics_hz)
                (directory/'audit.json').write_text(json.dumps(result,indent=2));results.append(result)
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({k:v for k,v in result.items() if k in ('robot','action','config','overall_pass','error','duration_seconds','max_tracking_error','reference_max_jerk_rad_s3','max_touch_force_n')}),flush=True)
    return 0 if results and all(r['overall_pass'] for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
