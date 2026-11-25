"""Run the requested 16 scenes on both robots with continuous Oracle control.

python -m env_eval.oracle_demo --output env_eval/demos/oracle_20261015
Physics-only qualification: add --no-video. Render the same seeds afterwards.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import copy
import importlib.util
import json
import xml.etree.ElementTree as ET
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import imageio_ffmpeg
from VLA.common.environment import ManipulationEnv, ASSET_ROOT
from VLA.common.body_environment import DERIVED_ASSETS
from VLA.common.body_gripper import BodyContactRobotiq
from VLA.common.tasks import TaskSpec, SceneObjectSpec
from VLA.collision_grasp_planner import plan_body_grasp, wrap_axis_angle
from env_eval.oracle_trajectory import PoseCurve, approach_curve, carry_curve, minimum_jerk
from utils.camera import DEMO_VIEWS

NATURAL = {'apple':'red apple', 'orange':'orange orange', 'lemon':'yellow lemon',
           'egg':'white egg', 'banana':'yellow banana', 'strawberry':'red strawberry',
           'knife':'gray knife', 'spoon':'gray spoon', 'spatula':'gray spatula'}

@dataclass(frozen=True)
class Scene:
    action: str
    config: int
    objects: tuple[str, ...]
    pick: str
    place: str | None = None

    def task(self):
        pick = NATURAL.get(self.pick, self.pick)
        place = NATURAL.get(self.place, self.place)
        front = place == 'toward camera (front)'
        # Deterministic, separated tabletop layouts; every requested object is present.
        pick_pos = (-.12, -.50, .045)
        place_pos = (-.12, -.70, .025) if front else (.15, -.50, .045)
        slots = [(-.26,-.29,.045), (.0,-.28,.045), (.26,-.28,.045),
                 (-.26,-.72,.045), (.17,-.72,.045), (.32,-.66,.045)]
        others=[]
        for obj in self.objects:
            name=NATURAL.get(obj,obj)
            if name in (pick, place):
                continue
            others.append(SceneObjectSpec(name, slots[len(others)], name.endswith(' bowl'), 'demo_distractor'))
        name = 'pick_place' if self.action == 'put' else self.action
        command = f'{self.action} the {self.pick}'
        if self.place:
            command += (' into ' if 'bowl' in self.place else ' to ') + self.place
        return TaskSpec(['touch','pick','move','put'].index(self.action), name, command,
                        pick, pick_pos, None if front else place,
                        place_pos if self.place else None, max_steps=10000,
                        scene_profile='oracle_demo16', scene_objects=tuple(others))


def scenes():
    rows=[
      ('touch','orange,lemon,orange block,yellow block,blue block,blue bowl','orange',None),
      ('touch','blue block,green block,knife,banana,blue bowl,spatula','green block',None),
      ('touch','egg,green block,spatula','egg',None),
      ('touch','orange block,yellow block,blue block,yellow bowl,spatula','orange block',None),
      ('pick','egg,apple,blue block,orange bowl,blue bowl','apple',None),
      ('pick','orange block,red block,blue block,apple','red block',None),
      ('pick','strawberry,banana,yellow block,orange bowl','yellow block',None),
      ('pick','blue block,apple,knife,red block,blue bowl','blue block',None),
      ('move','knife,blue block,blue bowl,green bowl,yellow block,red block','blue block','red block'),
      ('move','banana,yellow block,blue block,green block','yellow block','toward camera (front)'),
      ('move','lemon,red block,orange block,green block','lemon','orange block'),
      ('move','knife,green block,blue block,yellow bowl,spatula','knife','blue block'),
      ('put','lemon,orange block,orange,blue block,spoon,orange bowl','spoon','orange block'),
      ('put','lemon,yellow block,blue block,banana','lemon','yellow block'),
      ('put','blue block,yellow block,green block,orange block,red block,orange bowl','blue block','orange bowl'),
      ('put','orange bowl,blue block,orange block,green block,yellow bowl,banana','blue block','yellow bowl')]
    return [Scene(a, i%4+1, tuple(o.split(',')), pick, place) for i,(a,o,pick,place) in enumerate(rows)]


def parse_config(path):
    fields={}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            key,value=line.split(':',1)
            fields[key.strip().lower()]=value.strip()
    objects=tuple(x.strip() for x in fields['config'].split(','))
    pick=fields['pick object']; place=fields.get('place object')
    action=fields['command'].split()[0].lower()
    action={'put':'put','touch':'touch','pick':'pick','move':'move'}.get(action)
    if action is None or pick not in objects or (place and place not in objects and place!='toward camera (front)'):
        raise ValueError('Config must name a supported action and scene targets')
    return Scene(action,1,objects,pick,place)


def collision_urdf(spec, directory):
    """Restore missing arm collision meshes in a separate, inspectable URDF."""
    source=spec.urdf
    text=source.read_text()
    if spec.name in ('ur5e','robotiq_2f_85'):
        import re
        text=re.sub(r'<!--\s*(<collision>.*?</collision>)\s*-->',r'\1',text,flags=re.S)
    root=ET.fromstring(text)
    # Restoring commented collision blocks must not expose diagnostic visual
    # markers that share a comment in the native URDF (e.g. UR5e tool0 box).
    original=ET.fromstring(source.read_text())
    native_visuals={link.get('name'):link.findall('visual') for link in original.findall('link')}
    for link in root.findall('link'):
        for visual in link.findall('visual'):link.remove(visual)
        for visual in native_visuals.get(link.get('name'),[]):
            link.append(copy.deepcopy(visual))
    for mesh in root.findall('.//mesh'):
        mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
    for link in root.findall('link'):
        if link.find('collision') is None and link.find('visual/geometry/mesh') is not None:
            visual=link.find('visual')
            collision=ET.SubElement(link,'collision')
            origin=visual.find('origin')
            if origin is not None: collision.append(copy.deepcopy(origin))
            collision.append(copy.deepcopy(visual.find('geometry')))
        if link.find('collision') is not None and not any(v.find('geometry') is not None for v in link.findall('visual')):
            # Bullet otherwise draws a restored primitive collider as an
            # opaque white visual. Keep the originally invisible link hidden.
            visual=ET.SubElement(link,'visual')
            geometry=ET.SubElement(visual,'geometry')
            ET.SubElement(geometry,'sphere',radius='0.000001')
            material=ET.SubElement(visual,'material',name='hidden_collision_proxy')
            ET.SubElement(material,'color',rgba='0 0 0 0')
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'{spec.name}_collision.urdf'
    ET.ElementTree(root).write(target,encoding='unicode')
    return target


class DemoEnv(ManipulationEnv):
    def __init__(self, robot, asset_dir, video=False):
        super().__init__(robot=robot, physics_hz=240, control_hz=30,render_observations=False)
        self.robot_spec=replace(self.robot_spec,urdf=collision_urdf(self.robot_spec,asset_dir))
        self.observation_profile={'gripper_controller':'body_contact_v9'}
        from types import SimpleNamespace
        gripper_spec=SimpleNamespace(name='robotiq_2f_85',urdf=DERIVED_ASSETS/'robotiq_2f_85/robotiq_2f_85.urdf')
        self.gripper_asset_root=asset_dir/'full_gripper'
        collision_urdf(gripper_spec,self.gripper_asset_root/'robotiq_2f_85').rename(
            self.gripper_asset_root/'robotiq_2f_85/robotiq_2f_85.urdf')
        self.demo_renderer=p.ER_TINY_RENDERER
        if video:
            spec=importlib.util.find_spec('eglRenderer')
            if spec and p.loadPlugin(spec.origin,'_eglRendererPlugin')>=0:
                self.demo_renderer=p.ER_BULLET_HARDWARE_OPENGL

    def _spawn_object(self,name,position,fixed):
        # Preserve native fruit/utensil textures; color names in the task are aliases.
        body=super()._spawn_object(name,position,fixed)
        if name.split(' ',1)[1] not in ('block','bowl','egg'):
            p.changeVisualShape(body,-1,rgbaColor=(1,1,1,1))
        return body

    def reset(self,task):
        super().reset(task)
        p.removeBody(self.gripper.body)
        dynamic=[body for body in self.object_ids.values() if p.getDynamicsInfo(body,-1)[0]>0]
        self.gripper=BodyContactRobotiq(self.robot_id,self.robot_spec.mount_link,self.gripper_asset_root,dynamic,mimic_gear=False)
        self.q=np.array([p.getJointState(self.robot_id,j)[0] for j in self.robot_spec.arm_joint_ids])
        self.q_velocity=np.zeros(6)
        for _ in range(240):
            self._apply_arm_targets(self.q)
            self.gripper.apply_motor_control(); p.stepSimulation()
        self.self_pairs=[]
        parents={i:p.getJointInfo(self.robot_id,i)[16] for i in range(p.getNumJoints(self.robot_id))}
        links=[i for i in range(-1,p.getNumJoints(self.robot_id)) if p.getCollisionShapeData(self.robot_id,i)]
        def graph_distance(a,b):
            chain_a=[a]
            while chain_a[-1]!=-1: chain_a.append(parents[chain_a[-1]])
            chain_b=[b]
            while chain_b[-1]!=-1: chain_b.append(parents[chain_b[-1]])
            return min(chain_a.index(x)+chain_b.index(x) for x in set(chain_a)&set(chain_b))
        for i,a in enumerate(links):
            for b in links[i+1:]:
                if graph_distance(a,b)>2:
                    self.self_pairs.append((a,b))
                    p.setCollisionFilterPair(self.robot_id,self.robot_id,a,b,1)
        p.changeVisualShape(self.table_id,-1,rgbaColor=(.32,.35,.38,1.))
        self.arm_gripper_pairs=[link for link in links if graph_distance(link,self.robot_spec.mount_link)>2]
        self.initial_pick_position=np.array(p.getBasePositionAndOrientation(self.pick_id)[0])
        self.initial_objects={name:np.array(p.getBasePositionAndOrientation(body)[0]) for name,body in self.object_ids.items()}
        return self.observe()

    def pose(self):
        s=p.getLinkState(self.robot_id,self.robot_spec.mount_link,computeForwardKinematics=True)
        return np.array(s[4]),np.array(s[5])

    def tick(self,center,quaternion,command):
        current,qcur=self.pose()
        local=Rotation.from_quat(qcur).inv().apply(self.gripper.grasp_center()-current)
        mount=np.asarray(center)-Rotation.from_quat(quaternion).apply(local)
        ik=np.array(p.calculateInverseKinematics(self.robot_id,self.robot_spec.mount_link,mount,
                    targetOrientation=quaternion,maxNumIterations=100,residualThreshold=1e-7)[:6])
        ik=self.q+(ik-self.q+np.pi)%(2*np.pi)-np.pi
        dt=1/self.physics_hz
        velocity=np.clip((ik-self.q)/dt,-1.15,1.15)
        velocity=np.clip(velocity,self.q_velocity-3.*dt,self.q_velocity+3.*dt)
        self.q += velocity*dt
        self.q_velocity=velocity
        p.setJointMotorControlArray(self.robot_id,self.robot_spec.arm_joint_ids,p.POSITION_CONTROL,
                targetPositions=self.q,targetVelocities=velocity,forces=[250]*6,
                positionGains=[.16]*6,velocityGains=[1.]*6)
        self.gripper.command=float(command)
        if command < -.2 and self.gripper.grasp_constraint is not None:
            self.gripper.open()
        self.gripper.apply_motor_control(); p.stepSimulation(); self.gripper.post_step()


class Rollout:
    def __init__(self,env,scene,directory,video=False,width=1280,height=720,fps=30):
        self.env=env;self.scene=scene;self.directory=directory
        self.width=width;self.height=height;self.fps=fps
        self.tick_count=0;self.phase='initial';self.samples=[];self.writers=[]
        self.contacts=[];self.touch=False;self.verified=False
        self.self_min=1.;self.arm_gripper_min=1.;self.ground_min=1.;self.max_tracking_error=0.
        self.phase_progress=0.;self.phase_frames={};self.visibility=[];self.renderer=env.demo_renderer
        if video:
            directory.mkdir(parents=True,exist_ok=True)
            for i in range(3):
                writer=imageio_ffmpeg.write_frames(str(directory/f'view_{i+1}.mp4'),(width,height),fps=fps,
                   codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error',
                   output_params=['-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart'])
                writer.send(None);self.writers.append(writer)
        self.capture()

    def capture(self):
        e=self.env
        obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
        self.samples.append({'time':self.tick_count/e.physics_hz,'phase':self.phase,
            'tcp':e.gripper.grasp_center().tolist(),'object':obj.tolist(),
            'joints':[p.getJointState(e.robot_id,j)[0] for j in e.robot_spec.arm_joint_ids],
            'gripper_width':e.gripper.width(),'bilateral':e.gripper._bilateral(e.pick_id),
            'held':e.gripper.grasped_object==e.pick_id})
        if not self.writers:return
        for i,cfg in enumerate(DEMO_VIEWS):
            view=p.computeViewMatrix(cfg['eye'],cfg['target'],cfg['up'])
            proj=p.computeProjectionMatrixFOV(cfg['fov'],self.width/self.height,.02,4.)
            _,_,rgba,_,seg=p.getCameraImage(self.width,self.height,viewMatrix=view,projectionMatrix=proj,
                 renderer=self.renderer,shadow=1,lightDirection=[-1,-.6,2],lightColor=[1,1,.98],
                 lightAmbientCoeff=.55,lightDiffuseCoeff=.65,lightSpecularCoeff=.15,
                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            rgb=np.asarray(rgba,dtype=np.uint8).reshape(self.height,self.width,4)[:,:,:3].copy()
            self.writers[i].send(rgb)
            mask=(np.asarray(seg,dtype=np.int64)&((1<<24)-1))==e.pick_id
            self.visibility.append({'time':self.tick_count/e.physics_hz,'phase':self.phase,
                'view':i+1,'visible_pick_pixels':int(mask.sum())})
            stage='start' if self.phase_progress<.1 else 'middle' if self.phase_progress<.9 else 'end'
            key=(self.phase,stage,i)
            if key not in self.phase_frames:
                import cv2
                folder=self.directory/'frames';folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder/f'{self.phase}_{stage}_view_{i+1}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                self.phase_frames[key]=True

    def step(self,center,q,command):
        e=self.env;e.tick(center,q,command);self.tick_count+=1
        self.max_tracking_error=max(self.max_tracking_error,float(np.linalg.norm(e.gripper.grasp_center()-center)))
        contacts=p.getContactPoints(bodyA=e.gripper.body,bodyB=e.pick_id)
        self.touch |= any(c[9]>.03 for c in contacts)
        self.verified |= e.gripper.grasped_object==e.pick_id and e.gripper._bilateral(e.pick_id)
        # Contacts sampled at every physics step; expected pick/place contacts are allowed.
        for body in (e.robot_id,e.gripper.body):
            for c in p.getContactPoints(bodyA=body):
                if body==e.robot_id and c[2]==e.robot_id and (min(c[3],c[4]),max(c[3],c[4])) in e.self_pairs:
                    self.self_min=min(self.self_min,c[8])
                if c[2] in (e.robot_id,e.gripper.body):continue
                allowed=[e.pick_id]
                if self.phase in ('carry','release','retract','settle'):allowed.append(e.place_id)
                if c[2] not in allowed and c[9]>.5 and c[8]<-.0008:
                    if len(self.contacts)<30:
                        self.contacts.append({'time':self.tick_count/e.physics_hz,'phase':self.phase,
                          'body':body,'other':c[2],'link':c[3],'penetration':-c[8],'force':c[9]})
        if self.tick_count%8==0:
            for a,b in e.self_pairs:
                points=p.getClosestPoints(e.robot_id,e.robot_id,.01,linkIndexA=a,linkIndexB=b)
                self.self_min=min(self.self_min,min((c[8] for c in points),default=1.))
            for link in e.arm_gripper_pairs:
                points=p.getClosestPoints(e.robot_id,e.gripper.body,.01,linkIndexA=link)
                self.arm_gripper_min=min(self.arm_gripper_min,min((c[8] for c in points),default=1.))
            points=p.getClosestPoints(e.gripper.body,e.table_id,.02)
            self.ground_min=min(self.ground_min,min((c[8] for c in points),default=1.))
        if self.tick_count%(e.physics_hz//self.fps)==0:self.capture()

    def curve(self,curve,phase,command_start=-1.,command_end=None):
        self.phase=phase
        if command_end is None:command_end=command_start
        for step in range(1,round(curve.duration*self.env.physics_hz)+1):
            elapsed=step/self.env.physics_hz
            self.phase_progress=elapsed/curve.duration
            center,q=curve.sample(elapsed)
            s=minimum_jerk(elapsed/curve.duration)
            self.step(center,q,command_start+(command_end-command_start)*s)
            if self.scene.action=='touch' and self.touch and phase in ('approach','touch'):
                break
            if phase=='grasp' and self.env.gripper.grasped_object==self.env.pick_id and self.env.gripper._bilateral(self.env.pick_id):
                break

    def hold(self,center,q,duration,phase,command_start,command_end=None):
        self.curve(PoseCurve([center,center],q,q,duration),phase,command_start,command_end)

    def close(self):
        for writer in self.writers:writer.close()

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
        bend=.008+np.random.default_rng(self.scene.config+e.robot_spec.name.__len__()).uniform(0,.012)
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
            self.curve(approach_curve(center,target,q0,q1,4.5,bend),'approach')
            if self.touch:
                target=e.gripper.grasp_center();q1=e.pose()[1]
            self.hold(target,q1,.5,'touch',-1.)
            if not self.touch:
                self.curve(PoseCurve([target,target-[0,0,.003]],q1,q1,.5),'touch')
                target-=np.array([0,0,.003])
            self.hold(target,q1,.5,'final',-1.)
            success=self.touch and np.linalg.norm(np.array(p.getBasePositionAndOrientation(e.pick_id)[0])-e.initial_pick_position)<.025
        else:
            width=plan.get('cross_section_width_m',p.getAABB(e.pick_id)[1][0]-p.getAABB(e.pick_id)[0][0])
            desired_gap=np.clip(width+.022,.050,.0885)
            preshape=float(np.clip(2*(1-(desired_gap-.0064)/.0822)-1,-1.,.1))
            if e.task.pick_name.endswith(('knife','spoon','spatula')):
                preshape=-1.
            self.curve(approach_curve(center,target,q0,q1,4.5,bend),'approach',-1.,preshape)
            self.hold(target,q1,.35,'align',preshape)
            self.hold(target,q1,1.4,'grasp',preshape,1.)
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
            clearance=max(p.getAABB(body)[1][2] for body in e.object_ids.values())+.16
            self.curve(carry_curve(center,target,q1,clearance,5.5,bend),'carry',1.)
            self.hold(target,q1,.6,'place' if self.scene.action=='put' else 'hover',1.)
            if self.scene.action=='put':
                self.hold(target,q1,.9,'release',1.,-1.)
                retract=target+[0,0,.15]
                self.curve(PoseCurve([target,target+[.01,0,.06],retract],q1,q1,2.),'retract',-1.)
                self.hold(retract,q1,1.5,'settle',-1.)
                obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0])
                support=bool(p.getContactPoints(bodyA=e.pick_id,bodyB=e.place_id))
                velocity=np.linalg.norm(p.getBaseVelocity(e.pick_id)[0])
                success=e.gripper.grasped_object is None and support and np.linalg.norm(obj[:2]-goal[:2])<.045 and velocity<.025
            else:
                self.hold(target,q1,.7,'final',1.)
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
            'task':e.task.to_dict()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robots',default='uf850,ur5e')
    parser.add_argument('--actions',default='touch,pick,move,put')
    parser.add_argument('--configs',default='1,2,3,4')
    parser.add_argument('--config-file',type=Path)
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--fps',type=int,choices=[30,60],default=30)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    selected=[parse_config(args.config_file)] if args.config_file else scenes()
    results=[]
    for robot in args.robots.split(','):
        with DemoEnv(robot,args.output/'assets',not args.no_video) as env:
            for scene in selected:
                if scene.action not in args.actions.split(',') or str(scene.config) not in args.configs.split(','):continue
                folder=args.output/('UF850' if robot=='uf850' else 'UR5e_Robotiq2F')/scene.action/f'config{scene.config}'
                folder.mkdir(parents=True,exist_ok=True)
                rollout=None
                try:
                    env.reset(scene.task())
                    rollout=Rollout(env,scene,folder,not args.no_video,args.width,args.height,args.fps)
                    result=rollout.run()
                except Exception as exc:
                    result={'robot':robot,'action':scene.action,'config':scene.config,'success':False,
                            'overall_pass':False,'error':str(exc)}
                finally:
                    if rollout:
                        rollout.close()
                        (folder/'trajectory.json').write_text(json.dumps(rollout.samples))
                        if rollout.writers:
                            (folder/'visibility.json').write_text(json.dumps(rollout.visibility))
                (folder/'audit.json').write_text(json.dumps(result,indent=2))
                results.append(result)
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({k:v for k,v in result.items() if k in ('robot','action','config','success','overall_pass','error','self_min_distance','gripper_table_min_distance','unexpected_contacts','max_tracking_error')}),flush=True)
    return 0 if all(r['overall_pass'] for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
