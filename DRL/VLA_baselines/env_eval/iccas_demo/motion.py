"""Continuous physical Oracle execution, including real release and home return."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from VLA.collision_grasp_planner import plan_body_grasp,wrap_axis_angle
from env_eval.oracle_feedback import FeedbackRollout
from env_eval.feedback_motion import approach_chunk,transport_chunk,retract_chunk,MotionProfile
from env_eval.oracle_trajectory import PoseCurve


class RequestedRollout(FeedbackRollout):
    def __init__(self,env,spec,directory,profile=None):
        self.spec=spec;self.task_index=0;self.frames=[];self.subtasks=[];self.home_checks=[]
        self.home_tcp=env.gripper.grasp_center().copy();self.home_mount,self.home_quat=env.pose()
        self.home_q=np.array([p.getJointState(env.robot_id,j)[0] for j in env.robot_spec.arm_joint_ids])
        self.tracked=[('robot',env.robot_id),('gripper',env.gripper.body),*env.object_ids.items()]
        self.initial_geometry={name:[list(v) for v in p.getAABB(body)] for name,body in env.object_ids.items()}
        step=spec.steps[0]
        scene=SimpleNamespace(action=step.action,config=spec.sid,pick=step.pick,place=step.place)
        super().__init__(env,scene,directory,False,1920,1080,30,profile=profile or MotionProfile())

    def hold(self,center,q,duration,phase,command_start,command_end=None):
        if phase=='release':return super().hold(center,q,duration,phase,command_start,command_end)
        if command_end is None:command_end=command_start
        center=np.asarray(center)+(self.target_feedback if phase=='grasp' else 0)
        # Constant poses have no interpolation domain. Avoid the old SLERP's
        # >1 by a few ulps at preview endpoints; do not alter legacy sources.
        self.curve(SimpleNamespace(duration=duration,sample=lambda elapsed:(center,np.asarray(q))),
            phase,command_start,command_end)

    def capture(self):
        super().capture()
        if self.initializing:return
        e=self.env;sample=self.samples[-1]
        sample.update(task_index=self.task_index,target_name=e.task.pick_name,goal_name=e.task.place_name,
            objects={name:{'position':list(p.getBasePositionAndOrientation(b)[0]),
                'quaternion':list(p.getBasePositionAndOrientation(b)[1]),
                'velocity':list(p.getBaseVelocity(b)[0])} for name,b in e.object_ids.items()})
        self.frames.append({'base':np.array([(*p.getBasePositionAndOrientation(b)[0],
            *p.getBasePositionAndOrientation(b)[1]) for name,b in self.tracked]),
            'robot_joints':np.array([s[0] for s in p.getJointStates(e.robot_id,range(p.getNumJoints(e.robot_id)))]),
            'gripper_joints':np.array([s[0] for s in p.getJointStates(e.gripper.body,range(p.getNumJoints(e.gripper.body)))])})

    def bind_task(self,index):
        e=self.env;step=self.spec.steps[index];task=self.spec.task(index)
        self.task_index=index;self.scene=SimpleNamespace(action=step.action,config=self.spec.sid,
            pick=step.pick,place=step.place)
        e.pick_id=e.object_ids[step.pick];e.place_id=e.object_ids.get(step.place)
        pos=np.asarray(p.getBasePositionAndOrientation(e.pick_id)[0])
        place=tuple(p.getBasePositionAndOrientation(e.place_id)[0]) if e.place_id is not None else None
        e.task=replace(task,pick_position=tuple(pos),place_position=place)
        e.initial_pick_position=pos.copy();e.initial_pick_z=pos[2]
        e.initial_pick_top=e._pick_top();e.initial_pick_bottom=e._pick_bottom()
        e.initial_place_top=e._body_top(e.place_id) if e.place_id is not None else None
        e.initial_objects={name:np.array(p.getBasePositionAndOrientation(b)[0]) for name,b in e.object_ids.items()}
        e.gripper.pregrasp_object=e.pick_id;e.touch_reference=None
        self.target_feedback=np.zeros(3);self.grasp_target=None;self.approach_close_elapsed=None
        self.touch=False;self.verified=False;self.contacts=[]

    def return_home(self):
        e=self.env;start=e.gripper.grasp_center();q=e.pose()[1]
        max_top=max(p.getAABB(b)[1][2] for b in e.object_ids.values())
        height=max(start[2]+.10,self.home_tcp[2]+.06,max_top+.24)
        duration=4.5 if e.robot_spec.name=='uf850' else 3.0
        self.curve(transport_chunk(start,self.home_tcp,q,self.home_quat,height,duration,
            wrist=0.),'home_return',-1.)
        self.hold(self.home_tcp,self.home_quat,.75,'perceive_home',-1.)
        q_actual=np.array([p.getJointState(e.robot_id,j)[0] for j in e.robot_spec.arm_joint_ids])
        error=float(np.linalg.norm(e.gripper.grasp_center()-self.home_tcp))
        joints=float(np.max(abs((q_actual-self.home_q+np.pi)%(2*np.pi)-np.pi)))
        rotation=float(np.degrees((Rotation.from_quat(self.home_quat).inv()*Rotation.from_quat(e.pose()[1])).magnitude()))
        self.home_checks.append({'after_task':self.task_index,'time':self.tick_count/e.physics_hz,
            'tcp_error_m':error,'max_joint_error_rad':joints,'orientation_error_deg':rotation,
            'passed':error<.012 and joints<.065 and rotation<3.})
        if not self.home_checks[-1]['passed']:raise RuntimeError('Measured home return not within tolerance')

    def release_audit(self):
        e=self.env;lo,hi=map(np.asarray,p.getAABB(e.pick_id));a,b=map(np.asarray,p.getAABB(e.place_id))
        obj=(lo+hi)/2;destination=(a+b)/2
        if self.scene.place=='blue cup':
            # Cup's external handle must not move the opening centre.
            destination[:2]=np.array(p.getBasePositionAndOrientation(e.place_id)[0])[:2]
        xy=float(np.linalg.norm(obj[:2]-destination[:2]));speed=float(np.linalg.norm(p.getBaseVelocity(e.pick_id)[0]))
        support=bool(p.getContactPoints(bodyA=e.pick_id,bodyB=e.place_id))
        container=self.scene.place.endswith((' bowl',' cup'))
        contained=xy<(.027 if container else .014)
        bottom_inside=lo[2]<b[2]-.001 if container else abs(lo[2]-b[2])<.005
        return {'released':e.gripper.grasped_object is None,'support_contact':support,
            'center_error_xy_m':xy,'object_bottom_z_m':float(lo[2]),'destination_rim_or_top_z_m':float(b[2]),
            'bottom_inside_or_stacked':bool(bottom_inside),'speed_m_s':speed,
            'passed':bool(e.gripper.grasped_object is None and support and contained and bottom_inside and speed<.018)}

    def execute_task(self):
        e=self.env;action=self.scene.action;start_sample=len(self.samples)
        center=e.gripper.grasp_center();q0=e.pose()[1]
        plan=plan_body_grasp(e,e.pick_id) if action!='touch' else {'point_world':e._pick_top()}
        if e.task.pick_name=='green chalk_brush':
            # The 8-vertex primitive fallback has no caliper axis. Pinch the
            # short 42 mm side, not its 62 mm length, leaving real jaw-opening
            # clearance for release instead of dragging a long edge on a pad.
            lo,hi=map(np.asarray,p.getAABB(e.pick_id))
            plan={'point_world':(lo+hi)/2,'jaw_axis_world':np.array([0.,1.,0.]),
                'cross_section_width_m':float(hi[1]-lo[1]),'method':'known_trapezoid_minimum_width_body_pinch'}
        q1=np.array(p.getQuaternionFromEuler([0,np.pi,0]));axis=plan.get('jaw_axis_world')
        if axis is not None:
            yaw=wrap_axis_angle(np.arctan2(axis[1],axis[0]));q1=(Rotation.from_euler('z',yaw)*Rotation.from_quat(q1)).as_quat()
        jaw=Rotation.from_quat(q1).apply([-1.,0.,0.]);jaw[2]=0;jaw/=np.linalg.norm(jaw)
        self.macro_axis=jaw;self.macro_normal=np.array([-jaw[1],jaw[0],0.])
        if self.macro_normal[1]>0:self.macro_normal=-self.macro_normal
        target=np.asarray(plan['point_world']).copy()
        seconds=(5.5 if e.robot_spec.name=='uf850' else 3.5)*self.profile.approach_scale
        if action=='touch':
            local=Rotation.from_quat(q0).inv().apply(e.gripper.world_pad_vertices()[0]-center)
            vertices=Rotation.from_quat(q1).apply(local)
            tip=vertices[vertices[:,2]<=vertices[:,2].min()+.001].mean(axis=0)
            target=e._pick_top()-tip+[0,0,-.0004]
            self.curve(approach_chunk(center,target,q0,q1,seconds),'approach',-1.)
            if not self.touch:
                self.curve(retract_chunk(target,target-[0,0,.003],q1,.5),'touch_search',-1.)
            target=e.gripper.grasp_center();q1=e.pose()[1]
            e.touch_reference={'mount':e.pose()[0],'tcp':target.copy()}
            self.hold(target,q1,.35,'touch',-1.)
            self.curve(retract_chunk(target,target+[0,0,.08],q1,1.),'touch_retract',-1.)
            e.touch_reference=None
            result={'success':bool(self.touch),'physical_contact':bool(self.touch)}
        else:
            target[2]=max(target[2],p.getAABB(e.table_id)[1][2]+.021)
            lo,hi=map(np.asarray,p.getAABB(e.pick_id))
            width=plan.get('cross_section_width_m',float(hi[0]-lo[0]))
            desired_gap=np.clip(width+.022,.050,.0885)
            preshape=float(np.clip(2*(1-(desired_gap-.0064)/.0822)-1,-1.,.1))
            self.grasp_target=target.copy()
            self.curve(approach_chunk(center,target,q0,q1,seconds),'approach',-1.,preshape)
            if e.gripper.grasped_object!=e.pick_id:
                self.hold(target,q1,.8,'grasp',self.current_command,1.)
            if not (e.gripper.grasped_object==e.pick_id and e.gripper._bilateral(e.pick_id)):
                raise RuntimeError('No physical verified bilateral target grasp')
            obj=np.array(p.getBasePositionAndOrientation(e.pick_id)[0]);center=e.gripper.grasp_center()
            q_contact=e.pose()[1]
            if action=='pick':
                target=center+[0,0,.15]
                self.curve(transport_chunk(center,target,q_contact,q1,target[2],3.,lift=True),'carry',1.)
                self.hold(target,q1,1.2,'final',1.)
                lift=float(p.getBasePositionAndOrientation(e.pick_id)[0][2]-e.initial_pick_position[2])
                result={'success':bool(lift>.13 and e.gripper.grasped_object==e.pick_id and e.gripper._bilateral(e.pick_id)),
                    'lift_m':lift,'verified_bilateral_grasp':bool(self.verified)}
            else:
                # Release ABOVE the opening, with pad/rim clearance, then let
                # actual gravity and contacts place the object. No attachment
                # or object pose reset after release, including multi-task scenes.
                low,high=map(np.asarray,p.getAABB(e.pick_id));a,b=map(np.asarray,p.getAABB(e.place_id))
                destination=(a+b)/2
                if self.scene.place=='blue cup':destination[:2]=np.array(p.getBasePositionAndOrientation(e.place_id)[0])[:2]
                goal=obj.copy();goal[:2]+=destination[:2]-(low[:2]+high[:2])/2
                goal[2]=b[2]+(obj[2]-low[2])+.018
                target=center+goal-obj
                target[2]=max(target[2],b[2]+.060)
                clearance=self.corridor_clearance(center,target)+.035
                self.curve(transport_chunk(center,target,q_contact,q1,clearance,4.2,wrist=.015),'carry',1.)
                # A small tracking hold precedes opening; do not release during
                # the transport controller's braking transient.
                self.hold(target,q1,.45,'place_align',1.)
                self.hold(target,q1,.75,'release',1.,-1.)
                # An opened encoder is not proof of physical release. A thin
                # trapezoid can remain friction-wedged against ONE pad. Let the
                # jaws reach full stroke, then recenter the open gap around the
                # measured object before lifting; never remove contact forces,
                # disable collisions, apply a release impulse or move the object.
                contact_force=lambda:sum(c[9] for c in p.getContactPoints(e.gripper.body,e.pick_id))
                if contact_force()>.2:
                    self.hold(e.gripper.grasp_center(),e.pose()[1],.65,'settle',-1.)
                    for _ in range(3):
                        if contact_force()<.2:break
                        low,high=map(np.asarray,p.getAABB(e.pick_id));start=e.gripper.grasp_center();q=e.pose()[1]
                        recenter=start.copy();recenter[:2]+=np.clip((low[:2]+high[:2])/2-start[:2],-.020,.020)
                        self.curve(retract_chunk(start,recenter,q,.7),'release',-1.)
                        self.hold(recenter,q,.6,'settle',-1.)
                    if contact_force()>.2:raise RuntimeError('Actual pad contact persists after full-open release recentering')
                start=e.gripper.grasp_center();q=e.pose()[1]
                self.curve(retract_chunk(start,start+[0,0,.13],q,1.5),'retract',-1.)
                self.hold(start+[0,0,.13],q,1.3,'settle',-1.)
                release=self.release_audit();result={'success':release['passed'],'release':release,
                    'verified_bilateral_grasp':bool(self.verified)}
        moved={name:float(np.linalg.norm(np.asarray(p.getBasePositionAndOrientation(body)[0])-e.initial_objects[name]))
            for name,body in e.object_ids.items() if body not in (e.pick_id,e.place_id)}
        result.update(task_index=self.task_index,command=e.task.prompt,action=action,
            distractor_displacement=moved,unexpected_contacts=list(self.contacts),
            plan={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in plan.items()},
            frames=(start_sample,len(self.samples)),time_s=self.tick_count/e.physics_hz,
            safety_pass=not self.contacts and max(moved.values(),default=0)<.02)
        self.subtasks.append(result)
        if not result['success']:raise RuntimeError('Physical task outcome failed: '+str(result))
        if not result['safety_pass']:raise RuntimeError('Unintended physical collision or distractor displacement')

    def run_sequence(self):
        self.capture()
        for index,step in enumerate(self.spec.steps):
            self.bind_task(index)
            # An actual held home pose before each new subtask is included in
            # the continuous timeline; no resetSimulation() between subtasks.
            if index:self.hold(self.home_tcp,self.home_quat,.6,'perceive_home',-1.)
            self.execute_task()
            if step.action!='pick':self.return_home()
        return {'success':True,'overall_pass':True,'subtasks':self.subtasks,'home_checks':self.home_checks,
            'frames':len(self.samples),'duration_s':self.tick_count/self.env.physics_hz,
            'self_min_m':self.self_min,'arm_gripper_min_m':self.arm_gripper_min,
            'gripper_ground_min_m':self.ground_min,'max_tracking_error_m':self.max_tracking_error,
            'controller':'privileged feedback Oracle, not trained RL or VLA',
            'no_object_or_joint_teleport_during_execution':True}
