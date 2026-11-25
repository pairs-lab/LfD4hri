"""Explicit scripted Oracle demonstrations on the exact paired VLA plant.

Faults change controller commands, NOT physics, objects, camera or VLA actions.
They are controlled illustrations, never native learned-RL failure evidence.
"""
import argparse
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.qualitative_recorder_r51 import physical_state,canonical_hash
from VLA.qualitative_oracle_shared_r51 import SharedFeedbackRollout,bind_executor
from .common import make_env,reset_put,read
from .capture import EvidenceRecorder


class TimeScaledCurve:
    def __init__(self,curve,scale):self.curve=curve;self.scale=scale;self.duration=curve.duration*scale
    def sample(self,t):return self.curve.sample(t/self.scale)


class ScriptedOracle(SharedFeedbackRollout):
    def __init__(self,*args,fault='none',**kwargs):
        self.fault=fault;self.fault_events=[];self.drop_started=False
        super().__init__(*args,**kwargs)
    def step(self,center,q,command):
        env=self.env
        if (self.fault=='drop_in_transit' and self.phase=='carry' and not self.drop_started
            and self.phase_progress>=.30 and env.gripper.grasped_object==env.pick_id
            and p.getBasePositionAndOrientation(env.pick_id)[0][2]>env.initial_pick_z+.04):
            self.drop_started=True
            self.fault_events.append({'type':'scripted_open_in_transit','time_s':self.tick_count/env.physics_hz,
                'phase':self.phase,'progress':self.phase_progress,'command_override':-1.})
        if self.drop_started:command=-1.
        if self.fault=='approach_overshoot' and self.phase=='approach' and self.phase_progress>.62:
            center=np.asarray(center).copy()
            center[0]+=.065*min((self.phase_progress-.62)/.16,1.)
            center[2]-=.008*min((self.phase_progress-.62)/.16,1.)
            if not self.fault_events:self.fault_events.append({'type':'scripted_approach_target_error',
                'time_s':self.tick_count/env.physics_hz,'maximum_x_offset_m':.065,'maximum_down_offset_m':.008,
                'impact_or_slam_NOT_assumed_without_contact_evidence':True})
        super().step(center,q,command)
    def curve(self,curve,phase,*args,**kwargs):
        if self.fault=='slow_transport_timeout' and phase=='carry':
            start=curve.sample(0)[0];end=curve.sample(curve.duration)[0]
            distance=float(np.linalg.norm(np.asarray(end)-start))
            duration=max(curve.duration,distance/.012)
            curve=TimeScaledCurve(curve,duration/curve.duration)
            self.fault_events.append({'type':'scripted_transport_speed_limit','distance_m':distance,
                'nominal_path_duration_s':duration,'maximum_nominal_speed_m_per_s':.012,
                'timeout_will_be_measured_at_shared_physical_budget':True})
        return super().curve(curve,phase,*args,**kwargs)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    expected=read(a.paired_capture/'initial_scene.json')
    env=make_env(a.seed);old_step=p.stepSimulation;old_contacts=p.getContactPoints
    recorder=None;ticks=0;rollout=None
    try:
        task,obs=reset_put(env,a.seed)
        state=physical_state(env);digest=canonical_hash(state)
        if digest!=expected['measured_state_sha256'] or canonical_hash(task.to_dict())!=canonical_hash(expected['task']):
            raise ValueError('Scripted Oracle initial world differs from VLA')
        atomic_json(a.output/'initial_scene.json',{'task':task.to_dict(),'seed':a.seed,
            'measured_state':state,'measured_state_sha256':digest})
        bind_executor(env)
        if canonical_hash(physical_state(env))!=digest:raise ValueError('Oracle API adapter mutated initial physical state')
        original_top=env._body_top
        def top(self,body):
            if body is None:return np.r_[task.place_position[:2],p.getAABB(self.table_id)[1][2]]
            return original_top(body)
        env._body_top=MethodType(top,env)
        # Fix the old Oracle's unsupported bodyB=None support QUERY for table
        # destinations. This is read-only; no fake contact or body is created.
        def contacts(*args,**kwargs):
            if kwargs.get('bodyA')==env.pick_id and 'bodyB' in kwargs and kwargs['bodyB'] is None:
                return tuple(c for body in env.ground_bodies for c in old_contacts(bodyA=env.pick_id,bodyB=body))
            return old_contacts(*args,**kwargs)
        p.getContactPoints=contacts
        scene=SimpleNamespace(action='put',config=a.seed,pick=task.pick_name,place=task.place_name)
        rollout=ScriptedOracle(env,scene,a.output,video=False,fault=a.fault)
        recorder=EvidenceRecorder(env,a.output);recorder.phase='scripted_feedback_Oracle_NOT_trained_RL'
        original_post=env.gripper.post_step
        budget_ticks=round(22*env.physics_hz)
        def counted(*args,**kwargs):
            nonlocal ticks
            if ticks>=budget_ticks:raise TimeoutError('Shared22s qualitative physical-time budget reached')
            value=old_step(*args,**kwargs);ticks+=1;return value
        def post():
            try:return original_post()
            finally:
                recorder.record_tick(ticks,ticks//env.substeps)
                if ticks%8==0:
                    recorder.capture(ticks,ticks//env.substeps)
                    recorder.frames[-1]['oracle_phase']=rollout.phase
        p.stepSimulation=counted;env.gripper.post_step=post
        atomic_json(a.output/'protocol.json',{'method':'Scripted feedback Oracle, NOT learned RL',
            'fault':a.fault,'faults_change_controller_NOT_initial_scene_or_plant':True,
            'shared_initial_state_sha256':digest,'same_assets_home_solver_safety_and_camera':True,
            'asymmetric_Oracle_fault_NOT_symmetric_robustness_comparison':a.fault!='none',
            'support_None_API_query_fixed_without_physics_changes':True,
            'common_qualitative_maximum_seconds':22,'no_external_forces_or_teleports_or_safety_disable':True})
        try:result=rollout.run();error=None
        except Exception as exc:result={'success':False};error=f'{type(exc).__name__}: {exc}'
        atomic_json(a.output/'outcome.json',{'controller_result':result,'error':error,'fault_script':a.fault,
            'fault_events':rollout.fault_events,'physics_ticks':ticks,'duration_s':ticks/env.physics_hz,
            'controller_reported_success_NOT_independent_put_audit':result.get('success',False)})
        atomic_json(a.output/'trajectory.json',rollout.samples)
        atomic_json(a.output/'physics_commands.json',env.physics_commands)
        atomic_json(a.output/'status.json',{'status':'completed_CPU_capture_REQUIRES_independent_content_and_media_audit',
            'fault':a.fault,'error':error,'controller_reported_success':result.get('success',False)})
        print({'fault':a.fault,'seed':a.seed,'error':error,'controller_result_success':result.get('success')},flush=True)
    finally:
        p.stepSimulation=old_step;p.getContactPoints=old_contacts
        if recorder is not None:recorder.close()
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--paired-capture',type=Path,required=True)
    parser.add_argument('--fault',choices=('none','drop_in_transit','approach_overshoot','slow_transport_timeout'),default='none')
    main(parser.parse_args())
