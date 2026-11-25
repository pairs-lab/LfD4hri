"""Re-observe native joint corrective commands in the ACTUAL shared servo.

The old failed native-command parity proof is never waived. RGB/proprio/64
feedback are freshly recorded on the native execution path instead of copying
Cartesian-branch images/states. Only offline supervised labels are produced;
this never becomes a VLA evaluation fallback. All original40 attempts remain
preserved, including the two successful unlabeled student prefixes.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.tasks import TaskSpec
from VLA.collect_demos import collect_episode
from VLA.dataset_replay_context import audit_environment,reset_source_episode
from VLA.evaluate_smolvla_v9_source import reset_observation_errors,validate_reset_observation
from VLA.simstate_scene64 import scene_state
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


class NativeCapture:
    def __init__(self,env,source,original,row,native,closed):
        self.env,self.source,self.original,self.row=env,source,original,row
        self.native,self.closed=native,closed;self.states=[];self.requests=[];self.servo=[];self.history=[];self.cursor=0
        self.wrapper=NearSurfaceTouchCollection(env,None,None) if row['task']['name']=='touch' else env
    def __getattr__(self,name):return getattr(self.env,name)
    def reset(self,task):
        if task!=TaskSpec.from_dict(self.row['task']):raise ValueError('Original instruction changed')
        obs=reset_source_episode(self.env,self.source,self.original)
        self.previous_rgb=(obs.front_rgb.copy(),obs.wrist_rgb.copy())
        for action in self.row['recovery_start_actions_not_training_labels']:
            self.previous_rgb=(obs.front_rgb.copy(),obs.wrist_rgb.copy())
            obs,_,done,info=self.env.step(np.asarray(action,np.float32))
            if done or info.get('safety_violation'):raise ValueError('Original saved student prefix changed')
        np.testing.assert_allclose(scene_state(self.env,obs),self.row['initial_scene64'],rtol=0,atol=1e-6)
        self.env.step_count,self.env.success_streak,self.env._last_success=0,0,False
        return obs
    def step(self,action):
        current=self.env.observe()
        self.history.append(tuple(v.copy() for v in self.previous_rgb))
        self.previous_rgb=(current.front_rgb.copy(),current.wrist_rgb.copy())
        self.states.append(scene_state(self.env,current));self.requests.append(np.asarray(action,np.float32).copy())
        obs,reward,done,info=self.wrapper.step(action);self.servo.append(self.env.requested_arm_targets.copy());self.cursor+=1
        if info.get('safety_violation') or info.get('expert_quality_failure'):raise ValueError('Native corrective path unsafe')
        # Keep the whole predeclared native sequence, including any early
        # first-hit success. No trimming to artificially convenient paths.
        if self.cursor<len(self.native):done=False
        elif self.cursor==len(self.native):
            if not info['strict_raw_success']:raise ValueError('Native terminal strict task predicate failed')
            if self.row['task']['name']=='touch':
                if not self.wrapper._last_success:raise ValueError('Original gentle touch quality failed')
            elif self.env.gripper.grasped_object!=self.env.pick_id or not self.env.gripper._bilateral(self.env.pick_id):
                raise ValueError('Correct bilateral body grasp missing')
            done=True
        else:raise ValueError('No synthetic extension of corrective labels')
        return obs,reward,done,info


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('collect','replay'))
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true');a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    original_plan=json.loads((a.candidate/'plan.json').read_text());index=json.loads((a.candidate/'index.json').read_text())
    source=Path(original_plan['source']);originals={r['file']:r for r in json.loads((source/'index.json').read_text())}
    if a.mode=='collect':
        if len(index)!=40 or [r['source_file'] for r in index]!=original_plan['selected_files']:
            raise ValueError('Preserve the original40 predeclared TRAIN attempts')
        selected=[r for r in index if r.get('file')]
        if a.pilot:selected=[next(r for r in selected if r['task']['name']==t) for t in ('touch','pick','move','pick_place')]
        plan={'source':str(source),'selected_files':[r['source_file'] for r in selected],'pilot_only':a.pilot,
            'parent_all40_attempts':str(a.candidate.resolve()),'parent_index_sha256':sha256_file(a.candidate/'index.json'),
            'native_action_kind':'policy_requested_absolute_q_plus_requested_closure_through_shared_fk_ik_servo',
            'source_images_or_states_copied':False,'prefix_actions_supervised':False,'candidate_only':True,
            'real_previous_camera_at_student_handoff':True,
            'collector_sha256':sha256_file(Path(__file__))}
        atomic_json(a.output/'plan.json',plan);reports=[]
        for number,row in enumerate(selected):
            old=a.candidate/row['file']
            if sha256_file(old)!=row['payload_sha256']:raise ValueError('Source corrective labels changed')
            with np.load(old,allow_pickle=False) as z:
                # Exact typed boundary used by every actual neural learner.
                native=np.asarray(z['next_joint_targets'],np.float32);closed=np.asarray(native_gripper_closure(z),np.float32)
            result=copy.deepcopy(row);result.pop('file',None);result.update(success=False,payload_sha256=None,error=None)
            env=audit_environment(source,image_size=384,control_hz=10,seed=0,required_success_steps=1,use_semantic_success=False)
            capture=NativeCapture(env,source,originals[row['source_file']],row,native,closed)
            class Sequence:
                def __init__(self,_env):pass
                def reset(self,_task):self.cursor=0
                def predict(self,_obs):
                    i=self.cursor;self.cursor+=1
                    return env.joint_targets_to_canonical(native[i],float(closed[i]))
            try:
                arrays,meta=collect_episode(capture,TaskSpec.from_dict(row['task']),policy_factory=Sequence)
                if len(arrays['canonical_actions'])!=len(native):raise ValueError('Native corrective timeline truncated')
                arrays['audit_applied_servo_joint_targets']=np.asarray(capture.servo)
                arrays['audit_requested_canonical_before_safety']=np.asarray(capture.requests,np.float32)
                arrays['audit_previous_front_rgb']=np.asarray([v[0] for v in capture.history],np.uint8)
                arrays['audit_previous_wrist_rgb']=np.asarray([v[1] for v in capture.history],np.uint8)
                arrays['next_joint_targets']=native.copy() # output command ACTUALLY used by this native rollout
                arrays['next_requested_gripper_closure']=closed.copy()
                arrays['audit_pre_action_PRIVILEGED_state64']=np.asarray(capture.states,np.float32)
                arrays['audit_pre_action_PRIVILEGED_state18']=arrays['audit_pre_action_PRIVILEGED_state64'][:,:18].copy()
                np.testing.assert_allclose(arrays['joint_positions'],arrays['audit_pre_action_PRIVILEGED_state64'][:,:6],rtol=0,atol=1e-6)
                path=a.output/('native_'+row['file']);np.savez_compressed(path,**arrays)
                folder=a.output/f'case{number:03d}';folder.mkdir()
                for role in ('front','wrist'):
                    for phase,i in [('initial',0),('middle',len(native)//2)]:Image.fromarray(arrays[role+'_rgb'][i]).save(folder/f'{phase}_{role}.png')
                    Image.fromarray(getattr(env.observe(),role+'_rgb')).save(folder/f'terminal_{role}.png')
                result.update(file=path.name,payload_sha256=sha256_file(path),success=True,steps=len(native),
                    native_action_kind=plan['native_action_kind'],fresh_native_observations=True,native_record_metadata=meta)
            except Exception as error:result.update(error=f'{type(error).__name__}: {error}',success=False)
            finally:env.close()
            reports.append(result);atomic_json(a.output/'index.json',reports)
            atomic_json(a.output/'progress.json',{'completed_attempts':len(reports),'planned_attempts':len(selected),
                'expert_native_path_successes_NOT_VLA_SR':sum(bool(r['success']) for r in reports),'training_qualified':False,'rows':reports})
        atomic_json(a.output/'collection_audit.json',{'passed':all(r['success'] for r in reports),'pilot_only':a.pilot,
            'rows':reports,'unlabeled_original_attempts_preserved':[r['source_file'] for r in index if not r.get('file')],
            'training_qualified':False,'neural_model_success_rate':None})
    else:
        if a.pilot:raise ValueError('Replay inherits its collection coverage; do not filter')
        reports=[]
        for row in index:
            result={'file':row.get('file'),'passed':False,'error':None}
            if not row.get('file') or not row['success']:raise ValueError('Do not filter failed native collection rows')
            if sha256_file(a.candidate/row['file'])!=row['payload_sha256']:raise ValueError('Native observations changed')
            env=audit_environment(source,image_size=384,control_hz=10,seed=0,required_success_steps=1,use_semantic_success=False)
            try:
                with np.load(a.candidate/row['file'],allow_pickle=False) as z:
                    native=np.asarray(z['next_joint_targets'],np.float32);closed=native_gripper_closure(z)
                    capture=NativeCapture(env,source,originals[row['source_file']],row,native,closed)
                    obs=capture.reset(TaskSpec.from_dict(row['task']));maximum=0.
                    for frame in range(len(native)):
                        for camera,previous in zip(('front','wrist'),capture.previous_rgb,strict=True):
                            np.testing.assert_array_equal(previous,z['audit_previous_'+camera+'_rgb'][frame])
                        validate_reset_observation(reset_observation_errors(obs,env.normalized_gripper_closure(),
                            {k:z[k][frame:frame+1] for k in ('joint_positions','gripper_closure','front_rgb','wrist_rgb')}))
                        error=float(np.max(np.abs(scene_state(env,obs)-z['audit_pre_action_PRIVILEGED_state64'][frame])))
                        maximum=max(maximum,error)
                        if error>1e-6:raise ValueError('Native causal feedback timeline changed')
                        action=env.joint_targets_to_canonical(native[frame],float(closed[frame]))
                        np.testing.assert_array_equal(np.asarray(action,np.float32),z['audit_requested_canonical_before_safety'][frame])
                        obs,_,_,_=capture.step(action)
                result.update(passed=True,frames=len(native),causal_state_max_error=maximum)
            except Exception as error:result['error']=f'{type(error).__name__}: {error}'
            finally:env.close()
            reports.append(result);atomic_json(a.output/'progress.json',{'rows':reports})
        atomic_json(a.output/'audit.json',{'passed':all(r['passed'] for r in reports),'rows':reports,
            'scope':'fresh_native_corrective_observations_exact_replay_NOT_model_SR','training_qualified':False,
            'real_previous_camera_at_student_handoff':True,'collector_sha256':sha256_file(Path(__file__)),
            'parent_collection_sha256':sha256_file(a.candidate/'collection_audit.json')})
        if not all(r['passed'] for r in reports):raise ValueError('Native replay failed; do not admit into fitting')


if __name__=='__main__':main()
