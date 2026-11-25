"""Full native replay using the ALREADY frozen current/previous RGB limits.

The older failed bit-exact history proof is preserved. Previous and current
cameras now use the SAME existing 1-LSB /0.1%-pixels rounding gate. All
physical state, command, safety and task checks are unchanged.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.collect_native_recovery_view import NativeCapture
from VLA.evaluate_smolvla_v9_source import reset_observation_errors,validate_reset_observation
from VLA.simstate_scene64 import scene_state

SCOPE='fresh_native_corrective_observations_replay_with_frozen_RGB_rounding_NOT_model_SR'


def replay(candidate,output):
    candidate,output=Path(candidate).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((candidate/'plan.json').read_text());rows=json.loads((candidate/'index.json').read_text())
    producer=Path(__file__).with_name('collect_native_recovery_view.py')
    if sha256_file(producer)!=plan['collector_sha256']:raise ValueError('Actual native producer changed')
    source=Path(plan['source']);originals={r['file']:r for r in json.loads((source/'index.json').read_text())};reports=[]
    for row in rows:
        result={'file':row['file'],'passed':False,'error':None}
        if not row['success'] or sha256_file(candidate/row['file'])!=row['payload_sha256']:
            raise ValueError('Do not filter failures or altered data')
        env=audit_environment(source,image_size=384,control_hz=10,seed=0,required_success_steps=1,use_semantic_success=False)
        try:
            with np.load(candidate/row['file'],allow_pickle=False) as z:
                native=np.asarray(z['next_joint_targets'],np.float32);closed=native_gripper_closure(z)
                capture=NativeCapture(env,source,originals[row['source_file']],row,native,closed)
                obs=capture.reset(TaskSpec.from_dict(row['task']));maximum=servo_error=0.;pixels=[]
                for frame in range(len(native)):
                    recorded={k:z[k][frame:frame+1] for k in ('joint_positions','gripper_closure','front_rgb','wrist_rgb')}
                    actual=reset_observation_errors(obs,env.normalized_gripper_closure(),recorded)
                    validate_reset_observation(actual)
                    history=SimpleNamespace(joint_positions=obs.joint_positions,
                        front_rgb=capture.previous_rgb[0],wrist_rgb=capture.previous_rgb[1])
                    previous={**recorded,**{c+'_rgb':z['audit_previous_'+c+'_rgb'][frame:frame+1] for c in ('front','wrist')}}
                    previous_errors=reset_observation_errors(history,env.normalized_gripper_closure(),previous)
                    validate_reset_observation(previous_errors)
                    pixels.append({'frame':frame,'current':actual,'previous':previous_errors})
                    error=float(np.max(np.abs(scene_state(env,obs)-z['audit_pre_action_PRIVILEGED_state64'][frame])))
                    maximum=max(maximum,error)
                    if error>1e-6:raise ValueError('Native causal state timeline changed')
                    action=env.joint_targets_to_canonical(native[frame],float(closed[frame]))
                    np.testing.assert_array_equal(np.asarray(action,np.float32),z['audit_requested_canonical_before_safety'][frame])
                    obs,_,_,_=capture.step(action)
                    current_error=float(np.max(np.abs(capture.servo[-1]-z['audit_applied_servo_joint_targets'][frame])))
                    servo_error=max(servo_error,current_error)
                    if current_error>1e-6:raise ValueError('Applied motor commands changed')
            result.update(passed=True,frames=len(native),causal_state_max_error=maximum,
                applied_servo_max_error=servo_error,pixel_checks=pixels)
        except Exception as error:result['error']=f'{type(error).__name__}: {error}'
        finally:env.close()
        reports.append(result);atomic_json(output/'progress.json',{'completed_episodes':len(reports),'planned_episodes':len(rows),
            'passed_episodes':sum(r['passed'] for r in reports),'rows':reports,'model_SR_established':False})
    proof={'passed':all(r['passed'] for r in reports),'rows':reports,'scope':SCOPE,'training_qualified':False,
        'real_previous_camera_at_student_handoff':True,'collector_sha256':sha256_file(producer),
        'parent_collection_sha256':sha256_file(candidate/'collection_audit.json'),
        'RGB_gate':'same frozen current-observation 1 LSB and 0.001 changed-pixel fraction for previous images too',
        'bit_exact_all_RGB_claimed':False,'model_SR_established':False,
        'replay_implementation_sha256':sha256_file(Path(__file__)),
        'frozen_rounding_implementation_sha256':sha256_file(Path(__file__).with_name('evaluate_smolvla_v9_source.py'))}
    atomic_json(output/'audit.json',proof)
    if not proof['passed']:raise ValueError('New replay failed; preserve all original failures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();replay(a.candidate,a.output)
