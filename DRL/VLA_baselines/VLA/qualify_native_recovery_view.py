"""Qualify fresh native corrective observations, NOT a model or long fit.

The original failed command-parity proof is preserved. Admission depends on
new native-path observations and their own complete physical replay, including
the actual previous cameras at the student-to-teacher handoff.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import native_gripper_closure
from VLA.scene64_conditioning import encode_numeric


def require_all_attempts(parent_plan, parent_rows, plan, rows):
    if (parent_plan.get('planned_attempts')!=40 or len(parent_rows)!=40
        or parent_plan.get('selection_uses_neural_outcomes') is not False
        or parent_plan.get('validation_dev_final_used_for_training') is not False
        or parent_plan.get('prefix_actions_are_not_teacher_labels') is not True):
        raise ValueError('Original predeclared TRAIN40 lineage is required')
    if [r['source_file'] for r in parent_rows]!=parent_plan['selected_files']:
        raise ValueError('Original attempted scenes changed')
    labelled=[r['source_file'] for r in parent_rows if r.get('file')]
    if len(labelled)!=38 or len(set(labelled))!=38:
        raise ValueError('Preserve all38 labelled and both unlabeled original attempts')
    if plan.get('pilot_only') is not False or plan.get('real_previous_camera_at_student_handoff') is not True:
        raise ValueError('A pilot or duplicated handoff history does not qualify full data')
    if plan['selected_files']!=labelled or [r['source_file'] for r in rows]!=labelled:
        raise ValueError('Do not filter/reorder any native corrective attempt')
    if any(not r.get('success') or not r.get('file') or r.get('error') for r in rows):
        raise ValueError('All actual native corrective paths must pass, including failures')
    return labelled


def qualify(candidate, replay, output):
    candidate,replay,output=map(lambda p:Path(p).resolve(),(candidate,replay,output))
    if output.exists():raise FileExistsError('Preserve all prior qualifications')
    plan=json.loads((candidate/'plan.json').read_text());rows=json.loads((candidate/'index.json').read_text())
    parent=Path(plan['parent_all40_attempts']);parent_plan=json.loads((parent/'plan.json').read_text())
    parent_rows=json.loads((parent/'index.json').read_text());originals={r['source_file']:r for r in parent_rows}
    require_all_attempts(parent_plan,parent_rows,plan,rows)
    collection=json.loads((candidate/'collection_audit.json').read_text());physical=json.loads(replay.read_text())
    implementation=Path(__file__).with_name('collect_native_recovery_view.py')
    accepted_scope=physical.get('scope') in ('fresh_native_corrective_observations_exact_replay_NOT_model_SR',
        'fresh_native_corrective_observations_replay_with_frozen_RGB_rounding_NOT_model_SR')
    if (sha256_file(parent/'index.json')!=plan['parent_index_sha256']
        or sha256_file(implementation)!=plan['collector_sha256']
        or not collection.get('passed') or collection.get('pilot_only') is not False
        or not physical.get('passed') or physical.get('training_qualified') is not False
        or not accepted_scope
        or physical.get('real_previous_camera_at_student_handoff') is not True
        or physical.get('collector_sha256')!=sha256_file(implementation)
        or physical.get('parent_collection_sha256')!=sha256_file(candidate/'collection_audit.json')):
        raise ValueError('Actual complete native-path collection/replay is stale or failed')
    if [r['file'] for r in physical['rows']]!=[r['file'] for r in rows] or any(
        not r['passed'] or r['error'] or r['causal_state_max_error']!=0 for r in physical['rows']):
        raise ValueError('Exact complete native camera/contact/causal-state replay is required')
    if physical['scope'].endswith('frozen_RGB_rounding_NOT_model_SR'):
        from VLA.evaluate_smolvla_v9_source import validate_reset_observation
        replay_implementation=Path(__file__).with_name('replay_native_corrective_history.py')
        rounding_implementation=Path(__file__).with_name('evaluate_smolvla_v9_source.py')
        if (physical['replay_implementation_sha256']!=sha256_file(replay_implementation)
            or physical['frozen_rounding_implementation_sha256']!=sha256_file(rounding_implementation)):
            raise ValueError('Frozen replay/rounding implementation changed')
        for row in physical['rows']:
            if len(row['pixel_checks'])!=row['frames'] or row['applied_servo_max_error']>1e-6:
                raise ValueError('Incomplete actual physical/pixel timeline')
            for frame,checks in enumerate(row['pixel_checks']):
                if checks['frame']!=frame:raise ValueError('Missing replay frame')
                validate_reset_observation(checks['current']);validate_reset_observation(checks['previous'])
    source=Path(plan['source']);source_rows={r['file']:r for r in json.loads((source/'index.json').read_text())}
    bound=[candidate/'plan.json',candidate/'index.json',candidate/'collection_audit.json',replay,
        parent/'plan.json',parent/'index.json',source/'index.json',source/'dataset_manifest.json',
        implementation,Path(__file__)]
    if physical['scope'].endswith('frozen_RGB_rounding_NOT_model_SR'):
        bound.extend((replay_implementation,rounding_implementation))
    receipt_rows=[];frames=0
    for row,proof in zip(rows,physical['rows'],strict=True):
        old=originals[row['source_file']];path=candidate/row['file'];old_path=parent/old['file']
        if (sha256_file(path)!=row['payload_sha256'] or sha256_file(old_path)!=old['payload_sha256']
            or row['task']!=old['task'] or row['task']!=source_rows[row['source_file']]['task']
            or row['recovery_start_actions_not_training_labels']!=old['recovery_start_actions_not_training_labels']
            or row['initial_scene64']!=old['initial_scene64'] or row['steps']!=old['steps']
            or proof['frames']!=row['steps'] or not row['fresh_native_observations']):
            raise ValueError('Native correction labels, requested object or handoff changed')
        with np.load(path,allow_pickle=False) as z,np.load(old_path,allow_pickle=False) as previous:
            n=row['steps'];state=z['audit_pre_action_PRIVILEGED_state64']
            if state.shape!=(n,64) or state.dtype!=np.float32:raise ValueError('Wrong before-action timeline')
            for value in state:encode_numeric(value)
            np.testing.assert_array_equal(z['next_joint_targets'],np.asarray(previous['next_joint_targets'],np.float32))
            np.testing.assert_array_equal(native_gripper_closure(z),np.asarray(native_gripper_closure(previous),np.float32))
            np.testing.assert_allclose(state[0],old['initial_scene64'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(z['joint_positions'],state[:,:6],rtol=0,atol=1e-6)
            np.testing.assert_allclose(z['gripper_closure'],state[:,6],rtol=0,atol=1e-6)
            if str(z['prompt'].item())!=row['task']['prompt']:raise ValueError('Prompt/target identity mismatch')
            for role in ('front','wrist'):
                image,history=z[role+'_rgb'],z['audit_previous_'+role+'_rgb']
                if image.shape!=(n,384,384,3) or image.dtype!=np.uint8 or history.shape!=image.shape or history.dtype!=np.uint8:
                    raise ValueError('Real current/previous native camera frames are required')
                # The collector legitimately calls observe() twice without
                # advancing physics. TinyRenderer can round one channel by
                #1LSB even within this SAME recording. Keep original bytes;
                # use the same existing current-camera rounding bounds.
                difference=np.abs(history[1:].astype(np.int16)-image[:-1].astype(np.int16))
                if (difference.max(initial=0)>1 or np.any(np.mean(np.any(difference!=0,axis=-1),axis=(1,2))>.001)):
                    raise ValueError('Previous/current adjacent image timeline differs beyond frozen RGB rounding')
            if z['audit_applied_servo_joint_targets'].shape!=(n,6):raise ValueError('Actual native servo trace missing')
        frames+=n;bound.extend((path,old_path))
        receipt_rows.append({'file':row['file'],'source_file':row['source_file'],'frames':n,'payload_sha256':sha256_file(path)})
    result={'passed':True,'scope':'fresh_native_joint_corrective_view_full38_exact_physics_NOT_model_qualification',
        'candidate':str(candidate),'source':str(source),'episodes':38,'frames':frames,'rows':receipt_rows,
        'all40_original_attempts_preserved':True,'unlabeled_original_attempts':[
            r['source_file'] for r in parent_rows if not r.get('file')],
        'fresh_native_observation_action_view_qualified':True,'real_previous_camera_at_student_handoff':True,
        'old_native_joint_command_parity_failure_waived':False,'teacher_is_only_offline_supervision':True,
        'prefix_actions_supervised':False,'validation_dev_final_used':False,'model_SR_established':False,
        'long_finetune_qualified':False,'RGB_bit_exact_claimed':physical['scope']=='fresh_native_corrective_observations_exact_replay_NOT_model_SR',
        'RGB_gate':physical.get('RGB_gate','bit exact'),
        'adjacent_RGB_gate':'same existing current-camera 1LSB/0.001 changed-pixel fraction; no source images edited',
        'next_required':'model-specific actual mixed batch, own CUDA/save/reload/deployed qualification',
        'bound_files':{str(p):sha256_file(p) for p in bound}}
    atomic_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','replay','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=qualify(a.candidate,a.replay,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('bound_files','rows')}))
