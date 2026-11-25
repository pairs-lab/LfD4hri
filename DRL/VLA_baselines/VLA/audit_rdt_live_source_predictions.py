"""Read-only neural TRAIN/VAL diagnostics using the already loaded RDT.

No simulation action is executed. Every request supplies BOTH previous RGBs;
it cannot replace the ongoing evaluator's explicit history. The server uses
per-request forked deterministic noise. Missing responses are not task fails.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import requests
import pybullet as p
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.privileged_native_trial import png
from VLA.rdt.simstate_inputs import SCHEMA
from VLA.rdt.training_data import wrap_joints


def main():
    from VLA.dataset_replay_context import audit_environment
    from VLA.common.tasks import TaskSpec
    from VLA.common.native_action_fields import native_gripper_closure
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--port',type=int,default=5734);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,choices=(1,4,8),default=1);a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);session=requests.Session();session.trust_env=False
    url=f'http://127.0.0.1:{a.port}';reply=session.get(url+'/metadata',timeout=15);reply.raise_for_status();metadata=reply.json()
    if metadata.get('checkpoint',{}).get('path')!=str(a.checkpoint.resolve()) or metadata.get('simstate_schema')!=SCHEMA:
        raise ValueError('Wrong live checkpoint/schema; no inference requested')
    root=Path(__file__).resolve().parents[1];source_root=root/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013'
    sets=[('train',source_root/'qualified_train',root/'VLA/audits/recovery_r17_20261013/train'),
        ('validation',source_root/'validation_candidate',root/'VLA/audits/recovery_r19_metadata_gate_20261014/validation')]
    rows=[];errors=[];attempts=0
    def save(status):
        summary={}
        for split in ('train','validation'):
            entries=[r for r in rows if r['split']==split]
            if entries:summary[split]={k:float(np.mean([r[k] for r in entries])) for k in
                ('first_joint_periodic_RMSE_rad','first_mount_FK_error_m','single_mean_mount_FK_error_m',
                 'ensemble_member_spread_m','chunk_joint_RMSE_rad','gripper_open_MAE','open_close_agreement')}
        atomic_json(a.output/'audit.json',{'status':status,'scope':'offline_neural_prediction_errors_NOT_episode_success_rate',
            'metadata':metadata,'actual_native_neural_queries':len(rows)*a.repeats,'attempted_queries':attempts,
            'cases':len(rows),'neural_samples_per_case':a.repeats,'implementation_sha256':sha256_file(Path(__file__)),
            'robot_controls_executed':0,'training_or_scorer_changed':False,'explicit_source_image_history':True,
            'TRAIN_and_VAL_NOT_mixed_for_fitting':True,'summary':summary,'rows':rows,'errors':errors})
    env=None
    try:
        for split,source,projection in sets:
            index=json.loads((source/'index.json').read_text());proof=json.loads((projection/'audit.json').read_text())
            if not proof['passed'] or proof['source_index_sha256']!=sha256_file(source/'index.json'):raise ValueError('Actual source18 proof changed')
            source_proofs={r['file']:r for r in proof['rows']};selected=[]
            for task in ('touch','pick','move','pick_place'):
                names=set()
                for row in index:
                    if row['task']['name']==task and row['task']['pick_name'] not in names:
                        selected.append(row);names.add(row['task']['pick_name'])
                        if len(names)==2:break
            if len(selected)!=8:raise ValueError('All four tasks/two distinct targets per split required')
            if env is None:
                env=audit_environment(source,image_size=384,control_hz=10,seed=0,required_success_steps=1,
                    use_semantic_success=True);env.reset(TaskSpec.from_dict(selected[0]['task']))
            def fk(q):
                for jid,angle in zip(env.robot_spec.arm_joint_ids,q,strict=True):p.resetJointState(env.robot_id,jid,float(angle),physicsClientId=env.client_id)
                return np.asarray(p.getLinkState(env.robot_id,env.robot_spec.mount_link,computeForwardKinematics=True,physicsClientId=env.client_id)[4])
            for row in selected:
                name=row['file'];statepath=projection/'states'/name
                if sha256_file(statepath)!=source_proofs[name]['state_payload_sha256'] or sha256_file(source/name)!=source_proofs[name]['source_payload_sha256']:
                    raise ValueError('Actual source/states payload changed')
                with np.load(statepath,allow_pickle=False) as z:states=z['state'].copy()
                with np.load(source/name,allow_pickle=False) as z:
                    for frame in sorted({0,row['steps']//2,row['steps']-1}):
                        previous=max(frame-1,0);proprio=np.zeros(16,np.float32)
                        proprio[:6]=z['joint_positions'][frame];proprio[6]=z['gripper_open'][frame]
                        payload={'front_rgb_png':png(z['front_rgb'][frame]),'wrist_rgb_png':png(z['wrist_rgb'][frame]),
                            'previous_front_rgb_png':png(z['front_rgb'][previous]),'previous_wrist_rgb_png':png(z['wrist_rgb'][previous]),
                            'proprio':proprio.tolist(),'prompt':str(z['prompt'].item()),'step':frame,'seed':20261532,
                            'simstate_schema':SCHEMA,'privileged_state18':states[frame].tolist()}
                        members=[]
                        for repeat in range(a.repeats):
                            payload['seed']=20261532+repeat;attempts+=1
                            response=session.post(url+'/act',json=payload,timeout=180)
                            if not response.ok:
                                errors.append({'split':split,'file':name,'frame':frame,'HTTP_status':response.status_code,'error':response.text[:300]});save('running');break
                            member=np.asarray(response.json()['joint_actions'],np.float32)
                            if member.shape!=(10,7) or not np.isfinite(member).all():raise ValueError('Malformed actual native prediction')
                            members.append(member)
                        if len(members)!=a.repeats:continue
                        members=np.stack(members);predicted=members.mean(0)
                        # Circular joint mean is a declared sampling-only
                        # ablation, not a target-aware planner/IK trajectory.
                        predicted[:,:6]=np.arctan2(np.sin(members[:,:,:6]).mean(0),np.cos(members[:,:,:6]).mean(0))
                        ids=np.minimum(frame+np.arange(10),row['steps']-1)
                        expected=np.c_[wrap_joints(z['next_joint_targets'][ids]),1-native_gripper_closure(z)[ids]]
                        qerror=(predicted[:,:6]-expected[:,:6]+np.pi)%(2*np.pi)-np.pi
                        truth=fk(expected[0,:6]);positions=np.asarray([fk(x[0,:6]) for x in members])
                        error=float(np.linalg.norm(fk(predicted[0,:6])-truth))
                        rows.append({'split':split,'file':name,'task':row['task']['name'],'frame':frame,
                            'first_joint_periodic_RMSE_rad':float(np.sqrt(np.mean(qerror[0]**2))),
                            'chunk_joint_RMSE_rad':float(np.sqrt(np.mean(qerror**2))),
                            'first_mount_FK_error_m':error,'gripper_open_MAE':float(np.abs(predicted[:,6]-expected[:,6]).mean()),
                            'single_mean_mount_FK_error_m':float(np.linalg.norm(positions-truth,axis=-1).mean()),
                            'ensemble_member_spread_m':float(np.linalg.norm(positions-positions.mean(0),axis=-1).mean()),
                            'open_close_agreement':float(np.mean((predicted[:,6]>.5)==(expected[:,6]>.5))),
                            'current_verified_held':bool(states[frame,17]),'predicted_native':predicted.tolist(),
                            'individual_native_predictions':members.tolist(),'teacher_native_NOT_runtime_action':expected.tolist()})
                        save('running')
        save('completed48_cases_native_neural_queries_NOT_SR' if len(rows)==48 and not errors else 'incomplete_native_queries_NOT_zero_SR')
    except Exception:
        save('failed_preserve_all_actual_predictions');raise
    finally:
        if env is not None:env.close()
        session.close()


if __name__=='__main__':main()
