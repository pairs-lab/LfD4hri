"""CPU-only measured FK impact of BF16 absolute-joint signal rounding.

FP32 model masters are distinct from signal/scheduler precision. This audits
the actual sparse joint convention without training/inference or changing the
running experiment. Kinematic error is NOT learned-policy performance.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from VLA.common.environment import ROBOT_SPECS
from VLA.common.provenance import sha256_file
from VLA.evaluate_smolvla_v9_source import verify_source_audit
from VLA.rdt.training_data import wrap_joints


def roundtrip_bf16(values):
    return torch.as_tensor(np.asarray(values, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def percentiles(values):
    return dict(zip(('minimum','median','p90','p99','maximum'),
                    map(float,np.percentile(values,(0,50,90,99,100)))))


def audit(args):
    if args.output.exists(): raise FileExistsError(args.output)
    source=args.source.resolve()
    proof,manifest=verify_source_audit(source,args.source_audit)
    if manifest.get('split')!='train': raise ValueError('TRAIN-only signal audit')
    index=json.loads((source/'index.json').read_text())
    spec=ROBOT_SPECS['ur5e']
    client=p.connect(p.DIRECT)
    robot=p.loadURDF(str(spec.urdf),(0,0,0),useFixedBase=True,physicsClientId=client)
    stats=defaultdict(lambda:defaultdict(list))
    largest=[]
    def fk(joints):
        for joint,value in zip(spec.arm_joint_ids,joints,strict=True):
            p.resetJointState(robot,joint,float(value),physicsClientId=client)
        position,orientation=p.getLinkState(robot,spec.mount_link,computeForwardKinematics=True,physicsClientId=client)[4:6]
        return np.asarray(position),np.asarray(orientation)
    try:
        for row in index:
            with np.load(source/row['file'],allow_pickle=False) as archive:
                current=wrap_joints(archive['joint_positions'])
                target=wrap_joints(archive['next_joint_targets'])
                canonical=archive['canonical_actions']
                held=np.r_[False,archive['next_grasped_object'][:-1].astype(bool)]
            rounded_current,rounded_target=roundtrip_bf16(current),roundtrip_bf16(target)
            for step,(state,goal,qs,qg) in enumerate(zip(current,target,rounded_current,rounded_target,strict=True)):
                state_pos,_=fk(state); goal_pos,goal_quat=fk(goal)
                qs_pos,_=fk(qs); qg_pos,qg_quat=fk(qg)
                target_error=float(np.linalg.norm(qg_pos-goal_pos))
                quat_dot=min(1.,abs(float(np.dot(qg_quat,goal_quat))))
                data=stats[row['task']['name']]
                data['state_mount_position_error_m'].append(float(np.linalg.norm(qs_pos-state_pos)))
                data['target_mount_position_error_m'].append(target_error)
                data['target_orientation_error_rad'].append(float(2*np.arccos(quat_dot)))
                signal=float(np.linalg.norm(goal_pos-state_pos))
                collapsed=bool(np.array_equal(qs,qg) and signal>1e-4)
                data['nonzero_translation_frames'].append(signal>1e-4)
                data['nonzero_translation_collapsed_to_identical_bf16_joints'].append(collapsed)
                data['target_error_over_4mm'].append(target_error>.004)
                critical=bool(canonical[step,6]>.2 and not held[step])
                data['close_before_target_held_frames'].append(critical)
                data['close_before_target_held_error_over_4mm'].append(critical and target_error>.004)
                largest.append({'file':row['file'],'task':row['task']['name'],'step':step,
                                'target_error_m':target_error,'original_mount_translation_m':signal,
                                'close_before_target_held':critical})
    finally:p.disconnect(client)
    result={}
    for task,data in stats.items():
        result[task]={key:(sum(map(int,values)) if not key.endswith(('_m','_rad')) else percentiles(values))
                      for key,values in data.items()}
        result[task]['frames']=len(data['target_mount_position_error_m'])
    verify_source_audit(source,args.source_audit)
    report={'scope':'BF16 SIGNAL precision FK diagnostic; NOT learned/model success',
        'source':str(source),'source_index_sha256':proof['index_sha256'],
        'source_payload_tree_sha256':proof['source_payload_fingerprint']['payload_tree_sha256'],
        'frames':sum(row['steps'] for row in index),'tasks':result,
        'largest_target_rounding_errors':sorted(largest,key=lambda r:-r['target_error_m'])[:12],
        'training_code_sha256':sha256_file(Path(__file__).with_name('training.py')),
        'server_code_sha256':sha256_file(Path(__file__).with_name('server.py')),
        'method':'FP32 wrapped joints -> BF16 -> FP32, same UR5e mount FK; no simulation stepping or model weights',
        'training_performed':False,'validation_used':False,'final_test_used':False,
        'caveats':['Mount FK, not actual contact/closed-loop error; TCP orientation offsets are not included.',
                   '4mm is expert alignment scale, not a replacement success gate.',
                   'Source numeric rounding does not identify the cause of previous SmolVLA failures.',
                   'Quantized FP32-master training/inference can still solve tasks; this alone is not a learned-failure certificate.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    return {key:report[key] for key in ('frames','tasks','largest_target_rounding_errors')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    torch.set_num_threads(1)
    print(json.dumps(audit(parser.parse_args()),indent=2))


if __name__=='__main__':main()
