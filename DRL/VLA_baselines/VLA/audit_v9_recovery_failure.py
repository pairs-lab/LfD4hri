"""Reproduce an exact predeclared recovery teacher exception, no scene reroll."""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image

from VLA.audit_counterfactual_collection import alternate_target
from VLA.collect_v9_recovery_candidate import GentleRecoveryCollection, SafeRecoveryOpeningTeacher
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.near_object_recovery_start import validated_near_object_reset_environment
from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchTeacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--layout',type=int,required=True)
    parser.add_argument('--variant',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--safe-opening',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    plan=json.loads((args.candidate/'collection_plan.json').read_text())
    source=Path(plan['source'])
    filename=plan['selected_source_files'][args.layout]
    row=next(r for r in json.loads((source/'index.json').read_text()) if r['file']==filename)
    task=TaskSpec.from_dict(row['task'])
    instruction=alternate_target(task,choice=args.layout) if args.variant else task
    startup=json.loads((args.candidate/f'layout{args.layout:03d}_startup_evidence.json').read_text())
    args.output.mkdir(parents=True)
    frames=[]
    with audit_environment(source,image_size=384,required_success_steps=1,success_threshold_scale=1.,use_semantic_success=False) as env:
        reset_source_episode(env,source,row)
        proxy=validated_near_object_reset_environment(env,task,startup['actions_NOT_labels'],startup['evidence'])
        collection=GentleRecoveryCollection(proxy)
        observation=collection.reset(instruction)
        teacher=(SafeRecoveryOpeningTeacher(collection,SurfaceLimitedTouchTeacher) if args.safe_opening
                 else SurfaceLimitedTouchTeacher(collection))
        teacher.reset(instruction)
        initial={name:np.array(p.getBasePositionAndOrientation(body)[0]) for name,body in env.object_ids.items()}
        error=None; final_info={}
        for step in range(instruction.max_steps):
            record={'step':step,'phase':teacher.phase,'closure':env.measured_gripper_closure(),
                    'tcp':env.gripper.grasp_center().tolist(),
                    'pick_position':list(p.getBasePositionAndOrientation(env.pick_id)[0]),
                    'object_translation_from_reset_m':{name:float(np.linalg.norm(np.array(p.getBasePositionAndOrientation(body)[0])-initial[name])) for name,body in env.object_ids.items()},
                    'gripper_contacts':len(p.getContactPoints(bodyA=env.gripper.body))}
            for camera in ('front','wrist'):
                Image.fromarray(getattr(observation,camera+'_rgb')).save(args.output/f'step{step:03d}_{camera}.png')
            frames.append(record)
            try:action=teacher.predict(observation)
            except Exception as exc:error=str(exc);break
            record['action']=action.tolist()
            observation,_,done,info=collection.step(action)
            final_info=info
            record['done']=done;record['safety_violation']=info.get('safety_violation')
            if done:break
        report={'source_file':filename,'layout':args.layout,'variant':args.variant,
                'task':instruction.to_dict(),'error':error,'frames':frames,
                'safe_opening_expert_only':args.safe_opening,
                'success':bool(final_info.get('success',False)),
                'expert_quality_failure':final_info.get('expert_quality_failure'),
                'expert_terminal_surface_gap_m':final_info.get('expert_terminal_surface_gap_m'),
                'expert_terminal_max_normal_force_n':final_info.get('expert_terminal_max_normal_force_n'),
                'source_layout_changed':False,'runtime_controller_changed':False,'training_performed':False}
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'error':error,'success':report['success'],'expert_quality_failure':report['expert_quality_failure'],
                          'surface_gap_m':report['expert_terminal_surface_gap_m'],
                          'force_n':report['expert_terminal_max_normal_force_n'],'frames':len(frames),'last':frames[-1]},indent=2))


if __name__=='__main__':main()
