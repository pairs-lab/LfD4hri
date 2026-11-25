"""Tune low-dimensional Oracle control profiles with physical rollout rewards.

This is model-based parameter search, not reinforcement learning of a network.
The remaining 24 nominal cases and perturbed layouts are validation data.
"""
import argparse,json
from pathlib import Path
import numpy as np
from env_eval.feedback_motion import MotionProfile
from env_eval.feedback_environment import FeedbackEnvironment
from env_eval.oracle_feedback import FeedbackRollout
from env_eval.oracle_demo import scenes
from env_eval.audit_feedback import score_run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('profile',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profile=MotionProfile(**json.loads(args.profile.read_text()))
    selected=[s for s in scenes() if (s.action,s.config) in [('touch',2),('pick',1),('move',3),('move',4),('put',1),('put',3)]]
    args.output.mkdir(parents=True,exist_ok=True);results=[]
    for robot in ('uf850','ur5e'):
        with FeedbackEnvironment(robot,args.output/'assets') as env:
            env.joint_jerk=profile.joint_jerk
            for scene in selected:
                path=args.output/robot/scene.action/f'config{scene.config}';path.mkdir(parents=True,exist_ok=True)
                env.reset(scene.task());rollout=FeedbackRollout(env,scene,path,profile=profile)
                try:
                    audit=rollout.run();result=score_run(audit,rollout.samples)
                except Exception as error:
                    audit={'overall_pass':False,'error':str(error)};result={'feasible':False,'score':-10000.,'error':str(error)}
                finally:rollout.close()
                result.update(robot=robot,action=scene.action,config=scene.config,audit=audit)
                results.append(result)
                (path/'trajectory.json').write_text(json.dumps(rollout.samples))
                (path/'audit.json').write_text(json.dumps(audit,indent=2))
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
                print(robot,scene.action,scene.config,result['feasible'],round(result['score'],3),flush=True)
    summary={'profile':profile.to_dict(),'feasible':all(r['feasible'] for r in results),
             'mean_score':float(np.mean([r['score'] for r in results])),'rows':results}
    (args.output/'selection.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
