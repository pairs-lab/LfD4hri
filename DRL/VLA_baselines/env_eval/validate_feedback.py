"""Held-out layouts and physical perturbations, separate from showcase videos."""
from dataclasses import replace
from pathlib import Path
import argparse,json
import numpy as np
import pybullet as p
from env_eval.feedback_motion import MotionProfile
from env_eval.feedback_environment import FeedbackEnvironment
from env_eval.oracle_feedback import FeedbackRollout
from env_eval.oracle_demo import scenes
from env_eval.audit_feedback import score_run


class PerturbedRollout(FeedbackRollout):
    pulse_ms=8.
    def step(self,center,q,command):
        # A force pulse during approach is a real physics disturbance. Do not
        # teleport an object or change the robot's state during execution.
        approach_seconds=(5.5 if self.env.robot_spec.name=='uf850' else 3.5)*self.profile.tempo*self.profile.approach_scale
        if self.phase=='approach' and .48<=self.phase_progress<=.48+self.pulse_ms/1000/approach_seconds:
            position=p.getBasePositionAndOrientation(self.env.pick_id)[0]
            mass=p.getDynamicsInfo(self.env.pick_id,-1)[0]
            p.applyExternalForce(self.env.pick_id,-1,[1.5*mass*9.81,0,0],position,p.WORLD_FRAME)
        super().step(center,q,command)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--robots',default='uf850,ur5e')
    parser.add_argument('--seeds',default='1,2');parser.add_argument('--force-pulse',action='store_true')
    parser.add_argument('--cases',help='Optional action:config pairs separated by commas')
    parser.add_argument('--pulse-ms',type=float,default=8.)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();profiles=json.loads(args.profile.read_text());args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for robot in args.robots.split(','):
        profile=MotionProfile(**profiles.get(robot,profiles))
        with FeedbackEnvironment(robot,args.output/f'assets_{robot}') as env:
            env.joint_jerk=profile.joint_jerk
            for seed in map(int,args.seeds.split(',')):
                for scene in scenes():
                    if args.cases and f'{scene.action}:{scene.config}' not in args.cases.split(','):continue
                    task=scene.task();rng=np.random.default_rng(30000+seed*101+task.task_id*13+scene.config)
                    shift=rng.uniform(-.008,.008,size=2)
                    position=np.array(task.pick_position);position[:2]+=shift
                    task=replace(task,pick_position=tuple(position),scene_profile='feedback_v3_heldout_layout')
                    path=args.output/robot/f'seed{seed}'/scene.action/f'config{scene.config}'
                    if args.resume and (path/'audit.json').exists():
                        rows.append(json.loads((path/'audit.json').read_text()))
                        continue
                    path.mkdir(parents=True,exist_ok=True);env.reset(task)
                    cls=PerturbedRollout if args.force_pulse else FeedbackRollout
                    PerturbedRollout.pulse_ms=args.pulse_ms
                    rollout=cls(env,scene,path,profile=profile)
                    try:
                        audit=rollout.run();selection=score_run(audit,rollout.samples)
                    except Exception as error:
                        audit={'overall_pass':False,'error':str(error)};selection={'feasible':False,'error':str(error)}
                    finally:rollout.close()
                    row={'robot':robot,'action':scene.action,'config':scene.config,'seed':seed,
                         'initial_xy_shift_m':shift.tolist(),'force_pulse':args.force_pulse,
                         'pulse_ms':args.pulse_ms if args.force_pulse else None,'audit':audit,'selection':selection}
                    rows.append(row);(path/'audit.json').write_text(json.dumps(row,indent=2))
                    (path/'trajectory.json').write_text(json.dumps(rollout.samples))
                    (args.output/'results.json').write_text(json.dumps(rows,indent=2))
                    print(robot,seed,scene.action,scene.config,audit.get('overall_pass'),selection['feasible'],flush=True)
    (args.output/'summary.json').write_text(json.dumps({'episodes':len(rows),
        'successes':sum(r['audit'].get('overall_pass',False) for r in rows),
        'feasible':sum(r['selection']['feasible'] for r in rows),'rows':rows},indent=2))


if __name__=='__main__':main()
