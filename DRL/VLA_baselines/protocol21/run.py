"""Bounded robot21 checks and real native inference. No training is started here."""
import argparse
import json
from pathlib import Path
import numpy as np

from .inputs import Inputs
from .specs import SPECS


PACKAGE = Path(__file__).resolve().parents[1]


def simulation_smoke(steps=3):
    """Real PyBullet capture/step; explicitly no learned VLA or success-rate claim."""
    from VLA.common.body_environment import BodyGraspEnvironment
    from VLA.common.tasks import TASKS
    from .sensors import capture
    from .state21_reference import fit_train_stats
    profile = PACKAGE/'VLA/profiles/body_contact_v9_wrist65_f80_encoder.json'
    result = {'mode': 'simulation_interface_smoke', 'native_policy_inference': False,
              'optimizer_steps': 0, 'success_rate_measured': False, 'tasks': {}}
    env = BodyGraspEnvironment(observation_profile=profile, robot='ur5e', image_size=384,
                               physics_hz=240, control_hz=10, seed=20261126,
                               required_success_steps=1, use_semantic_success=True)
    try:
        for task, spec in TASKS.items():
            obs = env.reset(spec)
            frames = [capture(env, obs, depth=True)]
            for _ in range(steps):
                # Explicit test motion, not a VLA, not an expert task rollout.
                obs, _, done, info = env.step(np.array([0,0,0,0,0,0,-1], np.float32))
                frames.append(capture(env, obs, depth=True))
                if done: break
            raw = np.stack([f.raw_state21 for f in frames])
            stats = {'robot21': fit_train_stats(raw),
                     'adapter': {'proprio': {'q01':[-1]*8, 'q99':[1]*8, 'mask':[True]*8}}}
            reports = {}
            for family, config in SPECS.items():
                inputs = Inputs(family, stats)
                for frame in frames: inputs.push(frame)
                s = inputs.sample()
                shape = {k:list(v.shape) for k,v in s.items() if isinstance(v,np.ndarray)}
                reports[family] = {'shapes':shape, 'H':config.horizon, 'K':config.execute,
                                   'history_times':s['history_times']}
            result['tasks'][task] = {'observations':len(frames),'models':reports}
    finally:
        env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke','audit','native'))
    parser.add_argument('--model', choices=tuple(SPECS), default='xvla')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--stats', type=Path)
    parser.add_argument('--resources', type=Path, default=PACKAGE)
    parser.add_argument('--foundation', type=Path)
    parser.add_argument('--adaptation', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20: parser.error('Use a bounded 1..20 control-step smoke check')
    if args.mode == 'smoke':
        result = simulation_smoke(args.steps)
    else:
        if args.manifest is None: parser.error('--manifest is required')
        from .data import Bank
        bank = Bank(args.manifest)
        stats = json.loads(args.stats.read_text()) if args.stats else bank.statistics()
        s = bank.sample(args.model, args.episode, args.frame, stats)
        result = {'mode':args.mode,'family':args.model,'optimizer_steps':0,
                  'sample_shapes':{k:list(v.shape) for k,v in s.items() if isinstance(v,np.ndarray)},
                  'native_policy_inference':False}
        if args.mode == 'native':
            from .native import create
            runtime = create(args.model,stats,resources=args.resources,foundation=args.foundation,device=args.device)
            if args.adaptation:
                from .checkpoint import load
                load(runtime,args.model,args.adaptation,stats)
            from .execution import prediction
            action = prediction(runtime,args.model,s,stats,seed=20261126)
            result.update(native_policy_inference=True,actions=action.tolist())
    message = json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        with args.output.open('x') as stream: stream.write(message+'\n')
    print(message)


if __name__ == '__main__': main()
