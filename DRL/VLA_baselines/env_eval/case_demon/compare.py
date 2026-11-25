"""Run every scene with the Oracle and the straight-line IK baseline; log trajectories.

The IK tool speed is tuned per scene so its episode lasts as long as the Oracle's.
"""
import json

import numpy as np
import pybullet as p

from env_eval.case_demon.ik_baseline import IKRollout
from env_eval.case_demon.rollout import PHYSICS_HZ, Rollout, SuctionEnv
from env_eval.case_demon.run import WORK
from env_eval.case_demon.scenes import TASKS, build

OUT = WORK/'compare'


class Recorder:
    """Tip, joints and joint-origin skeleton at every physics tick; grip events."""
    def __init__(self, env):
        self.env = env
        self.tip, self.q, self.skeleton, self.events = [], [], [], {}
        self.links = [p.getJointInfo(env.robot, j)[16] for j in env.joints]+[env.joints[-1]]
        self.held = False

    def __call__(self, env):
        self.tip.append(env.tip())
        self.q.append(env.q.copy())
        base = np.array(p.getBasePositionAndOrientation(env.robot)[0])
        chain = [base]+[np.array(p.getLinkState(env.robot, j, computeForwardKinematics=True)[4]) for j in env.joints]
        chain.append(env.tip())
        self.skeleton.append(np.array(chain))
        held = env.holding()
        if held != self.held:
            self.events['pick' if held else 'place'] = (len(self.tip)-1, env.tip().tolist())
            self.held = held


def episode(sid, candidate, method, speed=None):
    env = SuctionEnv(build(sid, candidate), egl=False)
    rollout = Rollout(env) if method == 'ours' else IKRollout(env, speed)
    recorder = Recorder(env)
    env.tick_hook = recorder
    result = rollout.run()
    scene = {'pick_xy': np.array(p.getBasePositionAndOrientation(env.pick)[0])[:2].tolist(),
             'bowl_aabb': [list(v) for v in p.getAABB(env.place)], 'table_top': .004,
             'pick_extent': (np.subtract(*p.getAABB(env.pick)[::-1])).tolist()}
    env.close()
    return result, recorder, scene


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for sid in range(1, len(TASKS)+1):
        report = json.loads((WORK/'qualify'/f'scene{sid:02d}.json').read_text())
        candidate = report['selected']
        ours, rec_ours, scene = episode(sid, candidate, 'ours')
        # Two passes: guess the speed from the Oracle's duration, then correct it.
        speed, ik = .10, None
        for _ in range(3):
            ik, rec_ik, _ = episode(sid, candidate, 'ik', speed)
            ratio = ik['seconds']/ours['seconds']
            if abs(ratio-1) < .02:
                break
            moving = ik['seconds']-1.1  # holds: attach .28 + release .12 + settle .7
            speed *= moving/max(ours['seconds']-1.1, .1)
        name = f"{ours['scene']['robot']}_env{ours['scene']['env']}"
        for method, result, rec in (('ours', ours, rec_ours), ('ik', ik, rec_ik)):
            np.savez_compressed(OUT/f'{name}_{method}.npz', tip=np.array(rec.tip), q=np.array(rec.q),
                                skeleton=np.array(rec.skeleton), hz=PHYSICS_HZ,
                                events=json.dumps(rec.events), scene=json.dumps(scene),
                                result=json.dumps(result, default=str))
        replay = np.allclose(ours.get('final_object', [9]*3), report['attempts'][-1].get('final_object', [0]*3), atol=1e-4)
        row = {'name': name, 'ours_pass': ours['overall_pass'], 'ik_pass': ik['overall_pass'],
               'ours_s': round(ours['seconds'], 2), 'ik_s': round(ik['seconds'], 2), 'ik_speed': round(speed, 4),
               'ours_replay_matches': bool(replay), 'ik_failure': ik.get('failure') or ik['unwanted_contacts'][:3]}
        summary.append(row)
        print(row, flush=True)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
