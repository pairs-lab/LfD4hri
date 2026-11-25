"""CASE_demon driver. Stages: preflight, qualify, render (see --stage)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from env_eval.case_demon.render import InsetPlanner, VIEWS, VideoRecorder, frame
from env_eval.case_demon.robots import ROBOTS
from env_eval.case_demon.rollout import Rollout, SuctionEnv
from env_eval.case_demon.scenes import TASKS, build

OUTPUT = Path('/home/uet/Downloads/CASE_demon')
WORK = Path(__file__).resolve().parents[2]/'eval/case_demon_work'
CANDIDATES = 8


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    temp.replace(path)


def preflight(scene_ids):
    out = {}
    for sid in scene_ids:
        scene = build(sid)
        env = SuctionEnv(scene)
        q1 = env.down_quaternion()
        errors, angles = [], []
        for x in (-.26, 0., .26):
            for y in (-.70, -.5, -.33):
                for z in (.04, .3):
                    target = np.array([x, y, z])
                    solution = env.solve_ik(target, q1, iterations=80)
                    error, angle = env.ik_error(solution, target, q1)
                    errors.append(error)
                    angles.append(angle)
        tool = env.tip()-p.getBasePositionAndOrientation(env.suction.body)[0]
        for view, cfg in VIEWS.items():
            rgb, _ = frame(env, cfg, 960, 540)
            path = WORK/'preflight'/f'{scene.robot}_view{view}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out[scene.robot] = {'ik_max_error_m': max(errors), 'ik_max_angle_rad': max(angles),
                            'home_tip': env.tip().tolist(), 'home_tool_axis': (tool/np.linalg.norm(tool)).tolist(),
                            'renderer_is_egl': env.renderer == p.ER_BULLET_HARDWARE_OPENGL}
        print(scene.robot, out[scene.robot], flush=True)
        env.close()
    write(WORK/'PREFLIGHT.json', out)


def qualify_one(sid):
    attempts = []
    for candidate in range(CANDIDATES):
        scene = build(sid, candidate)
        env = SuctionEnv(scene)
        rollout = Rollout(env)
        planner = InsetPlanner(env)
        env.tick_hook = planner
        result = rollout.run()
        result['insets'] = planner.choose()
        env.close()
        attempts.append(result)
        print(sid, scene.robot, f'cand{candidate}', 'PASS' if result['overall_pass'] else 'FAIL',
              result.get('failure') or (result.get('final_xy_offset_m'), result['unwanted_contacts'][:3],
              {k: round(v, 2) for k, v in result['suction_firmness'].items()}), flush=True)
        if result['overall_pass']:
            break
    write(WORK/'qualify'/f'scene{sid:02d}.json', {'attempts': attempts, 'selected': next(
        (a['scene']['candidate'] for a in attempts if a['overall_pass']), None)})


def render_one(sid):
    report = json.loads((WORK/'qualify'/f'scene{sid:02d}.json').read_text())
    if report['selected'] is None:
        raise RuntimeError(f'Scene {sid} has no qualified candidate')
    chosen = report['attempts'][report['selected']]
    scene = build(sid, report['selected'])
    directory = OUTPUT/scene.robot/f'env{scene.env}'
    env = SuctionEnv(scene)
    rollout = Rollout(env)
    recorder = VideoRecorder(env, rollout, directory, chosen['insets'])
    env.tick_hook = recorder
    recorder.capture()
    result = rollout.run()
    env.close()
    same = np.allclose(result['final_object'], chosen['final_object'], atol=1e-4) if result['overall_pass'] else False
    recorder.close(keep=bool(result['overall_pass'] and same))
    write(WORK/'rendered'/f'scene{sid:02d}.json', {**result, 'insets': chosen['insets'], 'frames': recorder.frames,
          'replay_matches_qualification': bool(same)})
    print(sid, scene.robot, f'env{scene.env}', 'frames', recorder.frames, 'pass', result['overall_pass'], 'replay', same, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'qualify', 'render'), required=True)
    parser.add_argument('--scenes', default='', help='comma list; default all 15 (first of each robot for preflight)')
    args = parser.parse_args()
    os.nice(10)
    ids = [int(s) for s in args.scenes.split(',') if s] or (list(range(1, 16, 3)) if args.stage == 'preflight' else list(range(1, len(TASKS)+1)))
    start = time.time()
    if args.stage == 'preflight':
        preflight(ids)
    else:
        for sid in ids:
            (qualify_one if args.stage == 'qualify' else render_one)(sid)
    print('done in', round(time.time()-start), 's', flush=True)


if __name__ == '__main__':
    main()
