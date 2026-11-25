"""Execution keyframes: five arms (rows) x five phases (columns), Robotiq gripper.

Replays the logged Ours env-1 episodes (deterministic) and renders five instants
from one shared three-quarter camera, supersampled, all cropped to one common
window so every panel has the same scale. Writes PDF and PNG.
"""
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from env_eval.case_demon.compare_gripper import DATA
from env_eval.case_demon.gripper import GripperEnv, GripperRollout
from env_eval.case_demon.robots import ROBOTS
from env_eval.case_demon.run import WORK
from env_eval.case_demon.scenes import build

OUT = Path('/home/uet/Downloads/trajectory_compare_gripper')
LABELS = {'Panda': 'Franka Panda', 'KUKA': 'KUKA LBR iiwa', 'Kinova': 'Kinova Gen3', 'UR5e': 'UR5e', 'UF850': 'UFACTORY 850'}
PHASES = ('Start', 'Approach', 'Grasp', 'Transport', 'Place')
SCENE = {robot: 1+3*i for i, robot in enumerate(ROBOTS)}  # env-1 scene id of each arm
CAMERA = {'eye': (1.15, -1.55, 1.85), 'target': (-.02, -.36, .32), 'fov': 40.}
SIZE, SUPER, TILES = (1600, 1200), 2, 4
BACKGROUND = 255


def render(env):
    """Supersampled image via TILES x TILES off-axis frusta (EGL caps near 2048 px)."""
    width, height = SIZE[0]*SUPER, SIZE[1]*SUPER
    view = p.computeViewMatrix(CAMERA['eye'], CAMERA['target'], (0, 0, 1))
    near, far = .3, 6.
    top = near*np.tan(np.radians(CAMERA['fov'])/2)
    right = top*width/height
    tw, th = width//TILES, height//TILES
    rgb = np.zeros((height, width, 3), np.uint8)
    seg = np.zeros((height, width), np.int64)
    for row in range(TILES):
        for col in range(TILES):
            left, upper = -right+2*right*col/TILES, top-2*top*row/TILES
            projection = p.computeProjectionMatrix(left, left+2*right/TILES, upper-2*top/TILES, upper, near, far)
            _, _, rgba, _, mask = p.getCameraImage(tw, th, viewMatrix=view, projectionMatrix=projection,
                renderer=p.ER_BULLET_HARDWARE_OPENGL, shadow=1, lightDirection=[-1, -.6, 2], lightColor=[1, 1, .98],
                lightAmbientCoeff=.55, lightDiffuseCoeff=.65, lightSpecularCoeff=.0)
            rgb[row*th:(row+1)*th, col*tw:(col+1)*tw] = np.asarray(rgba, np.uint8).reshape(th, tw, 4)[:, :, :3]
            seg[row*th:(row+1)*th, col*tw:(col+1)*tw] = np.asarray(mask).reshape(th, tw)
    rgb[seg < 0] = BACKGROUND  # the renderer's sky above the floor
    # Studio floor: rescale the lit floor to pure white, shadows keep their relative depth.
    floor = seg == env.plane
    lit = np.percentile(rgb[floor].astype(float), 99, axis=0)
    rgb[floor] = np.clip(rgb[floor]*(255/lit), 0, 255).astype(np.uint8)
    rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
    content = cv2.resize(((seg >= 0) & (seg != env.plane) & (seg != env.table)).astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
    return rgb, content


def capture(robot):
    """Replay the logged episode and grab the five phase instants."""
    log = np.load(DATA/f'{robot}_env1_ours.npz')
    events = json.loads(str(log['events']))
    pick, place = events['pick'][0], events['place'][0]
    ticks = {'Start': 1, 'Approach': pick-int(1.0*240), 'Grasp': pick+int(.3*240),
             'Transport': pick+int(.55*(place-pick)), 'Place': place+int(.5*240)}
    candidate = json.loads((WORK/'qualify'/f'scene{SCENE[robot]:02d}.json').read_text())['selected']
    env = GripperEnv(build(SCENE[robot], candidate), egl=True)
    p.changeVisualShape(env.plane, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=-1)  # white studio floor
    frames, state = {}, {}
    def hook(env):
        for phase, tick in ticks.items():
            if env.ticks == tick:
                frames[phase] = render(env)
                state[phase] = env.holding()
    env.tick_hook = hook
    result = GripperRollout(env).run()
    env.close()
    # Same episode as the logged one: grasp/release instants and outcome must match.
    if not result['overall_pass'] or not state['Grasp'] or state['Place'] or state['Approach']:
        raise RuntimeError(f'{robot}: replay does not match the logged episode {state}')
    return frames, ticks


def main():
    cache = WORK/'keyframes_cache.pkl'  # rendered frames; delete to re-simulate
    if cache.exists():
        rows, times = pickle.loads(cache.read_bytes())
    else:
        rows, times = {}, {}
        for robot in ROBOTS:
            rows[robot], ticks = capture(robot)
            times[robot] = {phase: (tick-1)/240 for phase, tick in ticks.items()}  # seconds since episode start
            print(robot, 'ticks', ticks, flush=True)
        cache.write_bytes(pickle.dumps((rows, times)))
    # One crop window for all 25 panels: same scale everywhere.
    mask = np.zeros(SIZE[::-1], bool)
    for frames in rows.values():
        for _, content in frames.values():
            mask |= content.astype(bool)
    ys, xs = np.nonzero(mask)
    pad = int(.035*SIZE[0])
    x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, SIZE[0])
    y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, SIZE[1])
    aspect = (y1-y0)/(x1-x0)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    left, top, gap = .42, .04, .03  # inches
    panel_w = (7.16-left-4*gap-.02)/5
    panel_h = panel_w*aspect
    height = top+5*panel_h+4*gap+.04
    fig = plt.figure(figsize=(7.16, height))
    for r, robot in enumerate(ROBOTS):
        for c, phase in enumerate(PHASES):
            x = (left+c*(panel_w+gap))/7.16
            y = 1-(top+(r+1)*panel_h+r*gap)/height
            ax = fig.add_axes([x, y, panel_w/7.16, panel_h/height])
            ax.imshow(rows[robot][phase][0][y0:y1, x0:x1], interpolation='lanczos')
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#c9ced3')
                spine.set_linewidth(.6)
            ax.text(.04, .97, f'$t$ = {times[robot][phase]:.1f} s', transform=ax.transAxes, ha='left', va='top',
                    fontsize=9, color='#111111')
            if c == 0:
                ax.set_ylabel(LABELS[robot], fontsize=10.5, fontweight='bold', labelpad=6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'execution_keyframes.pdf', dpi=600)
    fig.savefig(OUT/'execution_keyframes.png', dpi=600)
    print('crop', (x0, y0, x1, y1), 'figure in', 7.16, round(height, 2))


if __name__ == '__main__':
    main()
