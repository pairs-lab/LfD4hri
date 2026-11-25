"""Gripper comparison: IK | Ours on the same 15 tasks. Side-by-side videos + logs.

Per scene: Ours (with jitter) and Ours-without-jitter run first; the IK tool
speed is then tuned so its episode lasts as long as Ours. Every run is the real
physics episode; videos are captured live from one shared camera.
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from env_eval.case_demon.gripper import GripperEnv, GripperIKRollout, GripperRollout, SmoothOnlyRollout
from env_eval.case_demon.rollout import PHYSICS_HZ
from env_eval.case_demon.run import WORK
from env_eval.case_demon.scenes import TASKS, build

DATA = WORK/'compare_gripper'
VIDEOS = Path('/home/uet/Downloads/trajectory_compare_gripper/videos')
# Elevated three-quarter view: whole arm, table and both trails in frame.
CAMERA = {'eye': (1.05, -1.55, 1.45), 'target': (0., -.40, .25), 'up': (0, 0, 1), 'fov': 42.}
BACKGROUND = (244, 244, 244)  # replaces the renderer's lavender sky above the floor
W, H, FPS = 1920, 1080, 30
COLOR = {'ik': (213, 94, 0), 'ours': (0, 114, 178)}  # Okabe-Ito, RGB
TITLE = {'ik': 'IK', 'ours': 'Ours'}
PICK, PLACE = (230, 159, 0), (0, 158, 115)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 46)
CONTROLLERS = {'ours': GripperRollout, 'smooth': SmoothOnlyRollout, 'ik': GripperIKRollout}


def mesh_shapes(body):
    """Per link: drawable meshes (STL/OBJ visual, else collision) with local frames.

    PyBullet reports visual frames relative to the URDF link frame and collision
    frames relative to the link inertial (COM) frame; 'kind' records which.
    """
    shapes = []
    links = range(-1, p.getNumJoints(body))
    visual = {}
    for v in p.getVisualShapeData(body):
        visual.setdefault(v[1], []).append(v)
    for link in links:
        chosen = [(v[4].decode(), v[3], v[5], v[6], v[7]) for v in visual.get(link, [])
                  if v[2] == p.GEOM_MESH and v[4].decode().lower().endswith(('.stl', '.obj'))]
        if not chosen:
            rgba = visual[link][0][7] if link in visual else (.7, .7, .7, 1)
            chosen = [(c[4].decode(), c[3], c[5], c[6], rgba) for c in p.getCollisionShapeData(body, link)
                      if c[2] == p.GEOM_MESH]
        kind = 'visual' if visual.get(link) and any(v[4].decode().lower().endswith(('.stl', '.obj'))
                                                   for v in visual[link] if v[2] == p.GEOM_MESH) else 'collision'
        for filename, scale, position, orientation, rgba in chosen:
            shapes.append({'link': link, 'file': filename, 'scale': list(scale), 'pos': list(position),
                           'orn': list(orientation), 'rgba': list(rgba), 'kind': kind})
    return shapes


def link_frames(body):
    """World poses of every link: inertial (COM) frame and URDF link frame."""
    com = p.getBasePositionAndOrientation(body)
    inertial = p.getDynamicsInfo(body, -1)[3:5]
    urdf = p.multiplyTransforms(*com, *p.invertTransform(*inertial))
    frames = {-1: {'com': [list(x) for x in com], 'urdf': [list(x) for x in urdf]}}
    for link in range(p.getNumJoints(body)):
        state = p.getLinkState(body, link, computeForwardKinematics=True)
        frames[link] = {'com': [list(state[0]), list(state[1])], 'urdf': [list(state[4]), list(state[5])]}
    return frames


class Log:
    """Tip, joints, arm skeleton per physics tick; pick/place events; arm pose at grasp."""
    def __init__(self, env):
        self.env, self.tip, self.q, self.skeleton, self.events, self.held = env, [], [], [], {}, False
        self.shapes = {'robot': mesh_shapes(env.robot), 'gripper': mesh_shapes(env.gripper.body)}
        self.pick_frames = None
        self.home_frames = {'robot': link_frames(env.robot), 'gripper': link_frames(env.gripper.body)}

    def __call__(self, env):
        self.tip.append(env.tip())
        self.q.append(env.q.copy())
        chain = [np.array(p.getBasePositionAndOrientation(env.robot)[0])]
        chain += [np.array(p.getLinkState(env.robot, j, computeForwardKinematics=True)[4]) for j in env.joints]
        chain.append(np.array(p.getLinkState(env.robot, env.flange, computeForwardKinematics=True)[4]))
        chain.append(env.tip())
        self.skeleton.append(np.array(chain))
        held = env.holding()
        if held != self.held:
            self.events['pick' if held else 'place'] = (len(self.tip)-1, env.tip().tolist())
            if held:
                self.pick_frames = {'robot': link_frames(env.robot), 'gripper': link_frames(env.gripper.body)}
            self.held = held


class Camera:
    def __init__(self, cfg):
        self.view = p.computeViewMatrix(cfg['eye'], cfg['target'], cfg.get('up', (0, 0, 1)))
        self.projection = p.computeProjectionMatrixFOV(cfg['fov'], W/H, .015, 4.)
        self.matrix = np.array(self.projection).reshape(4, 4).T@np.array(self.view).reshape(4, 4).T

    def project(self, points):
        points = np.c_[np.asarray(points), np.ones(len(points))]@self.matrix.T
        ndc = points[:, :3]/points[:, 3:]
        return np.c_[(ndc[:, 0]+1)/2*W, (1-ndc[:, 1])/2*H]


def star(center, radius):
    angles = np.pi/2+np.arange(10)*np.pi/5
    radii = np.where(np.arange(10) % 2 == 0, radius, radius*.42)
    return [(center[0]+r*np.cos(a), center[1]-r*np.sin(a)) for r, a in zip(radii, angles)]


class Video:
    """Live capture at 30 FPS with the tool-tip trail drawn over the scene."""
    def __init__(self, env, log, method, path):
        self.env, self.log, self.method, self.camera = env, log, method, Camera(CAMERA)
        self.writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec='libx264', quality=9,
            macro_block_size=1, ffmpeg_log_level='error', output_params=['-pix_fmt', 'yuv420p'])
        self.writer.send(None)
        self.frames = 0

    def __call__(self, env):
        self.log(env)
        if env.ticks % (PHYSICS_HZ//FPS):
            return
        view, proj = self.camera.view, self.camera.projection
        _, _, rgba, _, seg = p.getCameraImage(W, H, viewMatrix=view, projectionMatrix=proj, renderer=env.renderer,
            shadow=1, lightDirection=[-1, -.6, 2], lightColor=[1, 1, .98], lightAmbientCoeff=.55,
            lightDiffuseCoeff=.65, lightSpecularCoeff=.15, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        rgb = np.asarray(rgba, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
        rgb[np.asarray(seg).reshape(H, W) < 0] = BACKGROUND
        image = Image.fromarray(rgb)
        overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        trail = self.camera.project(np.array(self.log.tip[::2]))
        if len(trail) > 1:
            draw.line([tuple(x) for x in trail], fill=COLOR[self.method]+(235,), width=6, joint='curve')
        for name, color in (('pick', PICK), ('place', PLACE)):
            if name in self.log.events:
                u, v = self.camera.project([self.log.events[name][1]])[0]
                shape = star((u, v), 22) if name == 'pick' else [(u-17, v-13), (u+17, v-13), (u, v+17)]
                draw.polygon(shape, fill=color+(255,), outline=(0, 0, 0, 255), width=2)
        draw.text((48, 36), TITLE[self.method], font=FONT, fill=COLOR[self.method]+(255,))
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        self.writer.send(np.asarray(image).tobytes())
        self.frames += 1

    def close(self):
        self.writer.close()


def episode(sid, candidate, method, speed=None, video=None):
    env = GripperEnv(build(sid, candidate), egl=video is not None)
    rollout = CONTROLLERS[method](env, speed) if method == 'ik' else CONTROLLERS[method](env)
    log = Log(env)
    initial_pick = np.array(p.getBasePositionAndOrientation(env.pick)[0]).tolist()
    recorder = Video(env, log, method, video) if video else None
    env.tick_hook = recorder or log
    if recorder:
        recorder(env)
    result = rollout.run()
    scene = {'pick_initial': initial_pick, 'pick_extent': np.subtract(*p.getAABB(env.pick)[::-1]).tolist(),
             'bowl_aabb': [list(v) for v in p.getAABB(env.place)], 'table_top': .004,
             'geometry': {k: np.asarray(v).tolist() for k, v in getattr(rollout, 'geometry', {}).items()}}
    if recorder:
        recorder.close()
    env.close()
    return result, log, scene


def save(name, method, result, log, scene):
    np.savez_compressed(DATA/f'{name}_{method}.npz', tip=np.array(log.tip), q=np.array(log.q),
                        skeleton=np.array(log.skeleton), hz=PHYSICS_HZ, events=json.dumps(log.events),
                        scene=json.dumps(scene), result=json.dumps(result, default=str),
                        shapes=json.dumps(log.shapes), pick_frames=json.dumps(log.pick_frames),
                        home_frames=json.dumps(log.home_frames))


def compose(left, right, out):
    """Pad the shorter clip with its last frame; 8 px white divider."""
    clips = []
    for path in (left, right):
        reader = imageio_ffmpeg.read_frames(str(path))
        next(reader)
        clips.append([np.frombuffer(f, np.uint8).reshape(H, W, 3) for f in reader])
    length = max(map(len, clips))
    writer = imageio_ffmpeg.write_frames(str(out), (2*W+8, H), fps=FPS, codec='libx264', quality=9,
        macro_block_size=1, ffmpeg_log_level='error', output_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    writer.send(None)
    divider = np.full((H, 8, 3), 255, np.uint8)
    for i in range(length):
        a, b = (clip[min(i, len(clip)-1)] for clip in clips)
        writer.send(np.hstack((a, divider, b)).tobytes())
    writer.close()
    return length


def relog(ids):
    """Deterministic rerun (no video) to refresh logs; refuses if any path changed."""
    for sid in ids:
        candidate = json.loads((WORK/'qualify'/f'scene{sid:02d}.json').read_text())['selected']
        name = f'{build(sid).robot}_env{build(sid).env}'
        for method in ('ours', 'smooth', 'ik'):
            old = np.load(DATA/f'{name}_{method}.npz')
            speed = json.loads(str(old['result'])).get('ik_speed_exact')
            speed = speed or next(json.loads(line)['ik_speed'] for line in open(DATA/'SUMMARY.jsonl')
                                  if json.loads(line)['name'] == name)
            result, log, scene = episode(sid, candidate, method, speed if method == 'ik' else None)
            reference = old['tip'][1:] if len(old['tip']) == len(log.tip)+1 else old['tip']  # video runs log one extra initial sample
            difference = float(np.abs(np.array(log.tip)-reference).max()) if len(log.tip) == len(reference) else np.inf
            if difference > 1e-6:
                raise RuntimeError(f'{name} {method}: trajectory changed by {difference}')
            save(name, method, result, log, scene)
            print(name, method, 'identical, max diff', difference, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', default='')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--relog', action='store_true', help='rerun without video; check trajectories match, rewrite logs')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    # Two environments per robot (env1, env2).
    ids = [int(s) for s in args.scenes.split(',') if s] or [i for i in range(1, len(TASKS)+1) if build(i).env in (1, 2)]
    if args.relog:
        return relog(ids)
    for sid in ids:
        candidate = json.loads((WORK/'qualify'/f'scene{sid:02d}.json').read_text())['selected']
        name = f'{build(sid).robot}_env{build(sid).env}'
        tmp = {m: DATA/f'{name}_{m}.partial.mp4' for m in ('ik', 'ours')}
        ours = episode(sid, candidate, 'ours', video=None if args.no_video else tmp['ours'])
        save(name, 'ours', *ours)
        save(name, 'smooth', *episode(sid, candidate, 'smooth'))
        target, speed = ours[0]['seconds'], .16
        for _ in range(4):
            ik = episode(sid, candidate, 'ik', speed)
            if abs(ik[0]['seconds']/target-1) < .01:
                break
            fixed = .65+.40+1.10  # grasp, release and idle holds are common to both
            speed *= (ik[0]['seconds']-fixed)/max(target-fixed, .1)
        if not args.no_video:
            ik = episode(sid, candidate, 'ik', speed, video=tmp['ik'])
        ik[0]['ik_speed_exact'] = speed  # unrounded, so the run can be reproduced exactly
        save(name, 'ik', *ik)
        frames = None
        if not args.no_video:
            out = VIDEOS/f'{name}_ik_vs_ours.mp4'
            frames = compose(tmp['ik'], tmp['ours'], out)
            for path in tmp.values():
                path.unlink()
        row = {'name': name, 'ours_pass': ours[0]['overall_pass'], 'ik_pass': ik[0]['overall_pass'],
               'ours_s': round(ours[0]['seconds'], 2), 'ik_s': round(ik[0]['seconds'], 2), 'ik_speed': round(speed, 4),
               'frames': frames, 'ik_failure': ik[0].get('failure') or ik[0]['unwanted_contacts'][:3],
               'ours_failure': ours[0].get('failure') or ours[0]['unwanted_contacts'][:3]}
        print(row, flush=True)
        with open(DATA/'SUMMARY.jsonl', 'a') as handle:
            handle.write(json.dumps(row)+'\n')


if __name__ == '__main__':
    main()
