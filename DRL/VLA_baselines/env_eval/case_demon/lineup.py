"""Paper figure: the five bare arms (no gripper/tool), one pose each, side by side.

URDFs come from /home/uet/Documents. Each arm is rendered alone with the same
camera and metric scale, supersampled, cut out with the segmentation mask, and
composed on white (plus a transparent-background copy).
"""
import argparse
import importlib.util
from pathlib import Path

import re

import cv2
import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation

from env_eval.case_demon.robots import KINOVA_COLORS, PANDA_COLORS

DOCS = Path('/home/uet/Documents')
OUT = Path('/home/uet/Downloads/robot_lineup')
CACHE = Path(__file__).resolve().parents[2]/'eval/case_demon_work/lineup_meshes'
# name, urdf, bending joints searched (others fixed), fixed values, tool link, colours
ARMS = [
    ('UR5e', DOCS/'drl_ur5/asset/ur5e/ur5e.urdf', (1, 2, 3), {4: -np.pi/2}, 9, ()),
    ('Franka Panda', DOCS/'drl_panda/asset/panda/panda.urdf', (1, 3, 5), {6: np.pi/4}, 6, PANDA_COLORS),
    ('KUKA LBR iiwa', DOCS/'drl_kuka/asset/kuka/kuka.urdf', (1, 3, 5), {}, 6, ()),
    ('Kinova Gen3', DOCS/'drl_kinova/asset/kinova/kinova.urdf', (1, 3, 5), {}, 7, KINOVA_COLORS),
    ('UFACTORY 850', DOCS/'drl_uf850_new2/asset/uf850/uf850.urdf', (1, 2, 4), {}, 6, ()),
]
REACH, HEIGHT, PITCH = .48, .40, np.radians(30)  # tool target as fractions of reach
AZIMUTH = np.radians(-150)  # common reach direction in the world, towards front-left


def smoothed_urdf(urdf):
    """Copy of the URDF whose STL visuals become smooth-shaded OBJs (crease 30 deg).

    Faceted STL normals otherwise show every triangle. Originals are untouched.
    """
    text = urdf.read_text()
    if '.stl' not in text.lower():
        return urdf
    folder = CACHE/urdf.stem
    folder.mkdir(parents=True, exist_ok=True)
    def visual(block):
        def swap(match):
            source = (urdf.parent/match.group(1)).resolve()
            target = folder/(source.stem+'.obj')
            if not target.exists():
                mesh = trimesh.load(source, force='mesh')
                mesh.merge_vertices()
                trimesh.graph.smooth_shade(mesh, angle=np.radians(30)).export(target, include_normals=True)
            return f'filename="{target}"'
        return re.sub(r'filename="([^"]+\.(?:stl|STL))"', swap, block.group(0))
    text = re.sub(r'<visual>.*?</visual>', visual, text, flags=re.S)
    # Collision meshes keep absolute paths to the originals.
    text = re.sub(r'filename="(?!/)([^"]+)"', lambda m: f'filename="{(urdf.parent/m.group(1)).resolve()}"', text)
    out = folder/urdf.name
    out.write_text(text)
    return out


def display_pose(robot, joints, bend, fixed, tool):
    """Planar 'ready to pick' pose: only the pitch joints move, so nothing twists.

    Random search for: tool at a common fraction of the arm's reach, tilted
    PITCH from vertical towards the reach direction, elbow raised.
    """
    info = [p.getJointInfo(robot, j) for j in joints]
    limits = [(i[8], i[9]) if i[8] < i[9] else (-np.pi, np.pi) for i in info]
    def set_pose(q):
        for j, v in zip(joints, q):
            p.resetJointState(robot, j, v)
        state = p.getLinkState(robot, tool, computeForwardKinematics=True)
        return np.array(state[4]), Rotation.from_quat(state[5]).apply((0, 0, 1))
    q0 = np.zeros(len(joints))
    for index, value in fixed.items():
        q0[index] = value
    shoulder = np.array(p.getLinkState(robot, joints[bend[0]], computeForwardKinematics=True)[4])
    rng = np.random.default_rng(0)
    reach = 0.
    for _ in range(800):
        q = q0.copy()
        for k in bend:
            q[k] = rng.uniform(max(limits[k][0], -np.pi), min(limits[k][1], np.pi))
        reach = max(reach, np.linalg.norm(set_pose(q)[0]-shoulder))
    elbow = joints[bend[1]]
    def cost(q):
        tip, axis = set_pose(q)
        radial = np.linalg.norm(tip[:2])
        direction = np.r_[tip[:2]/max(radial, 1e-9), 0.]
        pointing = np.sin(PITCH)*direction-np.r_[0., 0., np.cos(PITCH)]
        elbow_z = p.getLinkState(robot, elbow, computeForwardKinematics=True)[4][2]
        # Hard requirement: an elbow-up silhouette, clearly above shoulder and tool.
        low = max(0., shoulder[2]+.2*reach-elbow_z)+max(0., tip[2]+.3*reach-elbow_z)
        return (abs(radial-REACH*reach)+abs(tip[2]-HEIGHT*reach)+np.linalg.norm(axis-pointing)
                +4*low-.05*max(0., elbow_z-tip[2]))
    best, best_cost = q0, np.inf
    for scale in (np.pi, .4, .12, .04):
        for _ in range(3000 if scale == np.pi else 1200):
            q = best.copy()
            for k in bend:
                q[k] = (rng.uniform(limits[k][0], limits[k][1]) if scale == np.pi
                        else np.clip(best[k]+rng.normal(0, scale), *limits[k]))
            c = cost(q)
            if c < best_cost:
                best, best_cost = q, c
    tip, axis = set_pose(best)
    return best, float(best_cost), reach


def connect():
    p.connect(p.DIRECT)
    spec = importlib.util.find_spec('eglRenderer')
    if not spec or p.loadPlugin(spec.origin, '_eglRendererPlugin') < 0:
        raise RuntimeError('EGL renderer unavailable; refusing a low-quality CPU render')


TILES = 4  # PyBullet's EGL path silently caps near 2048 px; render a 4x4 mosaic


def tiled_image(camera, size):
    """Full-resolution image from TILES x TILES off-axis frusta of one camera."""
    eye, target, fov = camera
    view = p.computeViewMatrix(eye, target, (0, 0, 1))
    near, far = 2., 7.
    top = near*np.tan(np.radians(fov)/2)
    right = top*size[0]/size[1]
    tw, th = size[0]//TILES, size[1]//TILES
    rgb = np.zeros((th*TILES, tw*TILES, 3), np.uint8)
    seg = np.zeros((th*TILES, tw*TILES), np.int64)
    for row in range(TILES):
        for col in range(TILES):
            left = -right+2*right*col/TILES
            upper = top-2*top*row/TILES
            projection = p.computeProjectionMatrix(left, left+2*right/TILES, upper-2*top/TILES, upper, near, far)
            _, _, rgba, _, mask = p.getCameraImage(tw, th, viewMatrix=view, projectionMatrix=projection,
                renderer=p.ER_BULLET_HARDWARE_OPENGL, shadow=0, lightDirection=[1.2, -1.6, 2.4],
                lightColor=[1, 1, 1], lightAmbientCoeff=.62, lightDiffuseCoeff=.55, lightSpecularCoeff=.18)
            rgb[row*th:(row+1)*th, col*tw:(col+1)*tw] = np.asarray(rgba, np.uint8).reshape(th, tw, 4)[:, :, :3]
            seg[row*th:(row+1)*th, col*tw:(col+1)*tw] = np.asarray(mask).reshape(th, tw)
    return rgb, seg


def render_arm(urdf, bend, fixed, tool, colors, size, camera):
    p.resetSimulation()
    robot = p.loadURDF(str(urdf), (0, 0, 0), useFixedBase=True)
    joints = [i for i in range(p.getNumJoints(robot)) if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]
    pose, error, reach = display_pose(robot, joints, bend, fixed, tool)
    # URDF colours, re-applied after swapping in the smoothed OBJ visuals.
    original = {v[1]: v[7] for v in p.getVisualShapeData(robot)}
    flat = np.array(p.getLinkState(robot, tool, computeForwardKinematics=True)[4][:2])
    yaw = AZIMUTH-np.arctan2(flat[1], flat[0])
    p.removeBody(robot)
    robot = p.loadURDF(str(smoothed_urdf(urdf)), (0, 0, 0), p.getQuaternionFromEuler((0, 0, yaw)),
                       useFixedBase=True, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    for joint, value in zip(joints, pose):
        p.resetJointState(robot, joint, value)
    for link, rgba in original.items():
        p.changeVisualShape(robot, link, rgbaColor=rgba)
    for link, rgba in colors:
        p.changeVisualShape(robot, link, rgbaColor=rgba)
    rgb, seg = tiled_image(camera, size)
    alpha = (seg == robot).astype(np.float32)
    return rgb, alpha, {'pose_rad': [round(float(v), 4) for v in pose], 'pose_cost': error, 'reach_m': reach,
                        'base_yaw_rad': float(yaw)}


def downsample(rgb, alpha, factor):
    """Area-average premultiplied colour so silhouettes are anti-aliased."""
    h, w = alpha.shape
    size = (w//factor, h//factor)
    premultiplied = cv2.resize(rgb.astype(np.float32)*alpha[..., None], size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    color = premultiplied/np.maximum(a[..., None], 1e-6)
    return np.clip(color, 0, 255), a


def crop(color, alpha, pad):
    ys, xs = np.nonzero(alpha > .01)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    return color[y0:y1, x0:x1], alpha[y0:y1, x0:x1], (y0, y1)


def compose(tiles, ground, vertical, gap, margin):
    """tiles: (color, alpha, base_row) cropped; base_row aligns each robot's base."""
    if vertical:
        width = max(t[0].shape[1] for t in tiles)+2*margin
        height = sum(t[0].shape[0] for t in tiles)+gap*(len(tiles)-1)+2*margin
    else:
        above = max(t[2] for t in tiles)
        below = max(t[0].shape[0]-t[2] for t in tiles)
        width = sum(t[0].shape[1] for t in tiles)+gap*(len(tiles)-1)+2*margin
        height = above+below+2*margin
    canvas = np.zeros((height, width, 3), np.float32)
    coverage = np.zeros((height, width), np.float32)
    cursor = margin
    for color, alpha, base in tiles:
        h, w = alpha.shape
        if vertical:
            y, x = cursor, (width-w)//2
            cursor += h+gap
        else:
            y, x = margin+above-base, cursor
            cursor += w+gap
        a = alpha[..., None]
        region = (slice(y, y+h), slice(x, x+w))
        canvas[region] = canvas[region]*(1-a)+color*a
        coverage[region] = coverage[region]*(1-alpha)+alpha
    return canvas, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true', help='fast low-resolution preview')
    args = parser.parse_args()
    factor = 1 if args.draft else 2
    size = (700, 900) if args.draft else (2600, 3380)
    # One shared camera and scale: front-right three-quarter view, slightly high.
    camera = ((2.3, -3.3, 1.35), (0., 0., .48), 21.)
    connect()
    tiles = []
    for name, urdf, bend, fixed, tool, colors in ARMS:
        rgb, alpha, info = render_arm(urdf, bend, fixed, tool, colors, (size[0]*factor, size[1]*factor), camera)
        color, a = downsample(rgb, alpha, factor) if factor > 1 else (rgb.astype(np.float32), alpha)
        color, a, (y0, y1) = crop(color, a, 0)
        # Base row: lowest silhouette row is the base plate on the floor.
        tiles.append((color, a, a.shape[0]))
        print(name, 'tile', a.shape, info, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = '_draft' if args.draft else ''
    scale = size[0]/700
    for vertical, label in ((False, 'row'), (True, 'column')):
        color, coverage = compose(tiles, None, vertical, gap=int(60*scale), margin=int(30*scale))
        white = color*coverage[..., None]+255*(1-coverage[..., None])
        cv2.imwrite(str(OUT/f'robots_{label}{tag}.png'), cv2.cvtColor(np.clip(white, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        rgba = np.dstack((np.clip(color, 0, 255), 255*coverage)).astype(np.uint8)
        cv2.imwrite(str(OUT/f'robots_{label}_transparent{tag}.png'), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        print(label, white.shape, flush=True)
    p.disconnect()


if __name__ == '__main__':
    main()
