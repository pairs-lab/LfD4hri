"""Gripper comparison figures and smoothness table (IK vs Ours; Ours-no-jitter ablation).

Figures: end-effector paths in 3D, IK (top row) vs Ours (bottom row), five arms,
drawn over the real arm meshes (translucent). Per environment the arm is shown at
the grasp; in the combined figure once, at the shared home pose. Env 1 is solid
with filled markers, env 2 dashed, lighter, with hollow markers.
Metrics per motion phase (reach, transport), defined by the same grasp and
release events for every controller.
"""
import csv
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from env_eval.case_demon.plot_compare import kinematics, ldlj, sparc
from env_eval.case_demon.robots import ROBOTS
from env_eval.case_demon.run import WORK
from scipy.signal import savgol_filter

DATA = WORK/'compare_gripper'
OUT = Path('/home/uet/Downloads/trajectory_compare_gripper')
LABELS = {'Panda': 'Franka Panda', 'KUKA': 'KUKA LBR iiwa', 'Kinova': 'Kinova Gen3', 'UR5e': 'UR5e', 'UF850': 'UFACTORY 850'}
COLOR = {'ik': '#D55E00', 'ours': '#0072B2', 'smooth': '#56B4E9'}
LIGHT_COLOR = {'ik': '#F2A65A', 'ours': '#56B4E9'}  # env 2 in the combined figure
NAME = {'ik': 'IK', 'ours': 'Ours', 'smooth': 'Ours w/o jitter'}
ENVS = (1, 2)
STYLE = {1: '-', 2: (0, (3.2, 1.6))}
GRIPPER_RGB = (.13, .13, .15)
LIGHT = np.array([.45, -.65, .62])/np.linalg.norm([.45, -.65, .62])
ZOOM = 1.15  # panels are cropped tightly to the arm, bowl and paths
ARM_ALPHA = .45  # translucent enough that paths behind the arm stay visible
PICK, PLACE = '#E69F00', '#009E73'


def load(robot, env, method):
    d = np.load(DATA/f'{robot}_env{env}_{method}.npz')
    return {'robot': robot, 'tip': d['tip'], 'q': d['q'], 'skeleton': d['skeleton'], 'hz': float(d['hz']),
            'events': json.loads(str(d['events'])), 'scene': json.loads(str(d['scene'])),
            'result': json.loads(str(d['result'])), 'shapes': json.loads(str(d['shapes'])),
            'pick_frames': json.loads(str(d['pick_frames'])), 'home_frames': json.loads(str(d['home_frames']))}


def phases(run):
    """Reach (start->grasp) and transport (grasp->release); the episode ends idle."""
    tip, hz, events = run['tip'], run['hz'], run['events']
    speed = np.linalg.norm(kinematics(tip, hz)[0], axis=1)
    start = int(np.argmax(speed > .005))
    return [(start, events['pick'][0]), (events['pick'][0], events['place'][0])]


def movement_units(speed, threshold=.1):
    """Movement units: separate motions split where speed falls below 10% of peak."""
    moving = speed > threshold*speed.max()
    return int(np.sum(moving[1:] & ~moving[:-1])+moving[0])


def metrics(run):
    tip, hz = run['tip'], run['hz']
    velocity, _, jerk = kinematics(tip, hz)
    speed = np.linalg.norm(velocity, axis=1)
    rows = {'LDLJ': [], 'SPARC': [], 'units': []}
    for a, b in phases(run):
        rows['LDLJ'].append(ldlj(velocity[a:b], jerk[a:b], hz))
        rows['SPARC'].append(sparc(speed[a:b], hz))
        rows['units'].append(movement_units(speed[a:b]))
    joint_jerk = savgol_filter(run['q'], 25, 3, deriv=3, delta=1/hz, axis=0)
    slip = run['result']['suction_firmness']
    return {'success': bool(run['result']['overall_pass']), 'duration_s': len(tip)/hz,
            'path_m': float(np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1))),
            'units': float(np.sum(rows['units'])), 'LDLJ': float(np.mean(rows['LDLJ'])),
            'SPARC': float(np.mean(rows['SPARC'])), 'peak_joint_jerk': float(np.abs(joint_jerk).max()),
            'slip_deg': slip['object_deg'], 'slip_mm': slip['object_mm']}


def table():
    rows = [{'robot': r, 'env': e, 'method': m, **metrics(load(r, e, m))}
            for r in ROBOTS for e in ENVS for m in ('ik', 'ours', 'smooth')]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT/'smoothness_per_episode.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = [('success', 'Success', '{:.0f}/10'), ('duration_s', 'Time (s)', '{:.1f}'), ('path_m', 'Path (m)', '{:.2f}'),
            ('units', r'Mov.\ units $\downarrow$', '{:.1f}'), ('LDLJ', r'LDLJ $\uparrow$', '{:.2f}'),
            ('SPARC', r'SPARC $\uparrow$', '{:.2f}'), ('peak_joint_jerk', r'Peak jerk (rad/s$^3$) $\downarrow$', '{:.0f}'),
            ('slip_deg', r'Object slip ($^\circ$) $\downarrow$', '{:.2f}')]
    lines = [r'\begin{tabular}{l'+'c'*len(keys)+'}', r'\toprule',
             'Method & '+' & '.join(k[1] for k in keys)+r' \\', r'\midrule']
    summary = {}
    for method in ('ik', 'smooth', 'ours'):
        sel = [r for r in rows if r['method'] == method]
        cells, summary[method] = [], {}
        for key, _, fmt in keys:
            values = np.array([r[key] for r in sel], float)
            summary[method][key] = [float(values.mean()), float(values.std())]
            cells.append(fmt.format(values.sum()) if key == 'success' else fmt.format(values.mean())+r' $\pm$ '+fmt.format(values.std()))
        lines.append(NAME[method]+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT/'smoothness_table.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'smoothness_summary.json').write_text(json.dumps(summary, indent=1))
    return summary


_MESHES = {}


def mesh(path, scale):
    """Load once, decimate to a light, smooth-looking triangle set."""
    key = (path, tuple(scale))
    if key not in _MESHES:
        m = trimesh.load(path, force='mesh', process=True)
        m.apply_scale(scale)
        if len(m.faces) > 2400:
            o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
            o = o.simplify_quadric_decimation(2400)
            m = trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles), process=True)
        _MESHES[key] = (np.asarray(m.vertices), np.asarray(m.faces))
    return _MESHES[key]


_COLOURS = {}


def link_colours(robot):
    """Per-link RGB exactly as the videos show the arm: URDF colours + robots.py overrides."""
    if robot not in _COLOURS:
        import pybullet as p
        client = p.connect(p.DIRECT)
        body = p.loadURDF(str(ROBOTS[robot].urdf), useFixedBase=True, physicsClientId=client)
        colours = {v[1]: tuple(v[7][:3]) for v in p.getVisualShapeData(body, physicsClientId=client)}
        p.disconnect(client)
        colours.update({link: tuple(rgba[:3]) for link, rgba in ROBOTS[robot].colors})
        _COLOURS[robot] = colours
    return _COLOURS[robot]


_EXTRA = {}


def urdf_fallback_shapes(robot, recorded):
    """STL meshes straight from the URDF for links PyBullet gave nothing drawable.

    The UR5e URDF has .dae visuals (no reader installed) and its collision meshes
    commented out, so PyBullet loads neither; the STL files are still listed.
    Their origins are relative to the URDF link frame.
    """
    if robot not in _EXTRA:
        import re
        import pybullet as p
        urdf = Path(ROBOTS[robot].urdf)
        text = urdf.read_text()
        client = p.connect(p.DIRECT)
        body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
        names = {-1: p.getBodyInfo(body, physicsClientId=client)[0].decode()}
        names.update({i: p.getJointInfo(body, i, physicsClientId=client)[12].decode()
                      for i in range(p.getNumJoints(body, physicsClientId=client))})
        p.disconnect(client)
        have = {shape['link'] for shape in recorded}
        extra = []
        for link, name in names.items():
            block = re.search(r'<link\s+name="%s".*?</link>' % re.escape(name), text, re.S)
            if link in have or not block:
                continue
            for collision in re.findall(r'<collision>.*?</collision>', block.group(0), re.S):
                mesh_file = re.search(r'filename="([^"]+\.stl)"', collision, re.I)
                if not mesh_file:
                    continue
                origin = re.search(r'<origin([^>]*)/>', collision)
                xyz = re.search(r'xyz="([^"]+)"', origin.group(1)) if origin else None
                rpy = re.search(r'rpy="([^"]+)"', origin.group(1)) if origin else None
                scale = re.search(r'scale="([^"]+)"', collision)
                extra.append({'link': link, 'file': str((urdf.parent/mesh_file.group(1)).resolve()),
                              'scale': list(map(float, scale.group(1).split())) if scale else [1., 1., 1.],
                              'pos': list(map(float, xyz.group(1).split())) if xyz else [0., 0., 0.],
                              'orn': Rotation.from_euler('xyz', list(map(float, rpy.group(1).split())) if rpy else [0, 0, 0]).as_quat().tolist(),
                              'rgba': [.7, .7, .7, 1.], 'kind': 'visual'})
        _EXTRA[robot] = extra
    return _EXTRA[robot]


def frames_at(run, tick):
    """Link frames at any logged tick, rebuilt by forward kinematics from the logged
    joints; gripper links keep their logged pose relative to the flange at grasp."""
    import pybullet as p
    from env_eval.case_demon.gripper import FLANGE
    spec = ROBOTS[run['robot']]
    client = p.connect(p.DIRECT)
    body = p.loadURDF(str(spec.urdf), (0, 0, 0), p.getQuaternionFromEuler((0, 0, spec.base_yaw)),
                      useFixedBase=True, physicsClientId=client)
    joints = [i for i in range(p.getNumJoints(body, physicsClientId=client))
              if p.getJointInfo(body, i, physicsClientId=client)[2] == p.JOINT_REVOLUTE]
    for joint, value in zip(joints, run['q'][tick]):
        p.resetJointState(body, joint, float(value), physicsClientId=client)
    robot = {}
    com = p.getBasePositionAndOrientation(body, physicsClientId=client)
    inertial = p.getDynamicsInfo(body, -1, physicsClientId=client)[3:5]
    robot['-1'] = {'com': [list(x) for x in com], 'urdf': [list(x) for x in p.multiplyTransforms(*com, *p.invertTransform(*inertial))]}
    for link in range(p.getNumJoints(body, physicsClientId=client)):
        state = p.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=client)
        robot[str(link)] = {'com': [list(state[0]), list(state[1])], 'urdf': [list(state[4]), list(state[5])]}
    p.disconnect(client)
    # Fingers as they really were: open (home log) before the grasp, closed after.
    grasped = tick >= run['events']['pick'][0]
    reference = run['pick_frames'] if grasped else run['home_frames']
    reference_tip = run['tip'][run['events']['pick'][0] if grasped else 0]
    flange = str(FLANGE[run['robot']])
    to_old = p.invertTransform(*reference['robot'][flange]['urdf'])
    new = robot[flange]['urdf']
    gripper = {}
    for link, frame in reference['gripper'].items():
        gripper[link] = {key: [list(x) for x in p.multiplyTransforms(*new, *p.multiplyTransforms(*to_old, *pose))]
                         for key, pose in frame.items()}
    pad = p.multiplyTransforms(*new, *p.multiplyTransforms(*to_old, list(reference_tip), (0, 0, 0, 1)))[0]
    return {'robot': robot, 'gripper': gripper, 'pad_centre': list(pad)}


def transport_tick(run):
    """Midpoint in time between grasp and release."""
    return (run['events']['pick'][0]+run['events']['place'][0])//2


def arm_polygons(run, frames_key='pick_frames'):
    """World-space triangles and shaded colours of arm + gripper."""
    polygons, colours = [], []
    table = link_colours(run['robot'])
    pose = run[frames_key] if isinstance(frames_key, str) else frames_key
    for body in ('robot', 'gripper'):
        frames = pose[body]
        shapes = run['shapes'][body]
        if body == 'robot':
            shapes = shapes+urdf_fallback_shapes(run['robot'], shapes)
        for shape in shapes:
            # Visual frames are given relative to the URDF link frame, collision to the COM frame.
            frame_pos, frame_orn = frames[str(shape['link'])]['urdf' if shape['kind'] == 'visual' else 'com']
            rotation = Rotation.from_quat(frame_orn)*Rotation.from_quat(shape['orn'])
            origin = np.array(frame_pos)+Rotation.from_quat(frame_orn).apply(shape['pos'])
            vertices, faces = mesh(shape['file'], shape['scale'])
            triangles = (rotation.apply(vertices)+origin)[faces]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            base = np.array(GRIPPER_RGB if body == 'gripper' else table.get(shape['link'], (.75, .75, .75)))
            base = np.minimum(base, .90)  # white shells keep a visible tone on white paper
            shade = .38+.62*np.abs(normals@LIGHT)
            polygons.append(triangles)
            colours.append(np.clip(base[None]*shade[:, None], 0, 1))
    return np.vstack(polygons), np.vstack(colours)


def draw_scene(ax, run, limits, env, combined):
    (x0, x1), (y0, y1), _ = limits
    scene = run['scene']
    style = STYLE[env] if combined else '-'
    low, high = map(np.array, scene['bowl_aabb'])
    centre, radius = (low[:2]+high[:2])/2, (high[0]-low[0])/2
    t = np.linspace(0, 2*np.pi, 60)
    ax.plot(centre[0]+radius*np.cos(t), centre[1]+radius*np.sin(t), .005, color=PLACE, lw=.8, ls=style, alpha=.8)
    pick = np.array(scene['pick_initial'][:2])
    half = max(scene['pick_extent'][:2])/2
    square = pick+half*np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(square[:, 0], square[:, 1], .005, color=PICK, lw=.9, ls=style)


# Version 1 (one env per figure): every panel samples the arm at its own phase of
# the motion, so all ten arms differ. Index = column*2+row (row 0 IK, 1 Ours).
PHASES = (.30, .66, .52, .84, .22, .58, .72, .40, .88, .46)


def motion_span(run):
    """Ticks from motion onset to release (the idle hold after release is excluded)."""
    speed = np.linalg.norm(np.diff(run['tip'], axis=0), axis=1)*run['hz']
    return int(np.argmax(speed > .005)), run['events']['place'][0]


def tick_at(run, fraction):
    start, end = motion_span(run)
    return int(round(start+fraction*(end-start)))


def arm_at(run, tick):
    """Arm + gripper triangles at a logged tick, cached on the run (with pad centre)."""
    cache = run.setdefault('arm_cache', {})
    if tick not in cache:
        frames = frames_at(run, tick)
        cache[tick] = arm_polygons(run, frames)+(np.array(frames['pad_centre']),)
    return cache[tick]


def screen_basis(azimuth, elevation=24.):
    a, e = np.radians(azimuth), np.radians(elevation)
    return np.array([[-np.sin(a), np.cos(a), 0.], [-np.sin(e)*np.cos(a), -np.sin(e)*np.sin(a), np.cos(e)]])


def projected_arm(run, tick, basis):
    """Screen-plane samples along the distal half of the arm (elbow to gripper).

    Both arms share the base and shoulder, so only the distal links can separate."""
    chain = run['skeleton'][tick]
    chain = chain[len(chain)//2:]
    samples = np.vstack([a+(b-a)*t for a, b in zip(chain[:-1], chain[1:]) for t in np.linspace(0, 1, 8)])
    return samples@basis.T


def screen_gap(one, tick_one, two, tick_two, basis):
    """Closest approach of the two arms' distal halves on screen (reported in the audit)."""
    a, b = projected_arm(one, tick_one, basis), projected_arm(two, tick_two, basis)
    return float(np.min(np.linalg.norm(a[:, None]-b[None], axis=2)))


def screen_separation(one, tick_one, two, tick_two, basis):
    """Mean on-screen distance between corresponding joints beyond the shoulder:
    the whole upper arm, not only the gripper, has to stand apart."""
    a, b = one['skeleton'][tick_one][3:]@basis.T, two['skeleton'][tick_two][3:]@basis.T
    return float(np.linalg.norm(a-b, axis=1).mean())


def closing(run, tick):
    """True while the fingers are closing (their drawn state would be ambiguous)."""
    pick = run['events']['pick'][0]
    return pick-.65*run['hz'] <= tick < pick


def pair_phases(runs, method, basis, exclude=()):
    """Version 2: env-1 / env-2 phases whose arms stand furthest apart ON SCREEN.

    exclude: joint vectors already used (IK's), which these poses must differ
    from by at least 0.15 rad in some joint, in both environments."""
    grid = np.linspace(.12, .94, 22)
    best, score = None, -1.
    one, two = runs[method, 1], runs[method, 2]
    for f1 in grid:
        for f2 in grid:
            t1, t2 = tick_at(one, f1), tick_at(two, f2)
            if closing(one, t1) or closing(two, t2):
                continue
            # Ours' arms must look different from IK's arms of the same scene (on screen).
            if any(np.linalg.norm((one['skeleton'][t1][3:]-a)@basis.T, axis=1).mean() < .12 or
                   np.linalg.norm((two['skeleton'][t2][3:]-b)@basis.T, axis=1).mean() < .12 for a, b in exclude):
                continue
            spread = screen_separation(one, t1, two, t2, basis)
            if screen_gap(one, t1, two, t2, basis) > .04 and spread > score:
                best, score = (float(f1), float(f2)), spread
    return best


def draw_arm(ax, run, tick, alpha=ARM_ALPHA):
    polygons, colours, _ = arm_at(run, tick)
    collection = Poly3DCollection(polygons, facecolors=np.c_[colours, np.full(len(colours), alpha)],
                                  edgecolors='none', linewidths=0)
    collection.set_rasterized(True)
    ax.add_collection3d(collection)


def draw_path(ax, run, method, env, combined):
    tip, events = run['tip'], run['events']
    second = combined and env == 2
    colour = LIGHT_COLOR[method] if second else COLOR[method]
    halo = [effects.Stroke(linewidth=3.3, foreground='white'), effects.Normal()]
    ax.plot(*tip[::2].T, color=colour, lw=1.5, ls=STYLE[2] if second else '-', path_effects=halo)
    ax.scatter(*tip[0], s=14, color='k', depthshade=False)
    fill = (lambda c: 'white') if second else (lambda c: c)
    ax.scatter(*events['pick'][1], s=90, marker='*', facecolor=fill(PICK), edgecolor=PICK if second else 'k',
               linewidth=1. if second else .45, depthshade=False)
    ax.scatter(*events['place'][1], s=40, marker='v', facecolor=fill(PLACE), edgecolor=PLACE if second else 'k',
               linewidth=1. if second else .45, depthshade=False)


def best_azimuth(runs, elevation=26.):
    """Azimuth (-80..-20 deg) that spreads the column's paths most on screen."""
    points = np.vstack([r['tip'][::10] for r in runs])
    points = points-points.mean(0)
    best, score = -56., -1.
    for azimuth in np.arange(-80., -19., 4.):
        a, e = np.radians(azimuth), np.radians(elevation)
        right = np.array([-np.sin(a), np.cos(a), 0.])
        up = np.array([-np.sin(e)*np.cos(a), -np.sin(e)*np.sin(a), np.cos(e)])
        screen = np.c_[points@right, points@up]
        spread = np.linalg.det(np.cov(screen.T))
        if spread > score:
            best, score = azimuth, spread
    return best


def floor_for(runs):
    """Floor patch under the task: paths, objects and bowls plus 5 cm, on the 0.1 m grid."""
    xy = [r['tip'][:, :2] for r in runs]
    for r in runs:
        low, high = np.array(r['scene']['bowl_aabb'])[:, :2]
        pick, half = np.array(r['scene']['pick_initial'][:2]), max(r['scene']['pick_extent'][:2])/2
        xy += [np.array([low, high]), np.array([pick-half, pick+half])]
    xy = np.vstack(xy)
    low, high = np.floor((xy.min(0)-.05)/.1)*.1, np.ceil((xy.max(0)+.05)/.1)*.1
    return (low[0], high[0]), (low[1], high[1])


def style_axes(ax, limits, azimuth=-56., zoom=1., floor=None):
    """3D axes box with x/y/z in metres; a light floor patch marks the task area."""
    (x0, x1), (y0, y1), (z0, z1) = limits
    (fx0, fx1), (fy0, fy1) = floor if floor else ((x0, x1), (y0, y1))
    ax.add_collection3d(Poly3DCollection([[[fx0, fy0, .004], [fx1, fy0, .004], [fx1, fy1, .004], [fx0, fy1, .004]]],
                                         facecolor='#dfe3e7', alpha=.55, edgecolor='none'))
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1-x0, y1-y0, z1-z0), zoom=zoom)
    ax.view_init(elev=24, azim=azimuth)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.97, .975, .98, 1.))
        axis.pane.set_edgecolor('#b9bfc6')
        axis._axinfo['grid'].update(color='#dde1e5', linewidth=.4)
        axis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    # The y axis is foreshortened by the view: at most three numbers there.
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.tick_params(labelsize=4.8, pad=-3, length=1.5)
    # Transparent axes background: a neighbouring panel must never hide this one's labels.
    ax.patch.set_visible(False)
    ax.set_xlabel('x (m)', fontsize=5.2, labelpad=-10)
    ax.set_ylabel('y (m)', fontsize=5.2, labelpad=-3)
    ax.set_zlabel('z (m)', fontsize=5.2, labelpad=-2)


def limits_for(runs):
    """Box holding the whole drawn arm, both paths, the object and the bowl."""
    parts = []
    for r in runs:
        parts += [r['tip'], np.array(r['scene']['bowl_aabb'])]
        parts += [arm_at(r, tick)[0].reshape(-1, 3) for tick in r.get('pose_ticks', ())]
        pick, half = np.array(r['scene']['pick_initial']), max(r['scene']['pick_extent'][:2])/2
        parts.append(np.array([pick-[half, half, 0], pick+[half, half, 0]]))
    points = np.vstack(parts)
    low, high = points.min(0)-.015, points.max(0)+.015
    low[2] = 0.
    return list(zip(low, high))


def legend_handles(combined):
    if combined:
        handles = [Line2D([], [], color=COLOR['ik'], lw=1.8, label='IK env 1'),
                   Line2D([], [], color=LIGHT_COLOR['ik'], lw=1.8, ls=STYLE[2], label='IK env 2'),
                   Line2D([], [], color=COLOR['ours'], lw=1.8, label='Ours env 1'),
                   Line2D([], [], color=LIGHT_COLOR['ours'], lw=1.8, ls=STYLE[2], label='Ours env 2')]
    else:
        handles = [Line2D([], [], color=COLOR['ik'], lw=1.8, label='IK'),
                   Line2D([], [], color=COLOR['ours'], lw=1.8, label='Ours')]
    handles += [Line2D([], [], marker='o', ls='', color='k', ms=3.5, label='start'),
                Line2D([], [], marker='*', ls='', mfc=PICK, mec='k', mew=.5, ms=9, label='grasp'),
                Line2D([], [], marker='v', ls='', mfc=PLACE, mec='k', mew=.5, ms=6, label='release')]
    if combined:
        handles += [Line2D([], [], marker='*', ls='', mfc='white', mec=PICK, mew=1., ms=9, label='env 2: hollow')]
    handles += [Line2D([], [], marker='o', ls='', mfc='white', mec='k', mew=1.2, ms=5, label='gripper of the drawn arm (pad centre)')]
    return handles


def panel(ax, robot, method, envs, runs, box, azimuth, combined, zoom):
    style_axes(ax, box, azimuth, zoom, floor_for(list(runs.values())))
    for env in envs:
        run = runs[method, env]
        draw_arm(ax, run, run['panel_tick'][method], ARM_ALPHA if env == envs[0] else ARM_ALPHA*.8)
    for env in envs:
        draw_scene(ax, runs[method, env], box, env, combined)
    for env in envs:
        run = runs[method, env]
        draw_path(ax, run, method, env, combined)
        # Ring: where the drawn arm's gripper is on this path.
        ax.scatter(*arm_at(run, run['panel_tick'][method])[2], s=30, facecolor='white',
                   edgecolor=(LIGHT_COLOR if combined and env == 2 else COLOR)[method], linewidth=1.3, depthshade=False)
        if not run['result']['overall_pass']:
            ax.text2D(.5, .92, 'task failed', transform=ax.transAxes, ha='center', fontsize=6, color='#b00')


def assign_poses(robot, envs, runs, column, audit, azimuth):
    """Choose the tick at which each panel's arm is drawn; record it for the audit."""
    combined = len(envs) > 1
    used = []
    for row, method in enumerate(('ik', 'ours')):
        if combined:
            phases = pair_phases(runs, method, screen_basis(azimuth),
                                 [(runs[m, 1]['skeleton'][runs[m, 1]['panel_tick'][m]][3:],
                                   runs[m, 2]['skeleton'][runs[m, 2]['panel_tick'][m]][3:]) for m in used])
            used.append(method)
        else:
            phases = (PHASES[2*column+row],)
        for env, fraction in zip(envs, phases):
            run = runs[method, env]
            tick = tick_at(run, fraction)
            while closing(run, tick):  # step past the finger-closing window
                fraction = min(fraction+.02, .98)
                tick = tick_at(run, fraction)
            run.setdefault('panel_tick', {})[method] = tick
            run.setdefault('pose_ticks', []).append(tick)
            pad = arm_at(run, tick)[2]
            audit.append({'robot': robot, 'method': method, 'env': env, 'phase': round(fraction, 3), 'tick': tick,
                          'q': [round(float(v), 4) for v in run['q'][tick]],
                          'pad_vs_logged_path_mm': round(1e3*float(np.linalg.norm(pad-run['tip'][tick])), 3)})
        if combined:
            one, two = runs[method, 1], runs[method, 2]
            ticks = one['panel_tick'][method], two['panel_tick'][method]
            audit[-1]['screen_gap_m'] = round(screen_gap(one, ticks[0], two, ticks[1], screen_basis(azimuth)), 4)
            audit[-1]['screen_separation_m'] = round(screen_separation(one, ticks[0], two, ticks[1], screen_basis(azimuth)), 4)


MARKER_PAD = 9.  # display px: radius of the largest marker (grasp star), so markers are not clipped


def content_points(runs, method, envs, box=None):
    """Everything a panel draws, in 3D: arm meshes, paths, bowls, objects, table."""
    parts = []
    if box is not None:
        (x0, x1), (y0, y1) = box
        parts.append(np.array([[x0, y0, .004], [x1, y0, .004], [x1, y1, .004], [x0, y1, .004]]))
    for env in envs:
        run = runs[method, env]
        parts += [arm_at(run, run['panel_tick'][method])[0].reshape(-1, 3)[::7], run['tip'][::5]]
        low, high = map(np.array, run['scene']['bowl_aabb'])
        centre, radius = (low[:2]+high[:2])/2, (high[0]-low[0])/2
        t = np.linspace(0, 2*np.pi, 24)
        parts.append(np.c_[centre[0]+radius*np.cos(t), centre[1]+radius*np.sin(t), np.full(24, .005)])
    return np.vstack(parts)


def pixel_box(ax, points):
    """Screen bounding box (display pixels) of 3D points as drawn in this axes."""
    from mpl_toolkits.mplot3d import proj3d
    x, y, _ = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], ax.get_proj())
    xy = ax.transData.transform(np.c_[x, y])
    return xy.min(0)-MARKER_PAD, xy.max(0)+MARKER_PAD


def layout_problems(fig, panels):
    """Overlap/clipping audit. panels: (name, ax, 3D points). Returns problem list + boxes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes, problems = {}, []
    for name, ax, points in panels:
        low, high = pixel_box(ax, points)
        boxes[name] = (low, high)
        frame = ax.bbox
        if low[0] < frame.x0-1 or low[1] < frame.y0-1 or high[0] > frame.x1+1 or high[1] > frame.y1+1:
            problems.append(f'{name}: content clipped by its panel')
        if ax.title.get_text():
            title = ax.title.get_window_extent(renderer)
            if title.y0 < high[1] and title.x0 < high[0] and title.x1 > low[0]:
                problems.append(f'{name}: title touches the drawing')
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            (la, ha), (lb, hb) = boxes[a], boxes[b]
            if (la < hb).all() and (lb < ha).all():
                problems.append(f'{a} overlaps {b}')
    others = [t.get_window_extent(renderer) for t in fig.texts]+[l.get_window_extent(renderer) for l in fig.legends]
    for name, (low, high) in boxes.items():
        for other in others:
            if other.x0 < high[0] and other.x1 > low[0] and other.y0 < high[1] and other.y1 > low[1]:
                problems.append(f'{name}: touches a label/legend')
    width, height = fig.bbox.width, fig.bbox.height
    for name, (low, high) in boxes.items():
        if low[0] < 0 or low[1] < 0 or high[0] > width or high[1] > height:
            problems.append(f'{name}: outside the figure')
    return problems, {k: [float(v) for v in (*lo, *hi)] for k, (lo, hi) in boxes.items()}


def raster_extent(fig, ax):
    """Pixel box (display coords, origin bottom-left) of what this axes actually draws."""
    others = [a for a in fig.axes if a is not ax]+list(fig.texts)+list(fig.legends)
    states = [o.get_visible() for o in others]
    for o in others:
        o.set_visible(False)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    for o, state in zip(others, states):
        o.set_visible(state)
    ink = np.nonzero((255-image).sum(2) > 12)
    if not len(ink[0]):
        return np.zeros(2), np.zeros(2)
    height = image.shape[0]
    return np.array([ink[1].min(), height-1-ink[0].max()], float), np.array([ink[1].max()+1, height-ink[0].min()], float)


def fit_panels(fig, panels, groups, fill=.96):
    """Centre each panel's drawing in its grid cell and zoom it to fill the cell.

    panels: dicts {name, ax, points, cell (figure-fraction Bbox), group, aspect}.
    3D zoom scales about the axes centre, so the axes are moved (and enlarged,
    their background is off) to put the drawing's centre on the cell centre.
    Panels in one group (IK and Ours of a robot) share one zoom.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    scale = {}
    for item in panels:
        ax = item['ax']
        low, high = raster_extent(fig, ax)
        frame = ax.bbox
        centre_ax = np.array([(frame.x0+frame.x1)/2, (frame.y0+frame.y1)/2])
        # Drawing centre relative to the axes centre (aligned frames later share one offset).
        item['offset'] = (low+high)/2-centre_ax
        item['half'] = (high-low)/2                      # drawing half-size, zoom 1
        cell = item['cell']
        top = cell.y1*height
        if item.get('title_px'):
            top -= item['title_px']
        bottom = cell.y0*height+item.get('bottom_px', 0.)
        item['target'] = np.array([(cell.x0+cell.x1)/2*width, (bottom+top)/2])
        room = np.array([(cell.x1-cell.x0)*width/2, (top-bottom)/2])
        scale.setdefault(item['group'], []).append(float(np.min(room/item['half'])))
    # Aligned frames: one common shift for the whole group (mean offset), so every
    # frame keeps the same position inside its cell; the spread is added to the size.
    common = {}
    for item in panels:
        if item.get('aligned'):
            common.setdefault(item['group'], []).append(item['offset'])
    for group, offsets in common.items():
        mean = np.mean(offsets, axis=0)
        fits = []
        for item in panels:
            if item['group'] == group:
                cell = item['cell']
                room = np.array([(cell.x1-cell.x0)*width/2, (cell.y1-cell.y0)*height/2])
                fits.append(float(np.min(room/(item['half']+np.abs(item['offset']-mean)))))
                item['offset'] = mean
        scale[group] = fits
    for item in panels:
        groups[item['group']] = groups.get(item['group'], fill*min(scale[item['group']]))
    place(fig, panels, groups)


def refit(fig, panels, groups, fill=.96, rounds=2):
    """Re-measure the real drawing after placement and correct zoom and centring."""
    width, height = fig.bbox.width, fig.bbox.height
    for _ in range(rounds):
        ratio = {}
        for item in panels:
            low, high = raster_extent(fig, item['ax'])
            cell = item['cell']
            top = cell.y1*height-item.get('title_px', 0.)
            if item.get('aligned'):
                c = item['target']
                room = np.array([[c[0]-cell.x0*width, cell.x1*width-c[0]], [c[1]-cell.y0*height, top-c[1]]])
                need = np.array([[c[0]-low[0], high[0]-c[0]], [c[1]-low[1], high[1]-c[1]]])
                ratio.setdefault(item['group'], []).append(float(np.min(fill*room/np.maximum(need, 1.))))
                item.setdefault('errors', []).append(((low+high)/2-item['target'])/groups[item['group']])
                continue
            room = np.array([(cell.x1-cell.x0)*width/2, (top-cell.y0*height)/2])
            half = (high-low)/2
            ratio.setdefault(item['group'], []).append(float(np.min(fill*room/np.maximum(half, 1.))))
            # Measured centre error at the current zoom becomes part of the offset.
            item['offset'] = item['offset']+((low+high)/2-item['target'])/groups[item['group']]
        for group in ratio:
            errors = [i['errors'].pop() for i in panels if i['group'] == group and i.get('errors')]
            if errors:  # aligned group: one common centring correction
                shift = np.mean(errors, axis=0)
                for i in panels:
                    if i['group'] == group:
                        i['offset'] = i['offset']+shift
        for group, values in ratio.items():
            groups[group] *= min(values)
        place(fig, panels, groups)


def place(fig, panels, groups):
    width, height = fig.bbox.width, fig.bbox.height
    for item in panels:
        zoom = groups[item['group']]
        ax = item['ax']
        ax.set_box_aspect(item['aspect'], zoom=zoom)
        # Same size as the cell (3D drawing scale follows axes size), only shifted so the
        # drawing is centred; clipping is off, the audit guarantees it stays in its cell.
        cell = item['cell']
        size = np.array([(cell.x1-cell.x0)*width, (cell.y1-cell.y0)*height])
        centre = item['target']-item['offset']*zoom
        ax.set_position([(centre[0]-size[0]/2)/width, (centre[1]-size[1]/2)/height, size[0]/width, size[1]/height])
        for artist in ax.get_children():
            artist.set_clip_on(False)


def layout_check(fig, panels):
    """Every drawing inside its own cell, unclipped, clear of all text and the legend."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    problems, boxes = [], {}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t.get_window_extent(renderer) for t in fig.texts]+[l.get_window_extent(renderer) for l in fig.legends]
    texts += [p['ax'].title.get_window_extent(renderer) for p in panels if p['ax'].title.get_text()]
    for item in panels:
        low, high = raster_extent(fig, item['ax'])
        boxes[item['name']] = [float(v) for v in (*low, *high)]
        cell = item['cell']
        if low[0] < cell.x0*width-1 or high[0] > cell.x1*width+1 or low[1] < cell.y0*height-1 or high[1] > cell.y1*height+1:
            problems.append(f"{item['name']}: leaves its cell")
        for box in texts:
            if box.x0 < high[0] and box.x1 > low[0] and box.y0 < high[1] and box.y1 > low[1]:
                problems.append(f"{item['name']}: touches text")
                break
        if low[0] < 0 or low[1] < 0 or high[0] > width or high[1] > height:
            problems.append(f"{item['name']}: outside the figure")
    return problems, boxes


LAYOUT_AUDIT = {}
ZOOMS = (1.8, 1.7, 1.6, 1.5, 1.42, 1.35, 1.28, 1.22, 1.16, 1.1, 1.05, 1.0, .95, .9)


def save(fig, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f'{tag}.pdf', dpi=1200)
    fig.savefig(OUT/f'{tag}.png', dpi=600)
    plt.close(fig)


def figure(envs, tag):
    """All five arms: IK (top) vs Ours (bottom). Largest zoom that passes the layout audit."""
    combined = len(envs) > 1
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    audit, columns = [], []
    loaded = {robot: {(m, e): load(robot, e, m) for m in ('ik', 'ours') for e in envs} for robot in ROBOTS}
    # One view and one box size for every panel, so all frames are identical and aligned.
    azimuth = best_azimuth([run for runs in loaded.values() for run in runs.values()])
    for col, robot in enumerate(ROBOTS):
        assign_poses(robot, envs, loaded[robot], col, audit, azimuth)
    boxes = {robot: limits_for(list(runs.values())) for robot, runs in loaded.items()}
    spans = np.max([[high-low for low, high in box] for box in boxes.values()], axis=0)
    for robot, runs in loaded.items():
        box = boxes[robot]
        centre = [(low+high)/2 for low, high in box]
        common = [(c-span/2, c+span/2) for c, span in zip(centre[:2], spans[:2])]+[(0., spans[2])]
        columns.append((robot, runs, common, azimuth))
    # Height chosen so cells match the tall drawings: width and height both bind.
    fig = plt.figure(figsize=(7.16, 4.75))
    grid = fig.add_gridspec(2, len(ROBOTS), left=.055, right=.997, top=.955, bottom=.105 if combined else .08,
                            wspace=.02, hspace=.02)
    panels = []
    for col, (robot, runs, box, azimuth) in enumerate(columns):
        floor = floor_for(list(runs.values()))
        for row, method in enumerate(('ik', 'ours')):
            ax = fig.add_subplot(grid[row, col], projection='3d', computed_zorder=False)
            panel(ax, robot, method, envs, runs, box, azimuth, combined, 1.)
            (x0, x1), (y0, y1), (z0, z1) = box
            panels.append({'name': f'{robot} {NAME[method]}', 'ax': ax, 'group': 0, 'aligned': True,
                           'points': content_points(runs, method, envs, floor),
                           'cell': grid[row, col].get_position(fig), 'aspect': (x1-x0, y1-y0, z1-z0),
                           })
    # Robot names as figure text above each column (a 3D title would move with the axes).
    for col, (robot, *_ ) in enumerate(columns):
        cell = grid[0, col].get_position(fig)
        fig.text((cell.x0+cell.x1)/2, cell.y1+.008, LABELS[robot], ha='center', va='bottom', fontsize=8.5)
    for row, method in enumerate(('ik', 'ours')):
        cell = grid[row, 0].get_position(fig)
        fig.text(.026, (cell.y0+cell.y1)/2, NAME[method], va='center', ha='center', fontsize=9.5, fontweight='bold')
    handles = legend_handles(combined)
    fig.legend(handles=handles, loc='lower center', ncol=(len(handles)+1)//2 if combined else len(handles),
               fontsize=6.5, frameon=False, handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(.52, -.005))
    zooms = {}
    fit_panels(fig, panels, zooms)
    refit(fig, panels, zooms)
    for attempt in range(10):
        problems, boxes = layout_check(fig, panels)
        if os.environ.get('LAYOUT_DEBUG'):
            print('attempt', attempt, problems, flush=True)
        if not problems:
            break
        for item in panels:
            if any(problem.startswith(item['name']+':') for problem in problems):
                zooms[item['group']] *= .97
        place(fig, panels, zooms)
    zoom = round(zooms[0], 3)
    LAYOUT_AUDIT[f'trajectory_{tag}'] = {'zoom': zoom, 'problems': problems, 'content_boxes_px': boxes,
                                        'figure_px': [fig.bbox.width, fig.bbox.height]}
    save(fig, f'trajectory_{tag}')
    return audit


def robot_figure(robot, envs, tag):
    """One arm, large: IK | Ours side by side. Largest zoom that passes the layout audit."""
    combined = len(envs) > 1
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    runs = {(m, e): load(robot, e, m) for m in ('ik', 'ours') for e in envs}
    azimuth = best_azimuth(list(runs.values()))
    assign_poses(robot, envs, runs, list(ROBOTS).index(robot), [], azimuth)
    box = limits_for(list(runs.values()))
    fig = plt.figure(figsize=(7.16, 3.9))
    grid = fig.add_gridspec(1, 2, left=.01, right=.99, top=.9, bottom=.16, wspace=.04)
    panels = []
    floor = floor_for(list(runs.values()))
    (x0, x1), (y0, y1), (z0, z1) = box
    for col, method in enumerate(('ik', 'ours')):
        ax = fig.add_subplot(grid[0, col], projection='3d', computed_zorder=False)
        panel(ax, robot, method, envs, runs, box, azimuth, combined, 1.)
        panels.append({'name': NAME[method], 'ax': ax, 'group': 0, 'points': content_points(runs, method, envs, floor),
                       'cell': grid[0, col].get_position(fig), 'aspect': (x1-x0, y1-y0, z1-z0)})
        cell = grid[0, col].get_position(fig)
        fig.text((cell.x0+cell.x1)/2, cell.y1+.012, NAME[method], ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    fig.suptitle(LABELS[robot], fontsize=11.5, y=.995)
    handles = legend_handles(combined)
    fig.legend(handles=handles, loc='lower center', ncol=(len(handles)+1)//2, fontsize=7.5, frameon=False,
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(.5, .0))
    zooms = {}
    fit_panels(fig, panels, zooms)
    refit(fig, panels, zooms)
    for attempt in range(10):
        problems, boxes = layout_check(fig, panels)
        if not problems:
            break
        zooms[0] *= .97
        place(fig, panels, zooms)
    zoom = round(zooms[0], 3)
    LAYOUT_AUDIT[f'robots/{robot}_{tag}'] = {'zoom': zoom, 'problems': problems, 'content_boxes_px': boxes}
    save(fig, f'robots/{robot}_{tag}')


def mesh_check(path):
    """Matplotlib meshes vs PyBullet render of the same arms at home, side by side."""
    import pybullet as p
    from env_eval.case_demon.gripper import GripperEnv
    from env_eval.case_demon.scenes import build
    azim, elev, distance = np.radians(-56), np.radians(26), 8.
    direction = np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)])
    fig = plt.figure(figsize=(12, 5))
    for col, (robot, sid) in enumerate(zip(ROBOTS, (1, 4, 7, 10, 13))):
        run = load(robot, 1, 'ours')
        polygons, _ = arm_polygons(run, 'home_frames')
        points = polygons.reshape(-1, 3)
        centre = (points.min(0)+points.max(0))/2
        span = (points.max(0)-points.min(0)).max()*.6
        ax = fig.add_subplot(2, 5, col+1, projection='3d', computed_zorder=False)
        draw_arm(ax, run, 'home_frames')
        for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(centre[i]-span, centre[i]+span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=26, azim=-56)
        ax.set_axis_off()
        ax.set_title(robot)
        env = GripperEnv(build(sid))
        view = p.computeViewMatrix((centre+direction*distance).tolist(), centre.tolist(), (0, 0, 1))
        # Distant camera with a narrow field: close to matplotlib's orthographic-like view.
        projection = p.computeProjectionMatrixFOV(float(np.degrees(2*np.arctan(span/distance))), 1., 1., 12.)
        _, _, rgba, _, _ = p.getCameraImage(500, 500, view, projection, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        env.close()
        ax = fig.add_subplot(2, 5, col+6)
        ax.imshow(np.asarray(rgba, np.uint8).reshape(500, 500, 4)[:, :, :3])
        ax.set_axis_off()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='only write the mesh-vs-PyBullet check image')
    args = parser.parse_args()
    if args.check:
        mesh_check(WORK/'mesh_check.png')
        return
    audits = {tag: figure(envs, tag) for envs, tag in (((1,), 'env1'), ((2,), 'env2'), ((1, 2), 'combined'))}
    (WORK/'pose_audit_input.json').write_text(json.dumps(audits))
    (OUT/'robots').mkdir(parents=True, exist_ok=True)
    for robot in ROBOTS:
        for envs, tag in (((1,), 'env1'), ((2,), 'env2'), ((1, 2), 'combined')):
            robot_figure(robot, envs, tag)
    (WORK/'LAYOUT_AUDIT.json').write_text(json.dumps(LAYOUT_AUDIT, indent=1))
    for name, entry in LAYOUT_AUDIT.items():
        print('layout', name, 'zoom', entry['zoom'], 'problems', entry['problems'] or 'none')
    for method, values in table().items():
        print(method, {k: [round(x, 3) for x in v] for k, v in values.items()})


if __name__ == '__main__':
    main()
