"""Figure: end-effector paths, IK (top row) vs Ours (bottom row), five arms.

Also writes a smoothness table (LDLJ, SPARC, stops, path length) as CSV/LaTeX.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter

from env_eval.case_demon.robots import ROBOTS
from env_eval.case_demon.run import WORK

DATA = WORK/'compare'
OUT = Path('/home/uet/Downloads/trajectory_compare')
LABELS = {'Panda': 'Franka Panda', 'KUKA': 'KUKA LBR iiwa', 'Kinova': 'Kinova Gen3', 'UR5e': 'UR5e', 'UF850': 'UFACTORY 850'}
# Okabe-Ito: distinguishable for colour-blind readers and in greyscale.
COLOR = {'ik': '#D55E00', 'ours': '#0072B2'}
NAME = {'ik': 'IK (straight line)', 'ours': 'Ours'}
PICK, PLACE = '#E69F00', '#009E73'


def load(robot, env, method):
    d = np.load(DATA/f'{robot}_env{env}_{method}.npz')
    return {'tip': d['tip'], 'q': d['q'], 'skeleton': d['skeleton'], 'hz': float(d['hz']),
            'events': json.loads(str(d['events'])), 'scene': json.loads(str(d['scene'])),
            'result': json.loads(str(d['result']))}


# ---------- smoothness metrics (tool tip, measured at 240 Hz) ----------

def kinematics(tip, hz, window=25):
    dt = 1/hz
    velocity = savgol_filter(tip, window, 3, deriv=1, delta=dt, axis=0)
    acceleration = savgol_filter(tip, window, 3, deriv=2, delta=dt, axis=0)
    jerk = savgol_filter(tip, window, 3, deriv=3, delta=dt, axis=0)
    return velocity, acceleration, jerk


def ldlj(velocity, jerk, hz):
    """Log dimensionless jerk (velocity-normalised). Higher (less negative) is smoother."""
    speed = np.linalg.norm(velocity, axis=1)
    duration = len(speed)/hz
    integral = np.sum(np.sum(jerk**2, axis=1))/hz
    return float(-np.log(duration**3/speed.max()**2*integral))


def sparc(speed, hz, padlevel=4, cutoff=10., threshold=.05):
    """Spectral arc length (Balasubramanian et al. 2015). Higher (closer to 0) is smoother."""
    n = int(2**(np.ceil(np.log2(len(speed)))+padlevel))
    frequency = np.arange(n)*hz/n
    magnitude = np.abs(np.fft.fft(speed, n))
    magnitude = magnitude/magnitude.max()
    inside = frequency <= cutoff
    f, m = frequency[inside], magnitude[inside]
    keep = np.nonzero(m >= threshold)[0]
    f, m = f[:keep[-1]+1], m[:keep[-1]+1]
    df, dm = np.diff(f)/f[-1], np.diff(m)
    return float(-np.sum(np.sqrt(df**2+dm**2)))


def stops(speed, hz, moving=.01, still=.004, minimum=.06):
    """Times the tool comes to rest mid-episode (excluding the final settle)."""
    count, state, rest = 0, 'still', 0
    for v in speed[:-int(.8*hz)]:
        if state == 'still' and v > moving:
            state, rest = 'moving', 0
        elif state == 'moving' and v < still:
            rest += 1
            if rest >= minimum*hz:
                count, state = count+1, 'still'
        elif v >= still:
            rest = 0
    return count


def metrics(run):
    tip, hz = run['tip'], run['hz']
    velocity, _, jerk = kinematics(tip, hz)
    speed = np.linalg.norm(velocity, axis=1)
    q = run['q']
    joint_jerk = savgol_filter(q, 25, 3, deriv=3, delta=1/hz, axis=0)
    return {'success': bool(run['result']['overall_pass']), 'duration_s': len(tip)/hz,
            'path_m': float(np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1))),
            'stops': stops(speed, hz), 'LDLJ': ldlj(velocity, jerk, hz), 'SPARC': sparc(speed, hz),
            'peak_joint_jerk': float(np.abs(joint_jerk).max())}


def table():
    rows = []
    for robot in ROBOTS:
        for env in (1, 2, 3):
            for method in ('ik', 'ours'):
                rows.append({'robot': robot, 'env': env, 'method': method, **metrics(load(robot, env, method))})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT/'smoothness_per_episode.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = [('success', 'Success', '{:.0f}/15'), ('duration_s', 'Duration (s)', '{:.2f}'),
            ('path_m', 'Path (m)', '{:.2f}'), ('stops', 'Stops', '{:.1f}'),
            ('LDLJ', r'LDLJ $\uparrow$', '{:.2f}'), ('SPARC', r'SPARC $\uparrow$', '{:.2f}'),
            ('peak_joint_jerk', r'Peak joint jerk (rad/s$^3$) $\downarrow$', '{:.0f}')]
    lines = [r'\begin{tabular}{l' + 'c'*len(keys) + '}', r'\toprule',
             'Method & ' + ' & '.join(k[1] for k in keys) + r' \\', r'\midrule']
    summary = {}
    for method in ('ik', 'ours'):
        sel = [r for r in rows if r['method'] == method]
        cells, summary[method] = [], {}
        for key, _, fmt in keys:
            values = np.array([r[key] for r in sel], float)
            if key == 'success':
                cells.append(fmt.format(values.sum()))
            else:
                cells.append(fmt.format(values.mean())+r' $\pm$ '+fmt.format(values.std()))
            summary[method][key] = [float(values.mean()), float(values.std())]
        lines.append(NAME[method] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT/'smoothness_table.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'smoothness_summary.json').write_text(json.dumps(summary, indent=1))
    return rows, summary


# ---------- figure ----------

def draw(ax, run, method, limits):
    tip, skeleton, events, scene = run['tip'], run['skeleton'], run['events'], run['scene']
    (x0, x1), (y0, y1), (z0, z1) = limits
    table = [[x0, y0, .004], [x1, y0, .004], [x1, y1, .004], [x0, y1, .004]]
    ax.add_collection3d(Poly3DCollection([table], facecolor='#9aa3ab', alpha=.16, edgecolor='none'))
    low, high = map(np.array, scene['bowl_aabb'])
    center, radius = (low[:2]+high[:2])/2, (high[0]-low[0])/2
    angle = np.linspace(0, 2*np.pi, 60)
    ax.plot(center[0]+radius*np.cos(angle), center[1]+radius*np.sin(angle), .006, color=PLACE, lw=.8, alpha=.7)
    pick = np.array(scene['pick_xy'])
    half = max(scene['pick_extent'][:2])/2
    square = pick+half*np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(square[:, 0], square[:, 1], .006, color=PICK, lw=.8, alpha=.8)
    if 'pick' in events:
        bones = skeleton[events['pick'][0]]
        ax.plot(*bones.T, color='#5f6770', lw=2.4, alpha=.35, solid_capstyle='round', zorder=1)
        ax.scatter(*bones[1:-1].T, s=5, color='#5f6770', alpha=.45, depthshade=False)
    ax.plot(*tip[::2].T, color=COLOR[method], lw=1.5, zorder=5)
    ax.scatter(*tip[0], s=16, color='k', depthshade=False, zorder=6)
    if 'pick' in events:
        ax.scatter(*events['pick'][1], s=95, marker='*', color=PICK, edgecolor='k', linewidth=.5, depthshade=False, zorder=7)
    if 'place' in events:
        ax.scatter(*events['place'][1], s=42, marker='v', color=PLACE, edgecolor='k', linewidth=.5, depthshade=False, zorder=7)
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1-x0, y1-y0, z1-z0))
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor('#d0d4d8')
        axis._axinfo['grid'].update(color='#e3e6e9', linewidth=.4)
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.tick_params(labelsize=5, pad=-3)
    ax.set_xlabel('x (m)', fontsize=5.5, labelpad=-9)
    ax.set_ylabel('y (m)', fontsize=5.5, labelpad=-9)
    ax.set_zlabel('z (m)', fontsize=5.5, labelpad=-10)


def column_limits(runs):
    points = np.vstack([np.vstack([r['tip'], r['skeleton'][r['events'].get('pick', [0])[0]]]) for r in runs])
    low, high = points.min(0), points.max(0)
    low[2], pad = 0., .03
    return [(low[i]-pad, high[i]+pad) for i in range(3)]


def figure(env, tag):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(7.16, 3.3))
    grid = fig.add_gridspec(2, len(ROBOTS), left=.045, right=.995, top=.93, bottom=.1, wspace=.02, hspace=.0)
    for col, robot in enumerate(ROBOTS):
        runs = {m: load(robot, env, m) for m in ('ik', 'ours')}
        limits = column_limits(list(runs.values()))
        for row, method in enumerate(('ik', 'ours')):
            ax = fig.add_subplot(grid[row, col], projection='3d', computed_zorder=False)
            draw(ax, runs[method], method, limits)
            if not runs[method]['result']['overall_pass']:
                ax.text2D(.5, .9, 'task failed', transform=ax.transAxes, ha='center', fontsize=6, color='#b00')
            if row == 0:
                ax.set_title(LABELS[robot], fontsize=7.5, pad=-2)
    for row, method in enumerate(('ik', 'ours')):
        y = .93-(row+.5)*(.83/2)
        fig.text(.012, y, NAME[method] if method == 'ours' else 'IK', rotation=90, va='center', ha='center',
                 fontsize=8, fontweight='bold', color=COLOR[method])
    handles = [Line2D([], [], color=COLOR['ik'], lw=1.6, label='IK: straight lines, stop at waypoints'),
               Line2D([], [], color=COLOR['ours'], lw=1.6, label='Ours'),
               Line2D([], [], marker='o', ls='', color='k', ms=4, label='start'),
               Line2D([], [], marker='*', ls='', mfc=PICK, mec='k', mew=.5, ms=9, label='pick (suction contact)'),
               Line2D([], [], marker='v', ls='', mfc=PLACE, mec='k', mew=.5, ms=6, label='place (release)'),
               Line2D([], [], color='#5f6770', lw=2.4, alpha=.35, label='arm at pick')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=6, frameon=False, handlelength=1.6,
               columnspacing=1.1, bbox_to_anchor=(.52, .0))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f'trajectory_ik_vs_ours_{tag}.pdf')
    fig.savefig(OUT/f'trajectory_ik_vs_ours_{tag}.png', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    figure(1, 'main_env1')
    figure(2, 'appendix_env2')
    figure(3, 'appendix_env3')
    rows, summary = table()
    for method, values in summary.items():
        print(method, {k: [round(x, 3) for x in v] for k, v in values.items()})


if __name__ == '__main__':
    main()
