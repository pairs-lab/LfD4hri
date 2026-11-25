"""Ten new cluttered Oracle PUT demonstrations; clean wide view 6 only.

Independent addition: never changes the frozen VLA evaluation or old demos.
Stages: plan, qualify, record, extract, approve. All outputs stay in eval/.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np

from VLA.frame_export_r61 import export as media

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'eval/oracle_put_diverse10_view6_20261018'
PROFILE = ROOT / 'env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper/selected_profiles.json'
RESERVE = 30 * 1024**3
WIDTH, HEIGHT, FPS = 1920, 1080, 30


def space(out):
    if shutil.disk_usage(out).free < RESERVE:
        raise RuntimeError('Keep 30 GiB free for the existing VLA jobs; no files deleted.')


def bindings():
    paths = [Path(__file__).resolve(), PROFILE, Path(media.__file__).resolve()]
    paths += [ROOT / n for n in (
        'env_eval/oracle_feedback.py', 'env_eval/oracle_cinematic.py',
        'env_eval/oracle_demo.py', 'env_eval/feedback_environment.py',
        'env_eval/legacy_gripper_environment.py', 'env_eval/feedback_motion.py',
        'utils/Robotiq2F85.py', 'utils/camera.py')]
    return {str(p): media.sha(p) for p in paths}


def check_binding(out):
    frozen = media.read(out / 'SOURCE_BINDING.json')
    for name, digest in frozen.items():
        if media.sha(name) != digest:
            raise ValueError('Source changed since plan: ' + name)


def make_task(sid, candidate):
    from VLA.common.tasks import TaskSpec, SceneObjectSpec
    targets = ['red block', 'green block', 'orange block', 'purple block',
               'pink block', 'cyan block', 'brown block', 'gray block',
               'yellow lemon', 'red apple']
    goals = ['blue bowl', 'green bowl', 'orange bowl', 'yellow bowl',
             'purple bowl', 'pink bowl', 'cyan bowl', 'brown bowl', 'red bowl', 'gray bowl']
    starts = [(-.185, -.575), (-.065, -.625), (.045, -.535), (-.165, -.435),
              (-.04, -.465), (.035, -.64), (-.215, -.515), (-.095, -.55),
              (-.18, -.645), (-.035, -.59)]
    ends = [(.14, -.39), (.18, -.445), (-.16, -.69), (.135, -.59),
            (.19, -.66), (-.175, -.4), (.12, -.64), (.18, -.36),
            (.095, -.465), (.17, -.505)]
    i = sid - 1
    rng = np.random.default_rng(2026101800 + 100 * sid + candidate)
    pick = np.array(starts[i]) + (rng.uniform(-.012, .012, 2) if candidate else 0)
    place = np.array(ends[i]) + (rng.uniform(-.012, .012, 2) if candidate else 0)
    colors = ['blue', 'red', 'green', 'orange', 'yellow', 'purple', 'pink', 'cyan', 'brown', 'gray']
    blocks = [c + ' block' for c in colors if c + ' block' != targets[i]]
    rng.shuffle(blocks)
    fruits = [n for n in ['red apple', 'orange orange', 'yellow lemon', 'white egg', 'red strawberry']
              if n != targets[i]]
    rng.shuffle(fruits)
    # 10, 11, 12 physical objects; unchanged native assets, not a new table/robot.
    # Pack the second large bowl first, before the smaller distractors.
    names = [goals[(i + 4) % 10]] + fruits[:2] + blocks[:5 + i % 3]
    occupied = [(pick, .070), (place, .085)]
    extras = []
    for name in names:
        radius = .070 if name.endswith(' bowl') else .035
        for _ in range(20000):
            xy = rng.uniform((-.25 + radius, -.75 + radius), (.25 - radius, -.25 - radius))
            if all(np.linalg.norm(xy - pos) > radius + other + .008 for pos, other in occupied):
                break
        else:
            raise ValueError(f'Packing failed for scene {sid}, candidate {candidate}')
        occupied.append((xy, radius))
        extras.append(SceneObjectSpec(name, (*map(float, xy), .045), name.endswith(' bowl'),
                                      'new_diverse_put_distractor'))
    return TaskSpec(3, 'pick_place', f'Put the {targets[i]} into the {goals[i]}',
                    targets[i], (*map(float, pick), .045), goals[i], (*map(float, place), .045),
                    max_steps=10000, scene_profile='new_clutter_put10_view6', scene_objects=tuple(extras))


def plan(out):
    if (out / 'PLAN.json').exists():
        raise FileExistsError('Existing plan preserved; use qualify/record/extract.')
    out.mkdir(parents=True, exist_ok=True)
    space(out)
    rows = []
    for sid in range(1, 11):
        tasks = [make_task(sid, c).to_dict() for c in range(4)]
        for task in tasks:
            # The four previous PUT configs all started at (-.12,-.50).
            if np.linalg.norm(np.array(task['pick_position'][:2]) - [-.12, -.50]) < .04:
                raise ValueError('Too close to the old canonical pick layout')
        rows.append({'scene_id': sid, 'candidates': tasks})
    media.write(out / 'PLAN.json', {'created_UTC': media.utc(), 'rows': rows,
        'requested_episodes': 10, 'views': [6], 'insets': False,
        'robot': 'UR5e_Robotiq2F', 'controller': 'privileged feedback Oracle, NOT trained RL/VLA',
        'selection': 'first physically qualified candidate per scene; every attempt retained',
        'previous_layout_pick_xy': [-.12, -.50], 'different_target_goal_layouts': True,
        'physics_hz': 240, 'fps': FPS, 'resolution': [WIDTH, HEIGHT],
        'training_and_quantitative_jobs_untouched': True})
    media.write(out / 'SOURCE_BINDING.json', bindings())
    (out / 'PLAN.md').write_text(
        '# Ten new cluttered Oracle PUT demos\n\n'
        '1. Generate ten different tasks/layouts with 10–12 native objects each.\n'
        '2. Qualify contact-based grasp, release, physical support and safety without video; retain all attempts.\n'
        '3. Render one pilot and inspect the real view-6 images before the remaining nine.\n'
        '4. Render only clean cinematic view 6 at native 1920×1080/30 FPS, every eight physics ticks.\n'
        '5. Extract every decoded frame as native RGB PNG; verify complete decode, counts, CRC, SHA-256 and timing.\n'
        '6. Inspect contact sheets and native frames; publish gallery and truthful outcomes.\n\n'
        'No old media or VLA jobs are changed. Oracle assistance activates only after verified bilateral contact; '
        'these are illustrative model-based demonstrations, not learned-RL/VLA evidence.\n', encoding='utf-8')
    media.write(out / 'PROGRESS.json', {'stage': 'planned', 'scenes': 10, 'UTC': media.utc()})
    print('Plan ready:', out, flush=True)


def environment_imports():
    import pybullet as pb
    import imageio_ffmpeg
    from env_eval.oracle_feedback import FeedbackRollout, FeedbackEnvironment, MotionProfile
    from utils.camera import CINEMATIC_VIEWS
    from VLA.common.tasks import TaskSpec
    return pb, imageio_ffmpeg, FeedbackRollout, FeedbackEnvironment, MotionProfile, CINEMATIC_VIEWS, TaskSpec


def scene_for(sid, task):
    return SimpleNamespace(action='put', config=sid, place=task.place_name, task=lambda: task)


def actual_scene(env, pb):
    return {name: {'position': list(pb.getBasePositionAndOrientation(body)[0]),
                   'quaternion': list(pb.getBasePositionAndOrientation(body)[1]),
                   'AABB': [list(x) for x in pb.getAABB(body)], 'body_id': body}
            for name, body in env.object_ids.items()}


def qualify(out):
    check_binding(out)
    pb, _, Rollout, Environment, Profile, _, Task = environment_imports()
    profile = Profile(**media.read(PROFILE)['ur5e'])
    results = []
    with Environment('ur5e', out / '_derived_assets', False) as env:
        env.joint_jerk = profile.joint_jerk
        for row in media.read(out / 'PLAN.json')['rows']:
            sid = row['scene_id']
            selected = out / f'scene{sid:02d}/SCENE.json'
            if selected.exists():
                results.append(media.read(selected))
                continue
            for candidate, values in enumerate(row['candidates']):
                space(out)
                directory = out / '_qualification' / f'scene{sid:02d}_candidate{candidate}'
                if directory.exists():
                    result = media.read(directory / 'audit.json')
                else:
                    directory.mkdir(parents=True)
                    task = Task.from_dict(values)
                    env.reset(task)
                    initial = actual_scene(env, pb)
                    rollout = Rollout(env, scene_for(sid, task), directory, False, profile=profile)
                    try:
                        result = rollout.run()
                    except Exception as exc:
                        result = {'success': False, 'overall_pass': False, 'error': str(exc)}
                    finally:
                        rollout.close()
                    media.write(directory / 'initial_scene.json', initial)
                    media.write(directory / 'trajectory.json', rollout.samples)
                    media.write(directory / 'task.json', values)
                    media.write(directory / 'audit.json', result)
                print({'qualification': sid, 'candidate': candidate,
                       'success': result['success'], 'safe': result.get('safety_pass'),
                       'error': result.get('error')}, flush=True)
                if result.get('overall_pass'):
                    record = {'scene_id': sid, 'candidate': candidate, 'task': values,
                              'qualification': str(directory), 'object_count': 2 + len(values['scene_objects'])}
                    media.write(selected, record)
                    results.append(record)
                    break
            else:
                raise RuntimeError(f'Scene {sid}: no qualified candidate; attempts retained for diagnosis.')
            media.write(out / 'PROGRESS.json', {'stage': 'physical_qualification', 'qualified': len(results),
                                              'total': 10, 'UTC': media.utc()})
    media.write(out / 'QUALIFICATION.json', {'qualified_scenes': len(results), 'scenes': results})


def record(out, ids):
    check_binding(out)
    pb, imageio_ffmpeg, Base, Environment, Profile, views, Task = environment_imports()
    cfg = views[5]
    profile = Profile(**media.read(PROFILE)['ur5e'])

    class WideOnly(Base):
        def __init__(self, env, scene, directory):
            self.wide_writer = None
            super().__init__(env, scene, directory, False, WIDTH, HEIGHT, FPS, profile=profile)
            self.wide_writer = imageio_ffmpeg.write_frames(str(directory / 'view_6.mp4'),
                (WIDTH, HEIGHT), fps=FPS, codec='libx264', quality=8, macro_block_size=1,
                ffmpeg_log_level='error', output_params=['-preset', 'fast', '-tune', 'zerolatency',
                    '-rc-lookahead', '0', '-bf', '0', '-refs', '1', '-threads', '2',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
            self.wide_writer.send(None)

        def capture(self):
            # Parent captures only telemetry: self.writers remains empty, so
            # no macro/hand-eye/inset or any other wide camera is ever rendered.
            super().capture()
            if self.initializing or self.wide_writer is None:
                return
            rgb, segmentation = self.frame(cfg, WIDTH, HEIGHT)
            self.wide_writer.send(rgb)
            ids = segmentation & ((1 << 24) - 1)
            row = {'frame': len(self.visibility) + 1, 'physics_tick': self.tick_count,
                   'simulation_time_s': self.tick_count / self.env.physics_hz,
                   'encoded_PTS_s': len(self.visibility) / FPS, 'phase': self.phase,
                   'target_pixels': int(np.count_nonzero(ids == self.env.pick_id)),
                   'goal_pixels': int(np.count_nonzero(ids == self.env.place_id)),
                   'per_object_pixels': {name: int(np.count_nonzero(ids == body))
                                         for name, body in self.env.object_ids.items()}}
            self.visibility.append(row)

        def close(self):
            if self.wide_writer is not None:
                self.wide_writer.close()
                self.wide_writer = None
            super().close()

    with Environment('ur5e', out / '_derived_assets', True) as env:
        if env.demo_renderer != pb.ER_BULLET_HARDWARE_OPENGL:
            raise RuntimeError('EGL unavailable: do not silently substitute a different renderer.')
        env.joint_jerk = profile.joint_jerk
        for sid in ids:
            space(out)
            directory = out / f'scene{sid:02d}'
            if (directory / 'audit.json').exists():
                print('Existing completed scene preserved:', sid, flush=True)
                continue
            if (directory / 'view_6.mp4').exists():
                raise FileExistsError('Existing unsealed video preserved: ' + str(directory))
            selection = media.read(directory / 'SCENE.json')
            task = Task.from_dict(selection['task'])
            env.reset(task)
            initial = actual_scene(env, pb)
            baseline = media.read(Path(selection['qualification']) / 'initial_scene.json')
            max_initial_error = max(float(np.max(np.abs(np.array(initial[k]['position']) - baseline[k]['position'])))
                                    for k in initial)
            if max_initial_error > 1e-6:
                raise ValueError(f'Render vs qualification initial-state mismatch: {max_initial_error}')
            rollout = WideOnly(env, scene_for(sid, task), directory)
            began = time.monotonic()
            try:
                result = rollout.run()
            finally:
                rollout.close()
                media.write(directory / 'trajectory.json', rollout.samples)
                media.write(directory / 'frames.json', rollout.visibility)
                media.write(directory / 'initial_scene.json', initial)
            result.update(videos=['view_6.mp4'], camera_views=[dict(cfg)],
                renderer='PyBullet EGL hardware OpenGL', no_insets=True, no_video_text=True,
                frame_count=len(rollout.visibility), wall_seconds=time.monotonic() - began,
                qualification_initial_state_max_difference_m=max_initial_error,
                object_count=len(env.object_ids), scene_id=sid)
            result['media_sha256'] = media.sha(directory / 'view_6.mp4')
            media.write(directory / 'audit.json', result)
            media.write(out / 'PROGRESS.json', {'stage': 'recording', 'scene_id': sid,
                'recorded': len(list(out.glob('scene*/audit.json'))), 'total': 10, 'UTC': media.utc()})
            print({'recorded': sid, 'success': result['success'], 'safe': result['safety_pass'],
                   'frames': result['frame_count'], 'objects': len(env.object_ids),
                   'wall_seconds': round(result['wall_seconds'], 1)}, flush=True)
            if not result['overall_pass']:
                raise RuntimeError('Recorded outcome failed physical/safety gates; kept for review.')


def extract(out, ids):
    check_binding(out)
    cv2.setNumThreads(1)
    for sid in ids:
        space(out)
        directory = out / f'scene{sid:02d}'
        video = directory / 'view_6.mp4'
        audit = media.read(directory / 'audit.json')
        if media.sha(video) != audit['media_sha256']:
            raise ValueError('Recorded video changed')
        dest = directory / 'view6'
        if (directory / 'FRAME_AUDIT.json').exists():
            continue
        if dest.exists():
            raise FileExistsError('Existing partial frames preserved')
        stage = directory / '.view6.extracting'
        stage.mkdir(exist_ok=False)
        info = media.metadata(video)
        n = audit['frame_count']
        if info != {'frames': n, 'width': WIDTH, 'height': HEIGHT, 'fps': float(FPS)}:
            raise ValueError('Unexpected resolution, FPS or frame count')
        trace = media.read(directory / 'frames.json')
        np.testing.assert_allclose([t['simulation_time_s'] for t in trace], np.arange(n) / FPS, atol=1e-9)
        if [t['physics_tick'] for t in trace] != list(range(0, n * 8, 8)):
            raise ValueError('Camera cadence not eight physics ticks')
        with (directory / 'extract.log').open('wb') as log:
            process = subprocess.run(media.png_encode_command(video, stage), stdout=log, stderr=log,
                                     env=media.THREAD_ENV)
        if process.returncode:
            raise RuntimeError('PNG extraction failed; partial outputs retained')
        files = sorted(stage.glob('*.png'))
        if [p.name for p in files] != [f'frame{i:06d}.png' for i in range(1, n + 1)]:
            raise ValueError('Missing/extra/misnamed frame')
        hashes, sizes = [], []
        for p in files:
            digest, size = media.check_png(p, WIDTH, HEIGHT)
            image = cv2.imread(str(p))
            if image is None or image.shape != (HEIGHT, WIDTH, 3):
                raise ValueError('PNG does not fully decode')
            hashes.append((digest, p.name))
            sizes.append(size)
        independent_n = media.independent_count(video)
        if independent_n != n:
            raise ValueError('Independent source full-decode frame mismatch')
        indices = sorted({0, n // 2, n - 1})
        expected = media.raw_samples(video, indices, WIDTH, HEIGHT)
        for i, rgb in zip(indices, expected):
            actual = cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB)
            if not np.array_equal(actual, rgb):
                raise ValueError('PNG pixels differ from native video decode')
        (directory / 'frames.sha256').write_text(''.join(f'{h}  {name}\n' for h, name in hashes))
        with (directory / 'timestamps.tsv').open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['file', 'frame_1_based', 'video_index_0_based', 'video_PTS_s',
                             'simulation_time_s', 'physics_tick', 'phase'])
            for i, t in enumerate(trace):
                writer.writerow([files[i].name, i + 1, i, i / FPS, t['simulation_time_s'], t['physics_tick'], t['phase']])
        media.publish_dir(stage, dest)
        media.write(directory / 'FRAME_AUDIT.json', {'passed': True, **info, 'PNG_bytes': sum(sizes),
            'full_video_decode_frame_count': independent_n, 'all_PNG_CRC_SHA256_full_decode_verified': True,
            'sample_RGB_bit_exact': True, 'sample_indices_0_based': indices,
            'unchanged_native_resolution_all_frames_no_crop_no_resize_no_subsampling': True,
            'video_sha256': media.sha(video), 'UTC': media.utc()})
        print({'extracted': sid, 'frames': n, 'PNG_MiB': round(sum(sizes) / 1024**2, 1)}, flush=True)
    gallery(out)


def gallery(out):
    panels, rows = [], []
    for directory in sorted(out.glob('scene[0-9][0-9]')):
        if not (directory / 'FRAME_AUDIT.json').exists():
            continue
        task = media.read(directory / 'SCENE.json')['task']
        trace = media.read(directory / 'frames.json')
        audit = media.read(directory / 'audit.json')
        # Five actually encoded frames: reset, near pickup, carry, release, final.
        phases = ['initial', 'approach', 'carry', 'release', 'settle']
        indices = []
        for phase in phases:
            matches = [i for i, t in enumerate(trace) if t['phase'] == phase]
            indices.append(matches[-1] if phase == 'settle' else
                           matches[len(matches) // 2] if matches else len(trace) - 1)
        images = []
        for phase, i in zip(phases, indices):
            image = cv2.imread(str(directory / 'view6' / f'frame{i + 1:06d}.png'))
            thumb = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            canvas = np.full((300, 480, 3), 245, np.uint8)
            canvas[30:] = thumb
            cv2.putText(canvas, f'{directory.name} {phase} frame {i + 1}', (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (10, 10, 10), 1, cv2.LINE_AA)
            images.append(canvas)
        panel = np.concatenate(images, axis=1)
        cv2.imwrite(str(directory / 'CONTACT_SHEET.jpg'), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
        media.write(directory / 'VISUAL_SAMPLES.json', {'indices_0_based': indices, 'phases': phases,
                                                      'source': 'actual decoded native PNGs'})
        panels.append(panel)
        rows.append(f'<section><h2>{directory.name}: {html.escape(task["prompt"])}</h2>'
            f'<p>{audit["object_count"]} objects; physical PUT success: {audit["success"]}.</p>'
            f'<video controls preload="metadata" width="960" src="{directory.name}/view_6.mp4"></video>'
            f'<p><a href="{directory.name}/view6/">All PNG frames</a> | '
            f'<a href="{directory.name}/timestamps.tsv">Timestamps</a></p>'
            f'<img width="1200" src="{directory.name}/CONTACT_SHEET.jpg"></section>')
    if panels:
        for start in range(0, len(panels), 5):
            cv2.imwrite(str(out / f'CONTACT_SHEET_{start // 5 + 1}.jpg'),
                        np.concatenate(panels[start:start + 5], axis=0), [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>New Oracle PUT view6</title>'
        '<style>body{font-family:sans-serif;max-width:1260px;margin:auto}section{margin:35px 0}img{max-width:100%}</style>'
        '<h1>Ten new cluttered Oracle PUT scenes — clean view 6</h1>'
        '<p>Privileged feedback Oracle, not a learned RL/VLA policy. Native 1920×1080, 30 FPS. '
        'No inset, hand-eye, annotations or speed changes. Frame1 = time0.</p>' + ''.join(rows), encoding='utf-8')
    audits = [media.read(p) for p in sorted(out.glob('scene*/FRAME_AUDIT.json'))]
    media.write(out / 'FINAL_AUDIT.json', {'numeric_pass': len(audits) == 10 and all(a['passed'] for a in audits),
        'videos': len(audits), 'PNG_frames': sum(a['frames'] for a in audits),
        'PNG_GiB': sum(a['PNG_bytes'] for a in audits) / 1024**3,
        'only_view_6': [str(p.relative_to(out)) for p in out.glob('scene*/*.mp4')] ==
                       [f'scene{i:02d}/view_6.mp4' for i in range(1, 11)],
        'visual_review_pending': True, 'UTC': media.utc()})


def approve(out, note):
    check_binding(out)
    audit = media.read(out / 'FINAL_AUDIT.json')
    videos = sorted(out.glob('scene*/*.mp4'))
    if not audit['numeric_pass'] or len(videos) != 10 or {p.name for p in videos} != {'view_6.mp4'}:
        raise ValueError('Ten native view6 videos and all frame audits required')
    results = [media.read(out / f'scene{i:02d}/audit.json') for i in range(1, 11)]
    if not all(a['overall_pass'] for a in results):
        raise ValueError('Physical/safety outcomes need review')
    audit.update(only_view_6=True, visual_review_pending=False, visual_review_note=note,
                 physical_successes=sum(a['success'] for a in results),
                 scene_object_counts=[a['object_count'] for a in results], approved_UTC=media.utc())
    media.write(out / 'FINAL_AUDIT.json', audit)
    (out / 'README.md').write_text(
        '# Ten new diverse Oracle PUT demonstrations — view 6 only\n\n'
        f'10 videos; {audit["PNG_frames"]:,} full PNG frames; 1920×1080, 30 FPS. '
        'Each scene contains 10–12 objects, a new target/goal layout and distractors.\n\n'
        'Open `index.html`. Each `scene01`–`scene10` has `view_6.mp4` and '
        '`view6/frame000001.png`, `frame000002.png`, … . No front-gripper or hand-eye stream, '
        'no inset, text, resizing, cropping or frame skipping. Frame 1 is video index 0/time 0.\n\n'
        'These are privileged state-feedback Oracle demonstrations (IK and verified-contact grasp assistance), '
        'not results of learned RL or a VLA policy. Physical support, detached gripper and settling are verified '
        'after release; failed qualification candidates, if any, are retained in `_qualification/`. '
        'The ten selected successful demonstrations are illustrative, not an unbiased success-rate experiment.\n\n'
        f'Visual review: {note}\n\n'
        'Original media and live VLA jobs were not modified. Per-scene scene/task/state/trajectory, '
        'physics timestamps, visibility telemetry and SHA-256 manifests are included.\n', encoding='utf-8')
    media.write(out / 'PROGRESS.json', {'stage': 'complete', 'videos': 10,
                                      'PNG_frames': audit['PNG_frames'], 'UTC': media.utc()})
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256'
                   and not p.name.endswith('.log') and p.name != 'run.log')
    with (out / 'MANIFEST.sha256').open('w') as f:
        for p in paths:
            f.write(media.sha(p) + '  ' + str(p.relative_to(out)) + '\n')
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'qualify', 'record', 'extract', 'approve'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--scenes', default='1,2,3,4,5,6,7,8,9,10')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    os.nice(8)
    if args.stage == 'plan':
        plan(args.output)
    elif args.stage == 'qualify':
        qualify(args.output)
    elif args.stage == 'record':
        record(args.output, [int(v) for v in args.scenes.split(',')])
    elif args.stage == 'extract':
        extract(args.output, [int(v) for v in args.scenes.split(',')])
    else:
        if not args.note:
            parser.error('approve requires actual visual-review --note')
        approve(args.output, args.note)


if __name__ == '__main__':
    main()
