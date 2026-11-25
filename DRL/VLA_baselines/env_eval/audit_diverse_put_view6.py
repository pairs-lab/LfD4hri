"""Independent cross-check and optional download ZIP for the ten new PUT demos."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from VLA.frame_export_r61 import export as media

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'eval/oracle_put_diverse10_view6_20261018'


def verify(out):
    from utils.camera import CINEMATIC_VIEWS
    from env_eval.oracle_demo import scenes
    expected_camera = json.loads(json.dumps(CINEMATIC_VIEWS[5]))
    older = json.loads(json.dumps([s.task().to_dict() for s in scenes() if s.action == 'put']))
    rows = []
    for sid in range(1, 11):
        directory = out / f'scene{sid:02d}'
        selection = media.read(directory / 'SCENE.json')
        task = selection['task']
        audit = media.read(directory / 'audit.json')
        frames = media.read(directory / 'FRAME_AUDIT.json')
        trajectory = media.read(directory / 'trajectory.json')
        qualification = media.read(Path(selection['qualification']) / 'trajectory.json')
        times = media.read(directory / 'frames.json')
        if len(trajectory) != len(qualification) or len(times) != len(trajectory):
            raise ValueError('State/recording/qualification length differs')
        differences = {k: float(np.max(np.abs(np.array([r[k] for r in trajectory]) -
                                            np.array([r[k] for r in qualification]))))
                       for k in ('joints', 'object', 'tcp', 'gripper_width')}
        if max(differences.values()) > 1e-10:
            raise ValueError('Recorder changed the physical trajectory')
        if audit['camera_views'] != [expected_camera] or audit['videos'] != ['view_6.mp4'] or not audit['no_insets']:
            raise ValueError('Not the original clean cinematic view6')
        if not audit['overall_pass'] or audit['renderer'] != 'PyBullet EGL hardware OpenGL':
            raise ValueError('Physical/safety/renderer gate failed')
        count = len(trajectory)
        if media.independent_count(directory / 'view_6.mp4') != count:
            raise ValueError('Independent video decode length differs')
        paths = sorted((directory / 'view6').glob('*.png'))
        checksums = (directory / 'frames.sha256').read_text().splitlines()
        if len(paths) != count or len(checksums) != count or frames['frames'] != count:
            raise ValueError('Missing/excess PNG frames')
        for i, (p, line) in enumerate(zip(paths, checksums), 1):
            digest, name = line.split('  ', 1)
            if p.name != name or name != f'frame{i:06d}.png' or media.check_png(p, 1920, 1080)[0] != digest:
                raise ValueError('Independent PNG sequence, CRC or checksum failure')
        with (directory / 'timestamps.tsv').open() as f:
            timestamps = list(csv.DictReader(f, delimiter='\t'))
        if len(timestamps) != count or [int(t['physics_tick']) for t in timestamps] != list(range(0, count * 8, 8)):
            raise ValueError('Missing timestamp or incorrect physical cadence')
        if any(t['pick_name'] == task['pick_name'] and t['place_name'] == task['place_name'] and
               t['pick_position'] == task['pick_position'] and t['place_position'] == task['place_position']
               for t in older):
            raise ValueError('Duplicate old PUT configuration')
        all_names = [task['pick_name'], task['place_name']] + [t['name'] for t in task['scene_objects']]
        if len(set(all_names)) != len(all_names) or len(all_names) not in (10, 11, 12):
            raise ValueError('Incorrect clutter count or duplicate object identity')
        rows.append({'scene_id': sid, 'frames': count, 'objects': len(all_names),
                     'prompt': task['prompt'], 'success': audit['success'], 'safety_pass': audit['safety_pass'],
                     'recorder_vs_no_video_state_max_differences': differences,
                     'old_four_put_configs_duplicate': False,
                     'start_target_visible_pixels': times[0]['target_pixels'],
                     'start_goal_visible_pixels': times[0]['goal_pixels'],
                     'target_visible_pixels_min': min(t['target_pixels'] for t in times),
                     'goal_visible_pixels_min': min(t['goal_pixels'] for t in times),
                     'video_sha256': media.sha(directory / 'view_6.mp4')})
        print({'independently_verified_scene': sid, 'frames': count}, flush=True)
    if len(list(out.glob('scene*/*.mp4'))) != 10:
        raise ValueError('Unexpected additional video streams')
    media.write(out / 'INDEPENDENT_AUDIT.json', {'passed': True, 'videos': 10,
        'frames': sum(r['frames'] for r in rows), 'views': [6], 'insets': False,
        'all_10_measured_trajectories_unchanged_by_recorder': True,
        'all_PNG_sequence_CRC_SHA256_reverified': True, 'all_videos_fully_decoded_again': True,
        'source_code_sha256': media.sha(Path(__file__)), 'records': rows, 'UTC': media.utc()})


def package(out):
    final = media.read(out / 'FINAL_AUDIT.json')
    independent = media.read(out / 'INDEPENDENT_AUDIT.json')
    if not final['numeric_pass'] or final['visual_review_pending'] or not independent['passed']:
        raise ValueError('Numerical and actual visual review must finish first')
    manifest = (out / 'MANIFEST.sha256').read_text().splitlines()
    expected = dict(line.split('  ', 1)[::-1] for line in manifest)
    expected['MANIFEST.sha256'] = media.sha(out / 'MANIFEST.sha256')
    needed = sum((out / name).stat().st_size for name in expected)
    if shutil.disk_usage(out).free - needed < 30 * 1024**3:
        raise RuntimeError('ZIP would violate 30 GiB reserve')
    archive = out.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError('Existing ZIP preserved')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, digest in expected.items():
            source = out / name
            if media.sha(source) != digest:
                raise ValueError('Source changed after final seal: ' + name)
            z.write(source, str(Path(out.name) / name))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
        import hashlib
        for name, digest in expected.items():
            with z.open(str(Path(out.name) / name)) as f:
                h = hashlib.sha256()
                for data in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(data)
            if h.hexdigest() != digest:
                raise ValueError('Archived file SHA256 differs: ' + name)
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': media.sha(archive), 'all_member_CRC_and_SHA256_verified': True,
               'files': len(expected), 'PNG_frames': final['PNG_frames'], 'videos': 10, 'UTC': media.utc()}
    media.write(out.with_suffix('.archive_audit.json'), receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify', 'package'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    verify(args.output) if args.stage == 'verify' else package(args.output)
