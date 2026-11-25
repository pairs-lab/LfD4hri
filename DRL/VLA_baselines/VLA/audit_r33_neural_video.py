"""Decode actual post-action-only R33/Smol67 videos, without rerender/reroll.

Older video auditors assume steps+1 or an intro/terminal hold. Those mappings
do NOT apply to these evaluators: exactly one post-action frame per command.
Select a fixed seed, all four tasks including failures, never best outcomes.
"""
import argparse
import json
import math
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

TASKS = ('touch', 'pick', 'move', 'pick_place')


def transport_metadata(metadata, decoded_count):
    # imageio reports nframes=inf for an otherwise ordinary MP4 stream.
    # Keep standards-compliant JSON and the independently decoded count;
    # never turn that unknown estimate into a corrupt-video diagnosis.
    result = {key: metadata.get(key) for key in ('fps', 'duration', 'codec', 'pix_fmt', 'source_size', 'size')}
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite essential video metadata')
    result['exact_decoded_frame_count'] = decoded_count
    result['imageio_estimated_nframes_unknown'] = not math.isfinite(metadata.get('nframes', math.inf))
    return result


def frame_mapping(rows, traces):
    if tuple(r['task'] for r in rows) != TASKS or len({r['seed'] for r in rows}) != 1:
        raise ValueError('All four ordered terminal tasks of one fixed seed required')
    offset, result = 0, []
    for row in rows:
        task, count = row['task'], int(row['steps'])
        if count < 1 or len(traces[task]) != count:
            raise ValueError('Actual complete action trace/video timeline required')
        if bool(traces[task][-1]['success']) != bool(row['success']):
            raise ValueError('Progress/terminal trace outcome mismatch')
        result.append({'task': task, 'steps': count, 'success': bool(row['success']),
            'offset': offset, 'samples': [offset, offset+count//2, offset+count-1],
            'phases': ['after_first_command_NOT_reset', 'middle', 'after_terminal_command']})
        offset += count
    return offset, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, action='append', required=True)
    parser.add_argument('--seed', type=int, default=20268001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for number, progress in enumerate(args.progress):
        state = json.loads(progress.read_text())
        rows = [r for r in state['rows'] if r['seed'] == args.seed]
        paths = {r['trajectory'] for r in rows}
        if len(paths) != 1:
            raise ValueError('One complete retained source trajectory required')
        trajectory = Path(next(iter(paths)))
        video = trajectory.with_name('rollout.mp4')
        protected = {str(p.resolve()): sha256_file(p) for p in (trajectory, video)}
        traces = json.loads(trajectory.read_text())
        expected, mapping = frame_mapping(rows, traces)
        folder = args.output/str(number)
        folder.mkdir()
        with imageio.get_reader(video) as reader:
            metadata = reader.get_meta_data()
            if metadata['fps'] != 10 or reader.count_frames() != expected:
                raise ValueError('Wrong FPS/frame count for post-action-only recorder')
            # Every task/phase is taken from the ACTUAL compressed MP4,
            # including terminal failure/safety frames. No simulator runs.
            sheet = Image.new('RGB', (768*3, 280*4), 'black')
            for r, task in enumerate(mapping):
                for c, (phase, index) in enumerate(zip(task['phases'], task['samples'], strict=True)):
                    frame = Image.fromarray(reader.get_data(index))
                    frame.save(folder/f"{task['task']}_{phase}.png")
                    sheet.paste(frame.resize((768, 256)), (768*c, 280*r+24))
                    ImageDraw.Draw(sheet).text((768*c+4, 280*r+4),
                        f"{task['task']} {phase} | success={task['success']} | frame={index}", fill='white')
            sheet.save(folder/'actual_video_contact_sheet.png')
        if any(sha256_file(Path(p)) != h for p, h in protected.items()):
            raise ValueError('Completed seed video/trajectory changed during decode')
        reports.append({'progress': str(progress.resolve()), 'model': state['model'],
            'checkpoint': state['checkpoint'], 'seed': args.seed, 'source_terminal_rows_snapshot': rows,
            'video': str(video.resolve()), 'video_metadata': transport_metadata(metadata, expected),
            'decoded_frames': expected, 'task_mapping': mapping, 'bound_files': protected,
            'all_four_tasks_including_failures_included': True})
    atomic_json(args.output/'audit.json', {'passed': True,
        'scope': 'actual_compressed_neural_rollout_video_timeline_NOT_new_trials_or_SR',
        'reports': reports, 'physics_rerendered': False, 'neural_rollouts_rerolled': False,
        'model_input_images_changed': False, 'manual_visual_quality_certified': False,
        'implementation_sha256': sha256_file(Path(__file__))})


if __name__ == '__main__': main()
