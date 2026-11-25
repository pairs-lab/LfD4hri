"""Independent full-stream count/codec audit, gallery and immutable video hashes."""
import argparse
import html
import json
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def inspect(directory):
    audit = json.loads((directory/'camera_audit.json').read_text())
    rows = []
    for old in audit['streams']:
        path = directory/old['file']
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-threads', '1',
            '-i', str(path), '-map', '0:v:0', '-f', 'null', '-', '-nostats', '-progress', 'pipe:1'],
            capture_output=True, text=True)
        source = result.stderr.split('Stream mapping:')[0]
        counts = re.findall(r'^frame=(\d+)$', result.stdout, re.M)
        count = int(counts[-1]) if counts else -1
        row = {'file': old['file'], 'actual_full_decode_frame_count': count,
            'H264': bool(re.search(r'Video:\s+h264\b', source)), 'yuv420p': 'yuv420p' in source,
            'no_audio': 'Audio:' not in source, 'decode_exit_code': result.returncode,
            'source_SHA256': sha256_file(path), 'decoder_end_of_stream': 'progress=end' in result.stdout}
        row['pass'] = bool(result.returncode == 0 and count == old['frames'] and row['H264'] and
                           row['yuv420p'] and row['no_audio'] and row['decoder_end_of_stream'])
        rows.append(row)
    report = {'all7streams_full_decode_count_codec_noaudio_pass': len(rows) == 7 and all(r['pass'] for r in rows),
        'streams': rows, 'task_success_not_a_media_gate': True}
    atomic_json(directory/'independent_codec_audit.json', report)
    return report


def main(args):
    root = args.base.resolve()
    outputs = []
    for audit in sorted(root.rglob('camera_audit.json')):
        directory = audit.parent
        if 'pilots' in directory.parts and not args.include_pilots:
            continue
        if not json.loads(audit.read_text()).get('media_decode_sync_pass'):
            continue
        report = inspect(directory)
        outputs.append({'directory': str(directory.relative_to(root)),
                        'codec_pass': report['all7streams_full_decode_count_codec_noaudio_pass']})
        print(json.dumps(outputs[-1]), flush=True)
    sections = []
    for row in outputs:
        directory = root/row['directory']
        original = json.loads((directory/'camera_audit.json').read_text())
        name = row['directory']
        sections.append('<section><h2>'+html.escape(name)+'</h2><video controls preload="metadata" width="800" src="'+
            html.escape(name+'/view_1.mp4', quote=True)+'"></video><p>'+ ' · '.join(
                f'<a href="{html.escape(name+f"/view_{i}.mp4", quote=True)}">view{i}</a>' for i in range(1, 7))+
            f' · <a href="{html.escape(name+"/handeye.mp4", quote=True)}">native hand-eye</a></p><p>'+html.escape(
                f'Media: {row["codec_pass"]}; benchmark region success: {original["benchmark_region_success"]}; '+
                f'qualitative completion: {original.get("qualitative_task_complete", "see report")}.')+'</p></section>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R51 audited videos</title>'+
        '<h1>UR5e + Robotiq: audited qualitative recordings</h1><p>SmolVLA: verified replay of saved genuine neural controls; '+
        'NOT fresh inference or new SR trials. Oracle: explicitly adapted feedback controller on the SAME frozen physical world; '+
        'NOT trained RL. Only mode03 hand-eye inset. All videos play at real simulation time.</p>'+''.join(sections))
    atomic_json(root/'media_inventory.json', {'audited_method_outputs': outputs,
        'unique_streams_per_method_output': 7, 'number_of_camera_views_is_NOT_trial_count': True,
        'gallery_is_a_snapshot_not_automatic_full_suite_completion': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--include-pilots', action='store_true')
    main(parser.parse_args())
