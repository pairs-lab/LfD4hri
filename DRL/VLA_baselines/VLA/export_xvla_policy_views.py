"""Export three verified X-VLA rollouts with the real policy camera inputs.

No neural training, new SR trials, Oracle actions, or live control modifications.
Only recorded geometry is restored, in an isolated non-stepping environment.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('/home/uet/Documents/X_VLA_view6_policy_cameras_20261031')
SOURCE_CASES = (
    ('01_green_block_purple_bowl', 20269719,
     'VLA/qualitative/ur5e_xvla_put_r58_complete_20261017/_renders/scene01_seed20269719_VLA'),
    ('02_red_block_orange_block', 20269711,
     'VLA/qualitative/ur5e_xvla_put_r58_complete_20261017/_renders/scene02_seed20269711_VLA'),
    ('03_gray_block_target_miss', 20271060,
     'VLA/qualitative/ur5e_xvla_put_r60_named_failures_20261017/_renders/scene13_seed20271060_VLA'),
)
PINNED_BULLET = '43bacc0f4d1bdae2e05698a98cb3b0fe03238127c7c606992f55593e6841e2af'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    from VLA.common.artifact_io import atomic_json
    atomic_json(Path(path), value)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def save_rgb(path, pixels):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    Image.fromarray(pixels).save(path)


def native_images(pixels):
    """Same normalization -> CUDA native resize -> inverse normalization.

    This reconstructs the visual tensors; it does not claim a new neural run.
    Raw384 images are independently hash-verified against actual old RPCs.
    """
    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad
    if not torch.cuda.is_available():
        raise RuntimeError('Use the CUDA preprocessing path, no silent CPU substitution')
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
    x = torch.as_tensor(pixels, device='cuda').permute(2, 0, 1).float() / 255
    x = resize_with_pad(((x-mean)/std)[None], 224, 224, pad_value=0.0)[0]
    shown = ((x*std+mean).clamp(0, 1)*255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return shown, array_hash(x.detach().cpu().numpy())


def prepare(output):
    import pybullet as p
    from VLA.qualitative_r57.common import make_env, reset_put
    from VLA.qualitative_recorder_r51 import apply_render_state
    from VLA.new_vlas.evaluate import png
    from VLA.privileged_visual_r53.scene_map import render as render_map
    from utils.camera import CINEMATIC_VIEWS

    if digest(p.__file__) != PINNED_BULLET:
        raise ValueError('Wrong physics/policy image binding')
    cases = []
    for name, seed, relative in SOURCE_CASES:
        case_dir = output/'evidence'/name
        metadata_path = case_dir/'episode.json'
        if metadata_path.exists():
            saved = read(metadata_path)
            for path, expected in saved['source_sha256'].items():
                if digest(path) != expected:
                    raise ValueError('Existing export source changed')
            cases.append(saved)
            continue
        case_dir.mkdir(parents=True, exist_ok=False)
        inputs = case_dir/'policy_inputs'
        inputs.mkdir()
        source = ROOT/relative
        proof = read(source/'replay_proof.json')
        renderer = read(source/'renderer.json')
        capture = Path(renderer['source_capture'])
        trace_path = Path(proof['source_trajectory'])
        trace = read(trace_path)['pick_place']
        geometry = read(capture/'recorded_physics_states.json')
        by_tick = {r['physics_tick']: r for r in geometry}
        frames = read(source/'frames.json')
        if len(frames) != len(geometry) or len(trace) != len(proof['controls']):
            raise ValueError('Source frame/control coverage mismatch')
        if not proof['passed_exact_replay'] or proof['seed'] != seed:
            raise ValueError('Not a verified X-VLA source')
        if digest(trace_path) != proof['source_trajectory_sha256']:
            raise ValueError('Actual neural source trace changed')
        env = make_env(seed)
        queries = []
        try:
            task, obs = reset_put(env, seed)
            for control, row in enumerate(trace):
                if row['action_index_in_chunk'] != 0:
                    continue
                if control:
                    errors = apply_render_state(by_tick[control*24]['geometry'])
                    obs = env.observe()
                else:
                    errors = {'position_m': 0., 'quaternion_component': 0., 'joint_rad': 0.}
                expected = proof['controls'][control]['before_signature']
                raw = np.asarray(row['pre_action_scene64'], np.float32)
                for role, pixels in [('front', obs.front_rgb), ('wrist', obs.wrist_rgb)]:
                    if array_hash(pixels) != expected[role+'_rgb']:
                        raise ValueError(f'Actual {role} image hash differs at control {control}')
                    if array_hash(env._last_segmentation[role]) != expected['segmentation_cache'][role]:
                        raise ValueError('Policy camera segmentation differs')
                payload = {'instruction': obs.prompt, 'scene64': raw.tolist(),
                    'ee_position': obs.ee_position.tolist(), 'ee_rotation_6d': obs.ee_rotation_6d.tolist(),
                    'front_png': png(obs.front_rgb), 'wrist_png': png(obs.wrist_rgb)}
                rpc_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if rpc_hash != row['neural_chunk_input_sha256']:
                    raise ValueError('Original complete RGB/proprio/privileged RPC differs')
                query = {'query': len(queries), 'control_index': control, 'physics_tick': control*24,
                    'simulation_time_s': control/10, 'original_RPC_sha256': rpc_hash,
                    'raw_front_wrist_and_segmentation_bit_exact': True,
                    'geometry_restore_max_errors': errors, 'files': {}, 'tensor_sha256': {}}
                for role, pixels in [('front', obs.front_rgb), ('wrist', obs.wrist_rgb),
                                     ('map', render_map(raw, 'pick_place'))]:
                    stem = f'query{len(queries):02d}_step{control:03d}_{role}'
                    original = inputs/(stem+'_raw384.png')
                    viewed = inputs/(stem+'_model224.png')
                    save_rgb(original, pixels)
                    shown, tensor_hash = native_images(pixels)
                    save_rgb(viewed, shown)
                    query['files'][role] = {'raw384': str(original.relative_to(output)),
                        'model224': str(viewed.relative_to(output)), 'raw_array_sha256': array_hash(pixels),
                        'model224_png_sha256': digest(viewed)}
                    query['tensor_sha256'][role] = tensor_hash
                queries.append(query)
                dump(output/'progress.json', {'stage': 'verified_policy_inputs', 'case': name,
                    'verified_queries': len(queries), 'control': control})
                print(json.dumps({'case': name, 'query': len(queries)-1, 'control': control,
                    'RGB_and_RPC_exact': True}), flush=True)
            _, profile = __import__('VLA.qualitative_r57.common', fromlist=['protocol']).protocol()
        finally:
            env.close()
        source_paths = [source/'wide_6.mp4', source/'frames.json', source/'replay_proof.json',
            source/'renderer.json', capture/'recorded_physics_states.json', trace_path,
            ROOT/'VLA/new_vlas/runtime.py',
            ROOT/'VLA/third_party/lerobot/src/lerobot/policies/common/vla_utils.py']
        metadata = {'name': name, 'seed': seed, 'instruction': task.prompt, 'task': 'pick_place',
            'model': 'X-VLA + privileged simulation inputs', 'checkpoint': proof['checkpoint'],
            'scope': 'export_of_verified_actual_neural_rollout_NOT_new_inference_or_new_SR_trial',
            'source_clean_view6': str(source/'wide_6.mp4'), 'source_frames': str(source/'frames.json'),
            'physics_binding_sha256': PINNED_BULLET, 'camera_profile': profile,
            'wide_camera': CINEMATIC_VIEWS[5], 'frames': len(frames), 'fps': 30,
            'duration_s': len(frames)/30, 'source_first_frame_physics_tick': frames[0]['physics_tick'],
            'source_last_frame_physics_tick': frames[-1]['physics_tick'],
            'benchmark_region_success': proof['benchmark_region_success'],
            'benchmark_boundary_physics_tick': proof['benchmark_boundary_physics_tick'],
            'terminal_release_supervisor': proof.get('terminal_release_supervisor'),
            'camera_display': 'latest actual model-query input; sample-and-hold until next query',
            'policy_control_hz': 10, 'execution_steps_per_chunk': 30,
            'neural_query_interval_s': 3., 'displayed_camera_order': ['front', 'wrist'],
            'not_displayed_in_video': ['privileged scene-map image', 'numeric67/proprio/text'],
            'raw384_RGB_and_RPC_provenance': 'bit-exact against saved actual inference evidence',
            'model224_provenance': 'reconstructed native CUDA normalize/resize, inverse-normalized for viewing',
            'new_neural_tensor_or_RNG_parity_claimed': False,
            'source_sha256': {str(q): digest(q) for q in source_paths}, 'queries': queries}
        dump(metadata_path, metadata)
        cases.append(metadata)
    dump(output/'episodes.json', cases)
    return cases


def intersection(box, rectangle):
    x, y, w, h = rectangle
    a, b, c, d = box
    return max(0., min(c, x+w+3)-max(a, x-3))*max(0., min(d, y+h+3)-max(b, y-3))


def choose_layout(cases):
    frames = [r for case in cases for r in read(case['source_frames'])]
    for size in (448, 384, 336, 280):
        for side in ('left', 'right'):
            x = 35 if side == 'left' else 1920-35-size
            rects = [[x, 37, size, size], [x, 37+size+12, size, size]]
            maximum = 0.
            for row in frames:
                wide = row['wide'][5]
                boxes = [wide['projected_pick_box']]
                if wide.get('projected_place_box') is not None:
                    boxes.append(wide['projected_place_box'])
                maximum = max(maximum, sum(intersection(box, rect) for box in boxes for rect in rects))
            if maximum == 0:
                return {'front': rects[0], 'wrist': rects[1], 'side': side,
                    'square_aspect_ratio_preserved': True, 'border_pixels': 3,
                    'maximum_target_goal_AABB_overlap_all_three_trajectories': maximum}
    raise ValueError('No common non-occluding camera inset layout')


def shown_query(case, physics_tick):
    ticks = [q['physics_tick'] for q in case['queries']]
    return max(0, int(np.searchsorted(ticks, physics_tick, side='right'))-1)


def encode(output, cases):
    layout = choose_layout(cases)
    dump(output/'layout.json', layout)
    for case in cases:
        target = output/(case['name']+'_view6_front_wrist.mp4')
        if target.exists():
            raise FileExistsError('Never overwrite a completed video: '+str(target))
        frames = read(case['source_frames'])
        inputs = [{role: np.asarray(Image.open(output/q['files'][role]['model224']).convert('RGB'))
                   for role in ('front', 'wrist')} for q in case['queries']]
        source = cv2.VideoCapture(case['source_clean_view6'])
        if not source.isOpened() or abs(source.get(cv2.CAP_PROP_FPS)-30) > 1e-6:
            raise ValueError('Source FPS/open gate failed')
        stream = imageio_ffmpeg.write_frames(str(target), (1920, 1080), fps=30,
            codec='libx264', quality=None, macro_block_size=1, ffmpeg_log_level='error',
            output_params=['-preset', 'fast', '-crf', '10', '-tune', 'zerolatency',
                '-rc-lookahead', '0', '-bf', '0', '-refs', '1', '-threads', '2',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        stream.send(None)
        count = 0
        timing = []
        try:
            for i, row in enumerate(frames):
                ok, bgr = source.read()
                if not ok or bgr.shape != (1080, 1920, 3):
                    raise ValueError('Source stream truncated or wrong resolution')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                qi = shown_query(case, row['physics_tick'])
                for role in ('front', 'wrist'):
                    x, y, w, h = layout[role]
                    rgb[y-3:y+h+3, x-3:x+w+3] = (221, 227, 232)
                    rgb[y:y+h, x:x+w] = cv2.resize(inputs[qi][role], (w, h), interpolation=cv2.INTER_NEAREST)
                stream.send(np.ascontiguousarray(rgb))
                timing.append({'frame': i, 'encoded_pts_s': i/30,
                    'physics_tick': row['physics_tick'], 'simulation_time_s': row['physics_tick']/240,
                    'query': qi, 'observation_age_s': (row['physics_tick']-case['queries'][qi]['physics_tick'])/240,
                    'phase': 'VLA_native' if row['physics_tick'] <= case['benchmark_boundary_physics_tick']
                             else 'disclosed_terminal_release_supervisor_NOT_VLA'})
                count += 1
                if count % 60 == 0:
                    dump(output/'progress.json', {'stage': 'encoding', 'case': case['name'],
                        'frames': count, 'total_frames': len(frames)})
                    print(json.dumps({'encoding': case['name'], 'frames': count}), flush=True)
            if source.read()[0]:
                raise ValueError('Source contains more frames than the original timing evidence')
        finally:
            source.release()
            stream.close()
        dump(output/'evidence'/case['name']/'frame_timing.json', timing)


def audit(output, cases):
    layout = read(output/'layout.json')
    results = []
    for case in cases:
        video = output/(case['name']+'_view6_front_wrist.mp4')
        frame_rows = read(case['source_frames'])
        reader = cv2.VideoCapture(str(video))
        if not reader.isOpened():
            raise ValueError('Encoded video cannot be decoded')
        fps = reader.get(cv2.CAP_PROP_FPS)
        inputs = [{role: np.asarray(Image.open(output/q['files'][role]['model224']).convert('RGB'))
                   for role in ('front', 'wrist')} for q in case['queries']]
        tiles = []
        count = 0
        max_error = {'front': 0., 'wrist': 0.}
        # Show each actual input change, plus a mid-approach frame and the final image.
        review_indices = {0, len(frame_rows)//2, len(frame_rows)-1}
        for query in case['queries'][1:]:
            review_indices.add(next(i for i, r in enumerate(frame_rows) if r['physics_tick'] >= query['physics_tick']))
        while True:
            ok, bgr = reader.read()
            if not ok:
                break
            if count >= len(frame_rows) or bgr.shape != (1080, 1920, 3):
                raise ValueError('Encoded dimensions/frame count mismatch')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            qi = shown_query(case, frame_rows[count]['physics_tick'])
            for role in ('front', 'wrist'):
                x, y, w, h = layout[role]
                expected = cv2.resize(inputs[qi][role], (w, h), interpolation=cv2.INTER_NEAREST)
                error = float(np.abs(rgb[y:y+h, x:x+w].astype(np.int16)-expected.astype(np.int16)).mean())
                max_error[role] = max(max_error[role], error)
                if error >= 6:
                    raise ValueError('Encoded inset does not match the actual policy input')
            if count in review_indices:
                review_dir = output/'review'/case['name']
                review_dir.mkdir(parents=True, exist_ok=True)
                save_rgb(review_dir/f'frame_{count:05d}.png', rgb)
                tile = Image.fromarray(cv2.resize(rgb, (768, 432), interpolation=cv2.INTER_AREA))
                canvas = Image.new('RGB', (768, 461), (245, 245, 245))
                canvas.paste(tile, (0, 29))
                ImageDraw.Draw(canvas).text((8, 8),
                    f'frame {count} | simulation {frame_rows[count]["physics_tick"]/240:.2f}s | query {qi}',
                    fill=(0, 0, 0))
                tiles.append(canvas)
            count += 1
        reader.release()
        if count != case['frames'] or fps != 30:
            raise ValueError('Encoded FPS/count audit failed')
        cols = 2
        sheet = Image.new('RGB', (768*cols, 461*((len(tiles)+cols-1)//cols)), (245, 245, 245))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols)*768, (i//cols)*461))
        sheet.save(output/'review'/(case['name']+'_contact_sheet.jpg'), quality=92)
        result = {'case': case['name'], 'video': video.name, 'sha256': digest(video),
            'full_stream_decode': True, 'frames': count, 'fps': fps, 'resolution': [1920, 1080],
            'duration_s': count/fps, 'maximum_inset_mean_absolute_pixel_error': max_error,
            'all_original_query_RGB_and_RPC_hashes_exact': True,
            'actual_original_neural_queries': len(case['queries']), 'review_frame_indices': sorted(review_indices)}
        results.append(result)
        print(json.dumps({'audited': case['name'], 'frames': count, 'max_error': max_error}), flush=True)
    dump(output/'AUDIT.json', {'automated_passed': True,
        'manual_visual_review_required': True, 'exactly_three_main_videos': len(results) == 3,
        'no_new_neural_inference_or_SR_trials': True, 'results': results})
    return results


def write_delivery(output, cases, results):
    rows = []
    for case, result in zip(cases, results, strict=True):
        title = html.escape(case['instruction'])
        rows.append(f'<section><h2>{title}</h2><p>Seed {case["seed"]}; '
            f'{result["duration_s"]:.2f} seconds. Top inset: front. Bottom inset: wrist.</p>'
            f'<video controls preload="metadata" width="960" src="{result["video"]}"></video>'
            f'<p><a href="review/{case["name"]}_contact_sheet.jpg">Decoded-frame review</a> | '
            f'<a href="evidence/{case["name"]}/episode.json">Input provenance</a></p></section>')
    page = '<!doctype html><meta charset="utf-8"><title>X-VLA actual policy cameras</title>'
    page += '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;background:#fafafa}video{max-width:100%}section{margin:35px 0}</style>'
    page += '<h1>X-VLA: view 6 + front / wrist policy inputs</h1>'
    page += '<p>The square insets are actual query inputs, not presentation close-ups. They update every 3 seconds because the frozen policy executes 30 actions at 10 Hz. The additional privileged map/state inputs are recorded in the evidence, not shown as camera panels.</p>'
    page += '<p>These are exports of previously executed X-VLA rollouts, not new evaluations. Cases 1/2 retain the source opening-only release supervisor after the VLA boundary; case 3 retains the full timeout/failure.</p>'
    (output/'index.html').write_text(page+'\n'.join(rows))
    lines = ['# X-VLA — view 6 + actual front/wrist policy observations', '',
        'Exactly three main MP4 videos; 1920×1080, H.264, yuv420p, 30 FPS.', '',
        '- Top square inset: external **front policy camera**.',
        '- Bottom square inset: **wrist/eye-in-hand policy camera**.',
        '- These are inverse-normalized native 224×224 inputs, enlarged without changing aspect ratio.',
        '- Input updates are genuinely 3 seconds apart (30-action chunks, 10 Hz controls). Frozen inset frames are intentional, not a playback fault.',
        '- Every raw camera query is hash-identical to the original neural rollout, including its complete RPC payload.',
        '- Privileged scene map, numerical state and text were also supplied to X-VLA; this is not a wrist-only or vanilla visual baseline.',
        '- Source rollouts and their failures are preserved; no new policy evaluation or training is claimed.',
        '- Cases 1/2 include the previously disclosed terminal opening-only release supervisor. Case 3 is the full original target-miss/timeout rollout.',
        '- No camera, controller, checkpoint, action, physics setting, success threshold, or existing source video was modified.', '',
        '| Video | Prompt | Seconds | Original benchmark result |', '|---|---|---:|---|']
    for case, result in zip(cases, results, strict=True):
        lines.append(f'| [{result["video"]}]({result["video"]}) | {case["instruction"]} | '
            f'{result["duration_s"]:.2f} | {"Reached goal region" if case["benchmark_region_success"] else "Failure / timeout"} |')
    lines += ['', 'See `AUDIT.json`, `layout.json`, `review/`, and per-episode `evidence/`.',
        'The 224px tensors were reconstructed using native CUDA preprocessing; a fresh model forward/RNG parity test is not claimed.',
        'The model checkpoint and actual camera profile are recorded in every `episode.json`.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--phase', choices=('prepare', 'encode', 'audit', 'all'), default='all')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path('/home/uet/Documents')) or output == Path('/home/uet/Documents'):
        raise ValueError('Export only into a dedicated Documents subdirectory')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if args.phase in ('prepare', 'all'):
            cases = prepare(output)
        else:
            cases = read(output/'episodes.json')
        if args.phase in ('encode', 'all'):
            encode(output, cases)
        if args.phase in ('audit', 'all'):
            results = audit(output, cases)
            write_delivery(output, cases, results)
        dump(output/'progress.json', {'stage': 'completed_'+args.phase,
            'elapsed_wall_seconds': time.monotonic()-started, 'pid': os.getpid()})
    except Exception as error:
        dump(output/'ERROR.json', {'error': f'{type(error).__name__}: {error}',
            'partial_artifacts_preserved': True, 'elapsed_wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
