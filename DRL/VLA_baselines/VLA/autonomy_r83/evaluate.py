"""Generic canonical neural chunks; unchanged R62 PyBullet plant/metric/safety."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import requests
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.benchmark import _interaction_audit
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.new_vlas.evaluate import png
from VLA.scene64_conditioning import model_input
from VLA.privileged_visual_r53.scene_map import render
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.autonomy_r83.common import read, verify, PORT, TASKS, metrics


def main(a):
    method, cohort = read(a.method), read(a.cohort)
    verify(method['code'])
    verify(method['weights_sha256'])
    final = cohort['role'] == 'fresh_secondary_final800'
    if final and (len(cohort['seeds']) != 200 or not method.get('frozen_after_confirmation')):
        raise ValueError('Final800 requires200 locked seeds and confirmed frozen method')
    if cohort['role'] not in ('development_selection', 'development_confirmation', 'fresh_secondary_final800'):
        raise ValueError('Explicit disjoint evaluation role required')
    if len(set(cohort['seeds'])) != len(cohort['seeds']):
        raise ValueError('Duplicate scene seed')
    a.output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    url = f'http://127.0.0.1:{a.port}'
    r = session.get(url + '/health', timeout=30)
    r.raise_for_status()
    meta = r.json()
    if not meta['actual_neural_model'] or meta['method_sha256'] != sha256_file(a.method):
        raise ValueError('Wrong actual neural method')
    horizon = int(meta['native_horizon'])
    if set(method['prefixes']) != set(TASKS) or any(not 1 <= k <= horizon for k in method['prefixes'].values()):
        raise ValueError('Invalid native execution prefix')
    protocol = read(PROFILE_PROTOCOL)
    profile = read(protocol['observation_profile_file'])
    if profile_hash(profile) != protocol['observation_profile_sha256']:
        raise ValueError('Shared camera/robot profile changed')
    atomic_json(a.output / 'PLAN.json', {'method': method, 'cohort': cohort,
        'protocol': PROFILE_PROTOCOL, 'actual_primary_success': protocol['success'],
        'new_action_controller_or_Oracle': False, 'final_saves_video_or_images': False})
    rows, queries, controls = [], 0, 0
    started = time.monotonic()

    def progress(status, **extra):
        atomic_json(a.output / 'progress.json', {'status': status, 'rows': rows, 'per_action': metrics(rows),
            'completed_trials': len(rows), 'planned_trials': len(cohort['seeds']) * 4,
            'elapsed_wall_s': time.monotonic() - started, 'native_queries': queries, 'motor_controls': controls,
            'method': str(a.method), 'cohort_role': cohort['role'], **extra})
    try:
        with (a.output / 'NEURAL_QUERIES.jsonl').open('x') as journal:
            progress('running_actual_neural')
            for seed in cohort['seeds']:
                verify(method['code'])
                rng = np.random.default_rng(seed)
                tasks = [randomize_rl_scene(t, rng) for t in select_tasks('all')]
                folder = a.output / f'seed{seed}'
                folder.mkdir()
                traces = {}
                env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e',
                    image_size=384, control_hz=10, assist_grasp=True, seed=seed, required_success_steps=1,
                    use_semantic_success=True, success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                    semantic_success_thresholds=protocol['success']['semantic_thresholds'])
                try:
                    for task in tasks:
                        obs = env.reset(task)
                        if env.is_semantic_success():
                            raise ValueError('Initially successful scene: never reroll')
                        discard_snapshot(folder, task.name, obs, env)
                        queue, trace, index, done, digest = deque(), [], 0, False, None
                        snapshots, closest = [], float('inf')

                        def capture(label):
                            if final or seed != cohort['seeds'][0]:
                                return
                            snapshots.append((label, int(obs.step), [np.asarray(obs.front_rgb).copy(),
                                np.asarray(obs.wrist_rgb).copy(), render(model_input(env, obs)['scene64'], task.name)]))
                        capture('reset')
                        while not done:
                            raw = model_input(env, obs)['scene64']
                            before = _interaction_audit(env)
                            if not queue:
                                payload = {'instruction': obs.prompt, 'scene64': raw.tolist(),
                                    'ee_position': obs.ee_position.tolist(), 'ee_rotation_6d': obs.ee_rotation_6d.tolist(),
                                    'front_png': png(obs.front_rgb), 'wrist_png': png(obs.wrist_rgb),
                                    'seed': seed, 'inference_index': index}
                                t0 = time.monotonic()
                                response = session.post(url + '/act', json=payload, timeout=300)
                                response.raise_for_status()
                                reply = response.json()
                                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                                if (not reply['actual_neural_model'] or reply['input_sha256'] != digest or
                                        reply['receipt']['future_teacher_phase_or_labels'] or
                                        reply['receipt']['geometric_action_residual_or_Oracle'] or
                                        reply['map_RGB_sha256'] != hashlib.sha256(render(raw, task.name).tobytes()).hexdigest()):
                                    raise ValueError('Incorrect actual model/input receipt')
                                action = np.asarray(reply['actions'], np.float32)
                                if action.shape != (horizon, 7) or not np.isfinite(action).all():
                                    raise ValueError('Malformed native canonical chunk')
                                journal.write(json.dumps({'seed': seed, 'task': task.name, 'index': index,
                                    'actions': action.tolist(), 'input_sha256': digest, 'receipt': reply['receipt'],
                                    'latency_s': time.monotonic() - t0}) + '\n')
                                journal.flush()
                                queue.extend(action[:method['prefixes'][task.name]].copy())
                                queries += 1
                                index += 1
                            command = queue.popleft()
                            obs, _, done, info = env.step(command)
                            controls += 1
                            after = _interaction_audit(env)
                            post = model_input(env, obs)['scene64']
                            trace.append({'step': int(obs.step), 'action': command.tolist(),
                                'executed_canonical_action': info['executed_canonical_action'],
                                'pre_action_scene64': raw.tolist(), 'post_action_scene64': post.tolist(),
                                'before': before, 'after': after, 'query_sha256': digest,
                                'success': bool(info['success']), 'safety_violation': info.get('safety_violation')})
                            distance = float(np.linalg.norm(post[11:14]))
                            if distance < closest:
                                closest = distance
                        capture('terminal')
                        traces[task.name] = trace
                        atomic_json(folder / 'trajectory.json', traces)
                        if snapshots:
                            canvas = Image.new('RGB', (768, len(snapshots) * 280), 'white')
                            draw = ImageDraw.Draw(canvas)
                            for ri, (label, step, views) in enumerate(snapshots):
                                for col, image in enumerate(views):
                                    canvas.paste(Image.fromarray(image).resize((256, 256)), (col * 256, ri * 280 + 24))
                                    draw.text((col * 256 + 2, ri * 280 + 3), f'{label} t{step}', fill='black')
                            canvas.save(folder / (task.name + '_REVIEW.png'))
                        rows.append({'seed': seed, 'task': task.name, 'success': bool(info['success']),
                            'steps': len(trace), 'safety_violation': info.get('safety_violation'),
                            'trajectory': str(folder / 'trajectory.json'), 'minimum_target_distance_m': closest})
                        progress('running_actual_neural')
                finally:
                    env.close()
            progress('completed_actual_final800' if final else 'completed_actual_development')
    except Exception as error:
        progress('integration_error_NOT_zero_model_SR', error=str(error))
        raise
    finally:
        session.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('method', 'cohort', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, default=PORT)
    main(p.parse_args())
