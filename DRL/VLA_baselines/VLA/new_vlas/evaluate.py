"""Matched development40 (10/action), original SIM interpreter and video.

Native models run in isolated server processes; PyBullet, camera, robot,
gripper, scenes, thresholds and safety are the existing V9 shared protocol.
Integration failures do not become fake task failures or trigger rerolls.
"""
import argparse
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import requests

from VLA.benchmark import _interaction_audit, _save_observation_snapshot
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.new_vlas.action_contracts import canonical_from_absolute20
from VLA.new_vlas.policy_outputs import native_gripper_command
from VLA.scene64_conditioning import model_input
from VLA.run_cross_vla120 import PROFILE_PROTOCOL, SEEDS

ROOT = Path(__file__).resolve().parents[2]


def png(pixels):
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cp = args.checkpoint.resolve()
    cfg = json.loads((cp/'config.json').read_text())
    protocol = json.loads(PROFILE_PROTOCOL.read_text())
    profile = json.loads(Path(protocol['observation_profile_file']).read_text())
    if profile_hash(profile) != protocol['observation_profile_sha256']:
        raise ValueError('Frozen common camera/home profile changed')
    data_proof = json.loads((Path(cfg['data_root'])/'audit.json').read_text())
    original_proof_path = next(Path(p) for p in data_proof['bound_files'] if p.endswith('full_train64/audit.json'))
    original_proof = json.loads(original_proof_path.read_text())
    source_manifest = Path(original_proof['source'])/'dataset_manifest.json'
    if sha256_file(source_manifest) != original_proof['source_manifest_sha256']:
        raise ValueError('Training camera/robot source manifest changed')
    source_profile = json.loads(source_manifest.read_text())['observation_profile']
    if profile_hash(source_profile) != profile_hash(profile):
        raise ValueError('Train/deploy camera/home/gripper domain mismatch')
    if cfg['actual_optimizer_updates'] <= 2:
        raise ValueError('Two-step numerical gate is not a reportable fine-tuned VLA')
    seeds = SEEDS[:10]
    plan = {'scope': '40_matched_DEVELOPMENT_trials_NOT_reserved_final_test',
        'family': cfg['family'], 'checkpoint': str(cp), 'seeds': seeds,
        'trials_per_action': 10, 'planned_trials': 40,
        'same_robot_objects_gripper_camera_and_physics': True,
        'success': {**protocol['success'],
            'privileged_coordinates_used_only_by_evaluator': False,
            'policy_privileged_observations_explicitly_allowed': True},
        'oracle_actions': False, 'privileged67': True,
        'replan_steps': 1, 'rerolls': 0, 'same_safety_aborts_count_as_failure': True,
        'native_gripper_postprocessing': 'official_model_binary_threshold0.5_no_added_neutral_band',
        'released_stable_place_NOT_same_as_primary_region_success': True,
        'environment_fingerprint': environment_runtime_fingerprint(profile),
        'simulator_binary_sha256': sha256_file(__import__('pybullet').__file__),
        'frozen': {str(p): sha256_file(p) for p in (PROFILE_PROTOCOL, Path(__file__),
            Path(__file__).with_name('server.py'), Path(__file__).with_name('runtime.py'),
            Path(__file__).with_name('action_contracts.py'))}}
    atomic_json(out/'plan.json', plan)
    python = ROOT/'VLA/.venvs'/('xvla' if cfg['family'] == 'xvla' else 'openvla')/'bin/python'
    rows, neural_controls, child = [], 0, None
    def progress(status, **extra):
        atomic_json(out/'progress.json', {'status': status, 'family': cfg['family'],
            'checkpoint': str(cp), 'completed_trials': len(rows), 'planned_trials': 40,
            'successes': sum(row['success'] for row in rows), 'rows': rows,
            'neural_controls': neural_controls, 'statistics': success_statistics(rows) if rows and len(rows)%4 == 0 else None,
            'model_input_contains_privileged67': True, 'final_test_used': False, **extra})
    try:
        progress('starting_native_server')
        with (out/'server.log').open('x') as log:
            child = subprocess.Popen([str(python), '-m', 'VLA.new_vlas.server', '--checkpoint', str(cp),
                '--port', str(args.port)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        atomic_json(out/'server_owner.json', {'pid': child.pid, 'checkpoint': str(cp), 'port': args.port})
        url = f'http://127.0.0.1:{args.port}'
        session = requests.Session()
        until = time.monotonic()+1200
        while True:
            if child.poll() is not None:
                raise RuntimeError('Own native server exited: '+str(child.returncode))
            try:
                response = session.get(url+'/health', timeout=3)
                response.raise_for_status()
                metadata = response.json()
                if metadata['checkpoint'] != str(cp) or metadata['family'] != cfg['family']:
                    raise ValueError('Different server already owns declared port')
                atomic_json(out/'actual_server_metadata.json', metadata)
                break
            except requests.RequestException:
                if time.monotonic() > until:
                    raise TimeoutError('Own server did not become healthy')
                time.sleep(3)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tasks = [randomize_rl_scene(task, rng) for task in select_tasks('all')]
            folder = out/f'test_seed{seed}'
            folder.mkdir()
            traces, writer = {}, None
            env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e',
                image_size=384, control_hz=10, assist_grasp=True, seed=seed, required_success_steps=1,
                use_semantic_success=True, success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['success']['semantic_thresholds'])
            try:
                if seed == seeds[0]:
                    writer = imageio.get_writer(str(folder/'rollout.mp4'), fps=10, codec='libx264', quality=8)
                for task in tasks:
                    obs = env.reset(task)
                    if env.is_semantic_success():
                        raise ValueError('Initially successful scene invalid; do not replace scene')
                    _save_observation_snapshot(folder, task.name, obs, env)
                    trace, done = [], False
                    while not done:
                        state = model_input(env, obs)
                        payload = {'instruction': obs.prompt, 'scene64': state['scene64'].tolist(),
                            'ee_position': obs.ee_position.tolist(), 'ee_rotation_6d': obs.ee_rotation_6d.tolist(),
                            'front_png': png(obs.front_rgb), 'wrist_png': png(obs.wrist_rgb)}
                        before = _interaction_audit(env)
                        reply = session.post(url+'/act', json=payload, timeout=900)
                        reply.raise_for_status()
                        result = reply.json()
                        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                        if result.get('actual_neural_model') is not True or result['input_sha256'] != digest:
                            raise ValueError('Native neural action/input identity missing')
                        native = np.asarray(result['actions'], np.float32)
                        expected = (30, 20) if cfg['family'] == 'xvla' else (8, 7)
                        if native.shape != expected or not np.isfinite(native).all():
                            raise ValueError('Malformed native learned action')
                        if cfg['family'] == 'xvla':
                            if result['action_kind'] != 'native_absolute_ee20_closure':
                                raise ValueError('Native action contract differs')
                            action = canonical_from_absolute20(native[0], obs.ee_position, obs.ee_rotation_6d)
                            action[6] = native_gripper_command(native[0, 9], convention='xvla_closed01')
                        else:
                            if result['action_kind'] != 'delta_ee7_open01':
                                raise ValueError('Native action contract differs')
                            action = native[0].copy()
                            action[6] = native_gripper_command(action[6], convention='adapter_open01')
                        obs, _, done, info = env.step(action)
                        neural_controls += 1
                        trace.append({'task': task.name, 'step': obs.step, 'action': action.tolist(),
                            'native_neural_action': native[0].tolist(), 'before': before,
                            'after': _interaction_audit(env), 'pre_action_scene64': state['scene64'].tolist(),
                            'success': bool(info['success']), 'safety_violation': info.get('safety_violation'),
                            'input_sha256': digest})
                        traces[task.name] = trace
                        atomic_json(folder/'trajectory.json', traces)
                        if writer:
                            writer.append_data(np.concatenate((env.render('overview'), obs.front_rgb, obs.wrist_rgb), axis=1))
                    rows.append({'seed': seed, 'task': task.name, 'success': bool(info['success']),
                        'steps': len(trace), 'safety_violation': info.get('safety_violation'),
                        'trajectory': str(folder/'trajectory.json')})
                    progress('running_actual_native_VLA')
            finally:
                if writer:
                    writer.close()
                env.close()
        if len(rows) != 40:
            raise ValueError('Incomplete actual40 cohort')
        progress('completed40_privileged_native_VLA')
        report = ['# Native privileged VLA development40', '',
            'Same V9 PyBullet environment;10 trials/action. Development, not reserved final test.', '',
            '| Action | Successes | Trials | SR |', '|---|---:|---:|---:|']
        for task in ('touch', 'pick', 'move', 'pick_place'):
            subset = [r for r in rows if r['task'] == task]
            count = sum(r['success'] for r in subset)
            report.append(f'| {task} | {count} | {len(subset)} | {100*count/len(subset):.1f}% |')
        report += ['', f'Overall: {sum(r["success"] for r in rows)}/40.',
            'Primary place success is reaching the target region, not necessarily release/stable placement.',
            'These runs use privileged simulator observations and must be labeled as such in the paper.']
        (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    except Exception as error:
        progress('failed_preserve_all_actual_trials', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Never signal unrelated old models or other user sessions.
                child.kill()
                child.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5843)
    main(parser.parse_args())
