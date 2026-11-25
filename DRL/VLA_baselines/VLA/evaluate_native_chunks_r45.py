"""Matched development8 using native action queues on an already-loaded VLA.

X-VLA's pinned config specifies30 execution steps; Adapter's official LIBERO
runner uses8. Compare these with the existing first-action-only runner, not
by changing its live cohort. No second GPU model, oracle, new score or reroll.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
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
from VLA.new_vlas.evaluate import png
from VLA.new_vlas.policy_outputs import native_gripper_command
from VLA.run_cross_vla120 import PROFILE_PROTOCOL, SEEDS
from VLA.scene64_conditioning import model_input
from VLA.stable_native_r44.conditioning import require_contract, contract


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    url = f'http://127.0.0.1:{args.port}'
    response = session.get(url+'/health', timeout=10)
    response.raise_for_status()
    metadata = response.json()
    if metadata['family'] != args.family or metadata['actual_optimizer_updates'] < (7000 if args.family == 'xvla' else 6000):
        raise ValueError('Require own actual postfit native VLA, no fallback server')
    cp = Path(metadata['checkpoint'])
    cfg = require_contract(cp)
    protocol = json.loads(PROFILE_PROTOCOL.read_text())
    profile = json.loads(Path(protocol['observation_profile_file']).read_text())
    if profile_hash(profile) != protocol['observation_profile_sha256']:
        raise ValueError('Frozen shared camera/home profile changed')
    proof = json.loads((Path(cfg['data_root'])/'audit.json').read_text())
    original = next(Path(p) for p in proof['bound_files'] if p.endswith('full_train64/audit.json'))
    source = json.loads(original.read_text())
    manifest = Path(source['source'])/'dataset_manifest.json'
    if sha256_file(manifest) != source['source_manifest_sha256']:
        raise ValueError('Training camera/home/robot manifest changed')
    if profile_hash(json.loads(manifest.read_text())['observation_profile']) != profile_hash(profile):
        raise ValueError('Actual train/deploy observation domain differs')
    native_horizon = 30 if args.family == 'xvla' else 8
    native_steps = args.execution_steps or native_horizon
    if not 1 <= native_steps <= native_horizon:
        raise ValueError('Execution prefix must be inside actual learned native chunk')
    seeds = SEEDS[:args.seed_count]
    planned = len(seeds)*4
    plan = {'scope': f'paired_DEVELOPMENT{planned}_NATIVE_CHUNK_EXECUTION_NOT_reserved_final_test',
        'family': args.family, 'checkpoint': str(cp), 'server_metadata': metadata,
        'planned_trials': planned, 'per_action': len(seeds), 'seeds': seeds, 'execution_steps_per_native_query': native_steps,
        'condition': contract(), 'same_physics_robot_gripper_objects_camera_success': True,
        'oracle_actions_or_geometric_action_correction': False, 'extra_GPU_models_loaded': False,
        'shared_10Hz_motor_steps_and_episode_horizons_unchanged': True,
        'success_checked_after_every_motor_step': True, 'rerolls': 0,
        'primary_place_success_does_NOT_require_release': True,
        'environment_fingerprint': environment_runtime_fingerprint(profile),
        'simulator_binary_sha256': sha256_file(__import__('pybullet').__file__),
        'training_source_camera_home_profile_identity_verified': True,
        'frozen': {str(p): sha256_file(p) for p in (Path(__file__), PROFILE_PROTOCOL)}}
    atomic_json(out/'plan.json', plan)
    rows, controls, queries = [], 0, 0
    def progress(status, **extra):
        atomic_json(out/'progress.json', {'status': status, 'family': args.family,
            'checkpoint': str(cp), 'completed_trials': len(rows), 'planned_trials': planned,
            'successes': sum(r['success'] for r in rows), 'rows': rows,
            'neural_controls': controls, 'native_neural_queries': queries,
            'statistics': success_statistics(rows) if rows and len(rows)%4 == 0 else None,
            'native_execution_chunk': native_steps, 'privileged67': True, 'final_test_used': False, **extra})
    try:
        progress('running_native_chunk_ablation')
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
                    if env.is_semantic_success(): raise ValueError('Initially successful scene, do not reroll')
                    _save_observation_snapshot(folder, task.name, obs, env)
                    queue, trace, done = deque(), [], False
                    query_hash, query_step = None, None
                    while not done:
                        state = model_input(env, obs)
                        if not queue:
                            payload = {'instruction': obs.prompt, 'scene64': state['scene64'].tolist(),
                                'ee_position': obs.ee_position.tolist(), 'ee_rotation_6d': obs.ee_rotation_6d.tolist(),
                                'front_png': png(obs.front_rgb), 'wrist_png': png(obs.wrist_rgb)}
                            reply = session.post(url+'/act', json=payload, timeout=900)
                            reply.raise_for_status()
                            result = reply.json()
                            query_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                            if result.get('actual_neural_model') is not True or result['input_sha256'] != query_hash:
                                raise ValueError('Native neural/input identity mismatch')
                            native = np.asarray(result['actions'], np.float32)
                            expected = (30, 20) if args.family == 'xvla' else (8, 7)
                            if native.shape != expected or not np.isfinite(native).all():
                                raise ValueError('Malformed native learned action chunk')
                            queue.extend(native[:native_steps].copy())
                            query_step = int(obs.step)
                            queries += 1
                        native_action = queue.popleft()
                        if args.family == 'xvla':
                            # Absolute future EEF targets are decoded against
                            # CURRENT measured pose, never the old query pose.
                            action = canonical_from_absolute20(native_action, obs.ee_position, obs.ee_rotation_6d)
                            action[6] = native_gripper_command(native_action[9], convention='xvla_closed01')
                        else:
                            action = native_action.copy()
                            action[6] = native_gripper_command(action[6], convention='adapter_open01')
                        before = _interaction_audit(env)
                        obs, _, done, info = env.step(action)
                        controls += 1
                        trace.append({'task': task.name, 'step': obs.step, 'action': action.tolist(),
                            'native_neural_action': native_action.tolist(), 'before': before,
                            'after': _interaction_audit(env), 'pre_action_scene64': state['scene64'].tolist(),
                            'success': bool(info['success']), 'safety_violation': info.get('safety_violation'),
                            'safety_diagnostics_before_rollback': info.get('safety_diagnostics_before_rollback'),
                            'safety_clamped_action': info.get('safety_clamped_action'),
                            'executed_canonical_action': info.get('executed_canonical_action'),
                            'neural_chunk_input_sha256': query_hash, 'chunk_query_step': query_step,
                            'action_index_in_chunk': int(obs.step)-query_step-1})
                        traces[task.name] = trace
                        atomic_json(folder/'trajectory.json', traces)
                        if writer:
                            writer.append_data(np.concatenate((env.render('overview'), obs.front_rgb, obs.wrist_rgb), axis=1))
                    rows.append({'seed': seed, 'task': task.name, 'success': bool(info['success']),
                        'steps': len(trace), 'safety_violation': info.get('safety_violation'),
                        'trajectory': str(folder/'trajectory.json')})
                    progress('running_native_chunk_ablation')
            finally:
                if writer: writer.close()
                env.close()
        progress(f'completed{planned}_native_chunk_DEVELOPMENT_NOT_reserved_final_test')
        report = ['# Native chunk execution development', '', f'Model: {args.family}. Execution prefix: {native_steps}/{native_horizon}.', '',
            '| Action | Successes | Trials | SR |', '|---|---:|---:|---:|']
        for task in ('touch', 'pick', 'move', 'pick_place'):
            subset = [row for row in rows if row['task'] == task]
            count = sum(row['success'] for row in subset)
            report.append(f'| {task} | {count} | {len(subset)} | {100*count/len(subset):.1f}% |')
        report += ['', f'Overall: {sum(row["success"] for row in rows)}/{planned}.',
            'Privileged-state-augmented native VLA. Same frozen robot/camera/physics/safety/success.',
            'Primary place success is target-region reaching, not necessarily release/stable placement.',
            'Development cohort; do not label as reserved final-test performance.']
        (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    except Exception as error:
        progress('failed_integration_preserve_completed_and_partial_trials_NO_fake_failure',
                 error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-count', type=int, choices=(2, 10), default=10)
    parser.add_argument('--execution-steps', type=int, default=0)
    parser.add_argument('--own-server-checkpoint', type=Path)
    args = parser.parse_args()
    child = None
    try:
        if args.own_server_checkpoint is not None:
            import os, subprocess, time
            from VLA.new_vlas.train import wait_gpu
            args.output.parent.mkdir(parents=True, exist_ok=True)
            admission = args.output.with_name(args.output.name+'_GPU_admission')
            admission.mkdir(exist_ok=False)
            wait_gpu(admission)
            sim_root = Path(__file__).resolve().parents[1]
            python = sim_root/'VLA/.venvs'/('xvla' if args.family == 'xvla' else 'openvla')/'bin/python'
            with args.output.with_name(args.output.name+'_server.log').open('x') as log:
                child = subprocess.Popen([str(python), '-m', 'VLA.stable_native_r44.server',
                    '--checkpoint', str(args.own_server_checkpoint.resolve()), '--port', str(args.port)],
                    cwd=sim_root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
                atomic_json(admission/'owner.json', {'pid': child.pid, 'checkpoint': str(args.own_server_checkpoint.resolve())})
                until = time.monotonic()+1200
                while True:
                    if child.poll() is not None: raise RuntimeError('Own native server exited')
                    try:
                        health = requests.get(f'http://127.0.0.1:{args.port}/health', timeout=3)
                        health.raise_for_status()
                        if health.json()['checkpoint'] != str(args.own_server_checkpoint.resolve()):
                            raise ValueError('Wrong server owns this port')
                        break
                    except requests.RequestException:
                        if time.monotonic() > until: raise TimeoutError('Own server did not become healthy')
                        time.sleep(3)
                main(args)
        else:
            main(args)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except __import__('subprocess').TimeoutExpired:
                child.kill(); child.wait(timeout=10)

