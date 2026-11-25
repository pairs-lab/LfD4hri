"""Eight genuine neural controls in the ORIGINAL simulator. NOT a SR test.

Exercises the same HTTP protocol and native action decoder as development40,
using only the two-update preflight checkpoint. Never calls an oracle policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import imageio.v2 as imageio
import numpy as np
import requests

from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.new_vlas.action_contracts import canonical_from_absolute20
from VLA.new_vlas.evaluate import png
from VLA.new_vlas.policy_outputs import native_gripper_command
from VLA.run_cross_vla120 import PROFILE_PROTOCOL, SEEDS, gpu_state
from VLA.scene64_conditioning import model_input

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cp = args.checkpoint.resolve()
    cfg = json.loads((cp/'config.json').read_text())
    child, controls, traces, writer, env = None, 0, {}, None, None
    def save(status, **extra):
        atomic_json(out/'status.json', {'status': status, 'pid': os.getpid(), 'family': cfg['family'],
            'checkpoint': str(cp), 'neural_controls': controls, 'task_SR_established': False, **extra})
    try:
        save('waiting_exclusive_GPU')
        until = time.monotonic()+600
        while gpu_state()['compute_pids']:
            if time.monotonic() > until:
                raise TimeoutError('No exclusive GPU for actual neural robot smoke')
            time.sleep(3)
        python = ROOT/'VLA/.venvs'/('xvla' if cfg['family'] == 'xvla' else 'openvla')/'bin/python'
        with (out/'server.log').open('x') as log:
            child = subprocess.Popen([str(python), '-m', 'VLA.new_vlas.server', '--checkpoint', str(cp),
                '--port', str(args.port)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        save('loading_actual_native_checkpoint', server_pid=child.pid)
        url = f'http://127.0.0.1:{args.port}'
        session = requests.Session()
        until = time.monotonic()+900
        while True:
            if child.poll() is not None:
                raise RuntimeError('Native server exited: '+str(child.returncode))
            try:
                reply = session.get(url+'/health', timeout=3)
                reply.raise_for_status()
                meta = reply.json()
                if meta['checkpoint'] != str(cp):
                    raise ValueError('Wrong native server')
                atomic_json(out/'actual_server_metadata.json', meta)
                break
            except requests.RequestException:
                if time.monotonic() > until:
                    raise TimeoutError('Native server startup timeout')
                time.sleep(3)
        protocol = json.loads(PROFILE_PROTOCOL.read_text())
        profile = json.loads(Path(protocol['observation_profile_file']).read_text())
        env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e', image_size=384,
            control_hz=10, assist_grasp=True, seed=SEEDS[0], required_success_steps=1, use_semantic_success=True,
            success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
            semantic_success_thresholds=protocol['success']['semantic_thresholds'])
        rng = np.random.default_rng(SEEDS[0])
        tasks = [randomize_rl_scene(task, rng) for task in select_tasks('all')]
        writer = imageio.get_writer(str(out/'native_robot8_NOT_SR.mp4'), fps=10, codec='libx264', quality=8)
        save('actual_native_robot_controls')
        for task in tasks:
            obs = env.reset(task)
            trace = []
            for _ in range(2):
                raw = model_input(env, obs)['scene64']
                payload = {'instruction': obs.prompt, 'scene64': raw.tolist(),
                    'ee_position': obs.ee_position.tolist(), 'ee_rotation_6d': obs.ee_rotation_6d.tolist(),
                    'front_png': png(obs.front_rgb), 'wrist_png': png(obs.wrist_rgb)}
                reply = session.post(url+'/act', json=payload, timeout=900)
                reply.raise_for_status()
                result = reply.json()
                expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if result.get('actual_neural_model') is not True or result['input_sha256'] != expected:
                    raise ValueError('Wrong actual neural input/source')
                native = np.asarray(result['actions'], np.float32)
                if cfg['family'] == 'xvla':
                    if native.shape != (30, 20) or result['action_kind'] != 'native_absolute_ee20_closure':
                        raise ValueError('X-VLA native action contract differs')
                    action = canonical_from_absolute20(native[0], obs.ee_position, obs.ee_rotation_6d)
                    action[6] = native_gripper_command(native[0, 9], convention='xvla_closed01')
                else:
                    if native.shape != (8, 7) or result['action_kind'] != 'delta_ee7_open01':
                        raise ValueError('Adapter native action contract differs')
                    action = native[0].copy()
                    action[6] = native_gripper_command(action[6], convention='adapter_open01')
                obs, _, done, info = env.step(action)
                controls += 1
                trace.append({'step': obs.step, 'native_action': native[0].tolist(), 'canonical_action': action.tolist(),
                    'safety_violation': info.get('safety_violation'), 'input_sha256': expected,
                    'successful_task_NOT_claimed': True})
                traces[task.name] = trace
                atomic_json(out/'actual_controls.json', traces)
                writer.append_data(np.concatenate((env.render('overview'), obs.front_rgb, obs.wrist_rgb), axis=1))
                if info.get('safety_violation') or done:
                    raise ValueError('Robot smoke terminated early; preserve actual integration evidence')
        if controls != 8:
            raise ValueError('Actual native8 incomplete')
        atomic_json(out/'audit.json', {'passed': True, 'family': cfg['family'], 'neural_controls': controls,
            'all4_tasks': True, 'same_original_SIM_interpreter': True,
            'real_VLA_no_oracle_actions': True, 'same_shared_low_level_EEF_interface': True,
            'no_safety_aborts_in_these8_controls': True, 'task_SR_established': False,
            'scope': 'actual8_RPC_robot_integration_NOT_finetuned_policy_performance'})
        save('completed8_native_controls_NOT_SR')
    except Exception as error:
        save('failed_preserve_actual_evidence', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if writer:
            writer.close()
        if env:
            env.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    main(parser.parse_args())
