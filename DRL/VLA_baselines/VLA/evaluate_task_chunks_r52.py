"""Task-conditioned native execution lengths, NEVER geometric action repair.

Language task chooses a predeclared execution prefix. Every motor command is
still taken unchanged from that model's native neural output. Matched8 is a
development selection replay, NOT an independently estimated success rate.
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
from VLA.stable_native_r44.conditioning import require_contract

TASKS = ('touch', 'pick', 'move', 'pick_place')


def language_task(prompt):
    word = str(prompt).strip().lower().split()[0]
    mapping = {'touch': 'touch', 'pick': 'pick', 'move': 'move', 'put': 'pick_place'}
    if word not in mapping:
        raise ValueError('Unrecognized instruction task; no hidden geometric phase fallback')
    return mapping[word]


def choose_schedule(progresses, family):
    native = 30 if family == 'xvla' else 8
    checkpoint = None
    evidence = []
    for path in progresses:
        p = json.loads(Path(path).read_text())
        if p['family'] != family or not p.get('privileged67'):
            raise ValueError('Require same privileged model family')
        if checkpoint is not None and checkpoint != p['checkpoint']:
            raise ValueError('Do not pool execution prefixes from different weights')
        checkpoint = p['checkpoint']
        rows = [r for r in p['rows'] if r['seed'] in SEEDS[:2]]
        expected = {(seed, task) for seed in SEEDS[:2] for task in TASKS}
        if len(rows) != 8 or {(r['seed'], r['task']) for r in rows} != expected:
            raise ValueError('All identical matched8 required; no partial selection')
        prefix = p['native_execution_chunk']
        if not 4 <= prefix <= native:
            raise ValueError('Known lost-future-CLOSE prefix1 is forbidden')
        evidence.append({'prefix': prefix, 'rows': rows, 'progress': str(Path(path).resolve()),
                         'sha256': sha256_file(path)})
    if len({e['prefix'] for e in evidence}) < 3:
        raise ValueError('Need at least3 actual execution lengths per model')
    result = {}
    for task in TASKS:
        options = []
        for e in evidence:
            rows = [r for r in e['rows'] if r['task'] == task]
            options.append({'prefix': e['prefix'], 'successes': sum(bool(r['success']) for r in rows),
                'safety_aborts': sum(bool(r.get('safety_violation')) for r in rows), 'n': 2})
        best = max(options, key=lambda r: (r['successes'], -r['safety_aborts'], r['prefix'] == native, r['prefix']))
        result[task] = {'selected_prefix': best['prefix'], 'options': options}
    return {'checkpoint': checkpoint, 'family': family, 'tasks': result, 'evidence': evidence,
        'rule': 'per_task_most_success_then_fewer_aborts_then_native_then_longer_prefix',
        'selection_n_per_task': 2, 'selection_is_DEVELOPMENT_NOT_independent_SR': True,
        'schedule_uses_original_language_task_only_NOT_target_position_or_teacher_phase': True}


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    schedule_path = args.schedule.resolve()
    schedule = json.loads(schedule_path.read_text())
    session = requests.Session()
    url = f'http://127.0.0.1:{args.port}'
    health = session.get(url+'/health', timeout=20)
    health.raise_for_status()
    metadata = health.json()
    family, checkpoint = schedule['family'], schedule['checkpoint']
    if metadata['family'] != family or metadata['checkpoint'] != checkpoint:
        raise ValueError('Actual server does not own the selected exact weights')
    cp = Path(checkpoint)
    cfg = require_contract(cp)
    if metadata['actual_optimizer_updates'] < (7000 if family == 'xvla' else 6000):
        raise ValueError('Require completed own fit, not a two-step model')
    protocol = json.loads(PROFILE_PROTOCOL.read_text())
    profile = json.loads(Path(protocol['observation_profile_file']).read_text())
    if profile_hash(profile) != protocol['observation_profile_sha256']:
        raise ValueError('Shared robot/camera/home changed')
    data_proof = json.loads((Path(cfg['data_root'])/'audit.json').read_text())
    original = next(Path(p) for p in data_proof['bound_files'] if p.endswith('full_train64/audit.json'))
    original_proof = json.loads(original.read_text())
    source_manifest = Path(original_proof['source'])/'dataset_manifest.json'
    if sha256_file(source_manifest) != original_proof['source_manifest_sha256']:
        raise ValueError('Original train/deploy profile receipt changed')
    if profile_hash(json.loads(source_manifest.read_text())['observation_profile']) != profile_hash(profile):
        raise ValueError('Train/deploy camera or home mismatch')
    seeds = SEEDS[args.seed_start:args.seed_start+args.seed_count]
    if len(seeds) != args.seed_count:
        raise ValueError('No implicit seed replacement')
    atomic_json(out/'plan.json', {'family': family, 'checkpoint': checkpoint,
        'schedule': schedule, 'schedule_sha256': sha256_file(schedule_path), 'seeds': seeds,
        'scope': 'selected_matched8_replay_NOT_independent_SR' if args.seed_start == 0 else
                 'new_to_prefix_selection_DEVELOPMENT40_NOT_reserved_final_test',
        'same_robot_objects_physics_camera_home_success_and_horizons': True,
        'all_canonical_commands_from_genuine_neural_actions': True,
        'geometry_based_action_override_or_phase_machine': False,
        'language_conditioned_execution_prefix_only': True,
        'environment_fingerprint': environment_runtime_fingerprint(profile),
        'implementation_sha256': sha256_file(Path(__file__))})
    rows, queries, controls = [], 0, 0
    def progress(status, **kw):
        atomic_json(out/'progress.json', {'status': status, 'family': family, 'checkpoint': checkpoint,
            'completed_trials': len(rows), 'planned_trials': 4*len(seeds), 'rows': rows,
            'successes': sum(r['success'] for r in rows), 'queries': queries, 'neural_controls': controls,
            'privileged67': True, 'task_prefix': {t: schedule['tasks'][t]['selected_prefix'] for t in TASKS},
            'selection_matched8_not_independent_SR': args.seed_start == 0,
            'statistics': success_statistics(rows) if len(rows) and len(rows)%4 == 0 else None, **kw})
    progress('running')
    try:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tasks = [randomize_rl_scene(t, rng) for t in select_tasks('all')]
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
                    instruction_task = language_task(task.prompt)
                    if instruction_task != task.name:
                        raise ValueError('Language parse and task contract disagree')
                    prefix = schedule['tasks'][instruction_task]['selected_prefix']
                    obs = env.reset(task)
                    if env.is_semantic_success():
                        raise ValueError('Initial task already satisfied; no reroll')
                    _save_observation_snapshot(folder, task.name, obs, env)
                    queue, trace, done = deque(), [], False
                    while not done:
                        state = model_input(env, obs)
                        if not queue:
                            payload = {'instruction': obs.prompt, 'scene64': state['scene64'].tolist(),
                                'ee_position': obs.ee_position.tolist(), 'ee_rotation_6d': obs.ee_rotation_6d.tolist(),
                                'front_png': png(obs.front_rgb), 'wrist_png': png(obs.wrist_rgb)}
                            response = session.post(url+'/act', json=payload, timeout=900)
                            response.raise_for_status()
                            result = response.json()
                            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                            if result['actual_neural_model'] is not True or result['input_sha256'] != digest:
                                raise ValueError('Native input/model identity mismatch')
                            native = np.asarray(result['actions'], np.float32)
                            if native.shape != ((30, 20) if family == 'xvla' else (8, 7)) or not np.isfinite(native).all():
                                raise ValueError('Malformed neural output')
                            queue.extend(native[:prefix].copy())
                            query_step = int(obs.step)
                            queries += 1
                        native_action = queue.popleft()
                        if family == 'xvla':
                            action = canonical_from_absolute20(native_action, obs.ee_position, obs.ee_rotation_6d)
                            action[6] = native_gripper_command(native_action[9], convention='xvla_closed01')
                        else:
                            action = native_action.copy()
                            action[6] = native_gripper_command(action[6], convention='adapter_open01')
                        before = _interaction_audit(env)
                        obs, _, done, info = env.step(action)
                        controls += 1
                        trace.append({'task': task.name, 'step': obs.step, 'action': action.tolist(),
                            'native_neural_action': native_action.tolist(), 'before': before, 'after': _interaction_audit(env),
                            'pre_action_scene64': state['scene64'].tolist(), 'success': bool(info['success']),
                            'safety_violation': info.get('safety_violation'),
                            'safety_diagnostics_before_rollback': info.get('safety_diagnostics_before_rollback'),
                            'executed_canonical_action': info['executed_canonical_action'],
                            'chunk_query_step': query_step, 'action_index_in_chunk': int(obs.step)-query_step-1,
                            'neural_chunk_input_sha256': digest, 'task_execution_prefix': prefix})
                        traces[task.name] = trace
                        atomic_json(folder/'trajectory.json', traces)
                        if writer:
                            writer.append_data(np.concatenate((env.render('overview'), obs.front_rgb, obs.wrist_rgb), axis=1))
                    rows.append({'seed': seed, 'task': task.name, 'success': bool(info['success']), 'steps': len(trace),
                        'safety_violation': info.get('safety_violation'), 'trajectory': str(folder/'trajectory.json')})
                    progress('running')
            finally:
                if writer:
                    writer.close()
                env.close()
        progress('completed_selected_schedule_DEVELOPMENT_NOT_reserved_final_test')
    except Exception as error:
        progress('integration_error_preserve_partial_NO_fake_SR', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-start', type=int, choices=(0, 10), default=0)
    p.add_argument('--seed-count', type=int, choices=(2, 10), default=2)
    main(p.parse_args())
