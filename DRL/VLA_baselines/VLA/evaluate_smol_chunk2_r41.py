"""Locked development40 for native two-action SmolVLA chunk consumption.

Only the first TWO native predicted actions are consumed instead of one.
No learned value, model input, action projection, oracle, physics or scorer
changes. Never reads the800 outcomes. Cooperatively waits between800 seed
workers; no process is signalled. One disposable model/world process per
development seed keeps host memory bounded. A tiny coordinator reservation
prevents the original800/queued fits racing those gaps.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from VLA import evaluate_smol800_r39 as cohort
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.probe_smol_train_now_r40 import stable_window
from VLA.wait_render_then_resume_queues import identity, render_jobs

ROOT = cohort.ROOT
BASE = cohort.BASE
SIM = cohort.SIM
REFERENCE = BASE / 'scene67_resume_parallel_r34/parallel_attempt/development120/progress.json'
SEEDS = cohort.OLD_SEEDS[:10]


def paired_comparison(reference, candidate):
    expected = [(s, t) for s in SEEDS for t in cohort.TASKS]
    old = {(r['seed'], r['task']): r for r in reference}
    keys = [(r['seed'], r['task']) for r in candidate]
    if keys != expected or len(old) != 40 or set(old) != set(expected):
        raise ValueError('Exactly the same predeclared development40 required')
    wins = sum(r['success'] and not old[r['seed'], r['task']]['success'] for r in candidate)
    losses = sum(not r['success'] and old[r['seed'], r['task']]['success'] for r in candidate)
    previous_safety = sum(bool(r.get('safety_violation')) for r in reference)
    safety = sum(bool(r.get('safety_violation')) for r in candidate)
    return {'trials_each': 40, 'native1_successes': sum(r['success'] for r in reference),
            'native2_successes': sum(r['success'] for r in candidate),
            'paired_improvements': wins, 'paired_regressions': losses,
            'native1_safety_stops': previous_safety, 'native2_safety_stops': safety,
            'candidate_for_larger_DEVELOPMENT_validation_ONLY': wins > losses and safety <= previous_safety,
            'NO_automatic_promotion_or800_change': True, 'final800_used_for_selection': False}


def verify_plan(plan):
    for path, value in plan['frozen_file_identity'].items():
        s = Path(path).stat()
        if [s.st_ino, s.st_size, s.st_mtime_ns] != value:
            raise ValueError('Frozen checkpoint/runtime/source changed: ' + path)


def check_queued_action(policy, observation):
    """Audit the official queue and private noise; never alter its action."""
    import torch
    from unittest.mock import patch
    queue = policy.model._queues['action']
    count = len(queue)
    if queue.maxlen != 2:
        raise ValueError('Reset after setting n_action_steps=2 is required')
    expected = queue[0].clone() if count else None
    generator_before = policy._noise_generator.get_state().clone()
    select = policy.model.select_action
    checked = []

    def wrapped(batch, noise=None):
        raw = select(batch, noise=noise)
        if expected is not None:
            if noise is not None or not torch.equal(raw, expected):
                raise ValueError('Queued control is not the unmodified next native predicted action')
        elif noise is None:
            raise ValueError('A fresh native chunk requires private seeded noise')
        checked.append(True)
        return raw

    with patch.object(policy.model, 'select_action', side_effect=wrapped):
        action = policy.predict(observation)
    generator_after = policy._noise_generator.get_state()
    if len(checked) != 1 or len(queue) != (1 if not count else count - 1):
        raise ValueError('Unexpected native queue consumption')
    if torch.equal(generator_before, generator_after) != bool(count):
        raise ValueError('Private flow noise must advance ONLY for a new chunk')
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Malformed genuine VLA command')
    return action, {'new_native_chunk': not bool(count), 'queue_before': count, 'queue_after': len(queue),
                    'queued_value_and_private_noise_verified': True}


def run_seed(a):
    import torch
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.recorder import VideoRecorder, NullVideoRecorder
    from VLA.common.tasks import TaskSpec
    from VLA.benchmark import _interaction_audit, _save_observation_snapshot
    from VLA.simstate_scene64 import scene_state
    from VLA.smolvla.scene67_adapter import Scene67SmolVLA
    out = a.output.resolve()
    plan = json.loads((out / 'plan.json').read_text())
    verify_plan(plan)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3072 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    profile = json.loads(Path(plan['observation_profile_file']).read_text())
    env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e', image_size=384,
        control_hz=10, assist_grasp=True, seed=a.seed, required_success_steps=1, use_semantic_success=True,
        success_threshold_scale=plan['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=plan['success']['semantic_thresholds'])
    policy = Scene67SmolVLA(env, plan['checkpoint'], replan_steps=2, seed=a.seed)
    folder = out / f'test_seed{a.seed}'
    folder.mkdir(exist_ok=False)
    tasks = [r for r in plan['trials'] if r['seed'] == a.seed]
    traces = {}
    rows = []
    try:
        policy._load()
        policy.model.config.num_steps = 10
        if policy.model.config.chunk_size != 5:
            raise ValueError('The actual fitted checkpoint must predict five actions')
        # Check first-state semantics against the original native1 run before
        # evaluating this variant; no full reference episode is rerolled.
        reference = {(r['seed'], r['task']): r for r in plan['reference_rows']}
        record = VideoRecorder(folder / 'rollout.mp4', fps=10, overview_render=lambda: env.render('overview')) if a.seed == SEEDS[0] else NullVideoRecorder()
        with record:
            for item in tasks:
                task = TaskSpec.from_dict(item['task'])
                obs = env.reset(task)
                policy.reset(task)
                if env.is_semantic_success():
                    raise ValueError('Initially satisfied scene; no seed replacement')
                previous = json.loads(Path(reference[a.seed, task.name]['trajectory']).read_text())[task.name]
                np.testing.assert_allclose(scene_state(env, obs), previous[0]['pre_action_scene64'], rtol=0, atol=1e-6)
                _save_observation_snapshot(folder, task.name, obs, env)
                trace = []
                done = False
                with (folder / (task.name + '_applied_controls.jsonl')).open('x') as journal:
                    while not done:
                        before = _interaction_audit(env)
                        action, qa = check_queued_action(policy, obs)
                        if not trace:
                            # Same initial noise produces the same first action;
                            # subsequent differences are the declared queue2.
                            np.testing.assert_allclose(action, previous[0]['action'], rtol=0, atol=1e-6)
                        raw = policy.last_input['scene64'].copy()
                        obs, _, done, info = env.step(action)
                        frame = {'task': task.name, 'step': obs.step, 'action': action.tolist(),
                            'executed_canonical_action': info.get('executed_canonical_action'),
                            'action_source': 'native_neural_VLA_OFFICIAL_queue2_NO_action_repair',
                            'pre_action_scene64': raw.tolist(), 'native_queue_audit': qa,
                            'before': before, 'after': _interaction_audit(env),
                            'success': bool(info['success']), 'safety_violation': info.get('safety_violation'),
                            'safety_diagnostics_before_rollback': info.get('safety_diagnostics_before_rollback')}
                        trace.append(frame)
                        journal.write(json.dumps(frame, allow_nan=False) + '\n')
                        journal.flush()
                        record.add(obs, 'DEVELOPMENT SmolVLA native10 queue2 | ' + task.prompt)
                    os.fsync(journal.fileno())
                traces[task.name] = trace
                atomic_json(folder / 'trajectory.json', traces)
                rows.append({'seed': a.seed, 'task': task.name, 'success': bool(info['success']),
                    'steps': len(trace), 'safety_violation': info.get('safety_violation'),
                    'fresh_native_chunks': sum(f['native_queue_audit']['new_native_chunk'] for f in trace),
                    'trajectory': str(folder / 'trajectory.json')})
                atomic_json(folder / 'seed_result.json', {'status': 'running', 'rows': rows})
        verify_plan(plan)
        atomic_json(folder / 'seed_result.json', {'status': 'completed_seed', 'rows': rows,
            'same_initial_state_and_first_native_action_each_task_at1e6': True})
    finally:
        env.close()


def run(a):
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError('Fresh development root required; no outcomes overwritten')
    pointer = json.loads((BASE / 'Smol_800_r39_active.json').read_text())
    owner_root = Path(pointer['cycle'])
    # Only status/identity/locked plan are read. No800 result, image or action
    # is used in this development experiment or its promotion rule.
    owner_state = json.loads((owner_root / 'status.json').read_text())
    owner = identity(owner_state['pid'])
    lease = cohort.held_priority_lease()
    if (not owner or b'VLA.resume_smol800_r40' not in owner['tokens']
            or str(owner_root).encode() not in owner['tokens'] or not lease):
        raise ValueError('Exact process-isolated800 and existing owned resource lease required')
    original = json.loads((owner_root / 'plan.json').read_text())
    ref = json.loads(REFERENCE.read_text())
    reference = ref['rows'][:40]
    if ref['checkpoint'] != str(cohort.CHECKPOINT) or len(reference) != 40:
        raise ValueError('Same-weights native10 reference40 required')
    if set(SEEDS) & set(original['seeds']):
        raise ValueError('Development seeds must not overlap800')
    plan = {k: original[k] for k in ('checkpoint', 'success', 'observation_profile_file', 'observation_profile_sha256')}
    plan.update(variant='native10_native_chunk2', seeds=SEEDS, trials=cohort.trial_specs(SEEDS),
        reference_rows=reference, reference_file=str(REFERENCE), planned_trials=40,
        only_change='official n_action_steps=2, then native reset; trained chunk_size remains5',
        same_weights_inputs_actions_physics_success_horizons=True, final800_used_for_selection=False,
        no_oracle_or_action_projection=True, automatic_promotion=False,
        frozen_files=dict(original['frozen_files']), frozen_file_identity=dict(original['frozen_file_identity']))
    for path in (Path(__file__).resolve(), REFERENCE,
                 ROOT / 'VLA/third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py',
                 *sorted({Path(r['trajectory']) for r in reference})):
        plan['frozen_files'][str(path)] = sha256_file(path)
        st = path.stat()
        plan['frozen_file_identity'][str(path)] = [st.st_ino, st.st_size, st.st_mtime_ns]
    verify_plan(plan)
    out.mkdir(parents=True)
    atomic_json(out / 'plan.json', plan)
    rows = []
    stage = 'waiting_current800_seed_to_finish_NO_episode_signalled'
    error = None
    child = None
    deadline = time.monotonic() + a.wait_hours * 3600

    def save(status='running'):
        atomic_json(out / 'status.json', {'pid': os.getpid(), 'status': status, 'stage': stage, 'error': error,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_trials': len(rows),
            'planned_trials': 40, 'active_child_pid': child.pid if child and child.poll() is None else None,
            'other_process_signalled': False, 'final800_used_for_selection': False})
        atomic_json(out / 'progress.json', {'status': status, 'variant': 'native10_native_chunk2', 'rows': rows,
            'completed_trials': len(rows), 'planned_trials': 40, 'successes': sum(r['success'] for r in rows),
            'by_action': cohort.counts(rows), 'error': error, 'NOT_final_SR': True})

    save()
    try:
        import torch
        torch.set_num_threads(1)
        gpu = cohort.resources()
        if gpu['available_RAM_MiB'] < 2048 or gpu['free_mib'] < 512:
            raise ValueError('Insufficient even for a tiny gap reservation')
        torch.cuda.set_per_process_memory_fraction(64 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
        reservation = torch.empty(1, device='cuda')
        for seed in SEEDS:
            stage = 'waiting_safe_inter_seed800_window_and_initial_RAM'
            ready_since = None
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite development budget; keep partial outcomes')
                verify_plan(plan)
                actual_owner = identity(owner_state['pid'])
                current = cohort.held_priority_lease()
                if not actual_owner or actual_owner['start_ticks'] != owner['start_ticks'] or not current:
                    raise ValueError('Original800/priority lease ended; do not race a new trainer')
                state = json.loads((owner_root / 'status.json').read_text())
                worker = identity(state.get('active_child_pid') or 0)
                gpu = cohort.resources()
                expected = {os.getpid(), owner_state['pid'], current['trainer_pid']}
                if stable_window(state, worker, gpu, expected) and not render_jobs():
                    if ready_since is None:
                        ready_since = time.monotonic()
                    if time.monotonic() - ready_since >= 2:
                        break
                else:
                    ready_since = None
                save()
                time.sleep(2)
            atomic_json(out / f'seed{seed}_resource_admission.json', {'resources': gpu,
                'original800_seed_worker_absent': True, 'stable_inter_seed_window_seconds': 2,
                'own_worker_allocator_cap_MiB': 3072, 'existing_trainers_or_episodes_signalled': False})
            command = [str(SIM), '-m', 'VLA.evaluate_smol_chunk2_r41', '--output', str(out), '--seed', str(seed)]
            with (out / f'seed{seed}_worker.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'MALLOC_ARENA_MAX': '2',
                         'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                stage = 'actual_native_chunk2_development_seed'
                atomic_json(out / 'active_child.json', {'pid': child.pid, 'seed': seed, 'command': command})

                def harvest():
                    file = out / f'test_seed{seed}/seed_result.json'
                    if not file.exists():
                        return
                    data = json.loads(file.read_text())
                    for row in data['rows']:
                        if (row['seed'], row['task']) in {(r['seed'], r['task']) for r in rows}:
                            continue
                        expected = [(s, t) for s in SEEDS for t in cohort.TASKS]
                        if (row['seed'], row['task']) != expected[len(rows)]:
                            raise ValueError('No duplicate/omitted/reordered development outcomes')
                        trace = json.loads(Path(row['trajectory']).read_text())[row['task']]
                        if len(trace) != row['steps'] or bool(trace[-1]['success']) != row['success']:
                            raise ValueError('Terminal trajectory/result mismatch')
                        rows.append(row)
                        save()

                while child.poll() is None:
                    harvest()
                    save()
                    time.sleep(5)
                harvest()
                if child.returncode:
                    raise RuntimeError(f'Development seed{seed} exited{child.returncode}; NOT zero SR or rerolled')
                if len(rows) != 4 * (SEEDS.index(seed) + 1):
                    raise ValueError('Incomplete seed outcomes')
                child = None
        verify_plan(plan)
        atomic_json(out / 'matched_comparison.json', paired_comparison(reference, rows))
        stage = 'completed40_actual_native_chunk2_REQUIRES_review_not_promotion'
        save('completed')
        del reservation
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        save('failed_preserve_actual_trials_NOT_zero_fill')
        raise
    finally:
        # Never abandon/interrupt an executing seed or release its protected
        # resource gap while that own bounded worker remains resident.
        while child is not None and child.poll() is None:
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--wait-hours', type=float, default=3)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 4:
        raise ValueError('Finite development40 budget only')
    if args.seed is not None:
        run_seed(args)
    elif not args.launch:
        run(args)
    else:
        marker = args.output.with_name(args.output.name + '_launch.json')
        if args.output.exists() or marker.exists():
            raise FileExistsError('No duplicate development experiment')
        with args.output.with_name(args.output.name + '_launcher.log').open('x') as log:
            child = subprocess.Popen([str(SIM), '-m', 'VLA.evaluate_smol_chunk2_r41', '--output', str(args.output.resolve()),
                '--wait-hours', str(args.wait_hours)], cwd=ROOT,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2',
                     'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(marker, {'pid': child.pid, 'same_fixed_development40_NOT800': True,
                             'only_official_native_action_queue2': True})
        print(child.pid)
