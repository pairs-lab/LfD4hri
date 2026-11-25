"""Bounded requested RDT Scene67 qualification and 10 episodes per action.

Never changes the fitted model, environment, success rule, or prior runs.
The existing Smol800 coordinator naturally waits between its seed workers
while this independently owned inference server occupies the GPU.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'VLA/audits/scene67_deadline_r31_20261015'
FAMILY = BASE / 'all_family_privileged_r33_v4'
CHECKPOINT = FAMILY / 'rdt_fit_2000/checkpoints/step-002000'
RDT = ROOT / 'VLA/.venvs/rdt/bin/python'
SIM = ROOT / 'VLA/third_party/lerobot/.venv/bin/python'
VISION = ROOT / 'VLA/checkpoints/siglip-so400m-patch14-384'
EMBEDDINGS = ROOT / 'VLA/audits/development40_rdt_r14_20261013/qualified_embeddings.pt'
SOURCE = ROOT / 'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'


def read(path):
    return json.loads(Path(path).read_text())


def resources():
    from VLA.run_cross_vla120 import gpu_state
    gpu = gpu_state()
    memory = {line.split(':')[0]: int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines()}
    return {**gpu, 'available_RAM_MiB': memory['MemAvailable'] // 1024}


def approved_smol_pids():
    """Recognize exact existing inference owners, never arbitrary GPU jobs."""
    pointer = read(BASE / 'Smol_800_r39_active.json')
    state = read(Path(pointer['cycle']) / 'status.json')
    result = set()
    for pid in (pointer['pid'], state.get('active_child_pid')):
        if not pid:
            continue
        try:
            command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            if b'VLA.resume_smol800_r40' in command:
                result.add(int(pid))
        except FileNotFoundError:
            pass
    return result


def serve(args):
    import torch
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(8192 * 2**20 / total)
    sys.argv = ['VLA.rdt.scene67_server', '--checkpoint', str(CHECKPOINT),
                '--vision', str(VISION), '--embeddings', str(EMBEDDINGS),
                '--output', str(args.output / 'server_runtime'), '--port', str(args.port)]
    runpy.run_module('VLA.rdt.scene67_server', run_name='__main__')


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = read(CHECKPOINT / 'pybullet_rdt_config.json')
    files = {str(Path(__file__).resolve()): sha256_file(__file__),
             **config['scene67']['bound_files']}
    for name in ('VLA/evaluate_privileged_r33.py', 'VLA/scene64_r33_client.py',
                 'VLA/rdt/scene67_server.py', 'VLA/common/environment.py',
                 'VLA/common/body_gripper.py', 'VLA/common/physics_safety.py',
                 'VLA/evaluation_protocol_v9_postfit_10_per_action.json'):
        files[str(ROOT / name)] = sha256_file(ROOT / name)
    plan = {'checkpoint': str(CHECKPOINT), 'actual_optimizer_updates': 2000,
            'planned_trials': 40, 'per_action': 10,
            'seeds': list(range(20268001, 20268011)),
            'privileged_inputs': 'native67_causal_scene_contact_goal_and_neighbors',
            'robot': 'ur5e', 'gripper': 'Robotiq2F85', 'replan_steps': 1,
            'high_level_actions_native_RDT_only': True, 'oracle_action_fallback': False,
            'same_locked_scene_success_safety': True, 'development_NOT_final_paper_test': True,
            'GPU_allocator_cap_MiB': 8192, 'frozen_files': files,
            'created_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(out / 'plan.json', plan)
    stage = 'checkpoint_and_native_contract_validation'
    history = []
    server = active = None
    deadline = time.monotonic() + 12 * 3600
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'MALLOC_ARENA_MAX': '2', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}

    def save(status='running', error=None):
        atomic_json(out / 'status.json', {
            'pid': os.getpid(), 'status': status, 'stage': stage,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'error': error,
            'checkpoint': str(CHECKPOINT), 'history': history,
            'owned_server_pid': server.pid if server and server.poll() is None else None,
            'active_child_pid': active.pid if active and active.poll() is None else None,
            'no_other_job_signalled': True, 'planned_trials': 40})

    def check():
        for path, digest in files.items():
            if sha256_file(path) != digest:
                raise ValueError('Fitted/frozen implementation changed: ' + path)
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded RDT40 budget expired; actual artifacts preserved')

    def execute(name, command, cpu=False):
        nonlocal active, stage
        stage = name
        check()
        with (out / (name + '.log')).open('x') as log:
            active = subprocess.Popen(list(map(str, command)), cwd=ROOT,
                env={**env, **({'CUDA_VISIBLE_DEVICES': ''} if cpu else {})},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(out / 'active_child.json', {'pid': active.pid, 'stage': name,
                'command': list(map(str, command)), 'CPU_only': cpu})
            while active.poll() is None:
                save()
                time.sleep(5)
            history.append({'stage': name, 'exit_code': active.returncode})
            save()
            if active.returncode:
                raise RuntimeError(name + ' failed; do not score infrastructure failure as zero SR')

    def report():
        progress = read(out / 'development40/progress.json')
        rows = progress['rows']
        counts = Counter(r['task'] for r in rows)
        if len(rows) != 40 or set(counts.values()) != {10}:
            raise ValueError('The requested 10 actual trials per action are not complete')
        if len({(r['seed'], r['task']) for r in rows}) != 40:
            raise ValueError('Duplicate or rerolled task')
        details = []
        for row in rows:
            trace = read(row['trajectory'])[row['task']]
            if len(trace) != row['steps'] or bool(trace[-1]['success']) != row['success']:
                raise ValueError('Terminal trajectory and outcome differ')
            feedback = [step['pre_action_scene64'] for step in trace]
            details.append({**row,
                'correct_attachment_observed': any(bool(s[17]) for s in feedback),
                'minimum_target_pad_distance_m': min(sum(v*v for v in s[11:14])**.5 for s in feedback),
                'minimum_object_goal_XY_distance_m': min(sum(v*v for v in s[14:16])**.5 for s in feedback),
                'full64_inputs_all_steps': all(len(s) == 64 for s in feedback)})
        atomic_json(out / 'episode_audit.json', {'rows': details,
            'actual_native_neural_controls': progress['neural_controls'],
            'full64_inputs_all_steps': all(r['full64_inputs_all_steps'] for r in details),
            'success_rule_unchanged': True, 'attachment_is_control_tick_observation_not_subtick_proof': True})
        lines = ['# RDT Scene67 requested development40', '',
                 'Checkpoint: `' + str(CHECKPOINT) + '`.', '',
                 '2,000 optimizer updates; FP32 weights/Adam and exact saved-value reload verified.',
                 'Native RDT actions, privileged causal67 conditioning; no oracle/planner fallback.',
                 '10 locked development scenes/action, unchanged robot/profile/scoring/safety.', '',
                 '| Action | Success | Rate | Safety aborts |',
                 '|---|---:|---:|---:|']
        for task in ('touch', 'pick', 'move', 'pick_place'):
            part = [r for r in rows if r['task'] == task]
            hits = sum(r['success'] for r in part)
            lines.append(f'| {task} | {hits}/10 | {100*hits/10:.1f}% | {sum(bool(r["safety_violation"]) for r in part)} |')
        lines += ['', f'Total: **{progress["successes"]}/40 = {100*progress["successes"]/40:.1f}%**.', '',
                  'Development diagnostic, not a final paper result or a demonstrated model ceiling.',
                  'Video: `development40/test_seed20268001/rollout.mp4`.',
                  'See `episode_audit.json` for grasp/localization/safety diagnostics.',
                  'Region success and observed attachment are separate; no success thresholds changed.']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    try:
        save()
        check()
        execute('saved_precision_2000', [RDT, '-m', 'VLA.rdt.audit_training_precision',
                '--checkpoint', CHECKPOINT, '--updates', '2000',
                '--output', out / 'saved_precision_2000.json'], cpu=True)
        execute('checkpoint_fingerprint', [RDT, '-m', 'VLA.fingerprint_checkpoint',
                CHECKPOINT, '--verify-existing'], cpu=True)
        stage = 'waiting_safe_GPU_and_RAM_admission'
        from VLA.wait_render_then_resume_queues import render_jobs
        while True:
            check()
            gpu = resources()
            approved = approved_smol_pids()
            atomic_json(out / 'resource_admission.json', {**gpu, 'allowed_Smol_inference_pids': sorted(approved)})
            if (set(map(int, gpu['compute_pids'])) <= approved and gpu['free_mib'] >= 9500
                    and gpu['available_RAM_MiB'] >= 4096 and not render_jobs()):
                break
            save()
            time.sleep(15)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        stage = 'loading_exact_RDT2000_with_8192MiB_allocator_cap'
        with (out / 'server.log').open('x') as log:
            server = subprocess.Popen([str(RDT), '-m', 'VLA.run_rdt40_scene67_r42',
                '--output', str(out), '--port', str(args.port), '--server-child'],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        until = time.monotonic() + 1800
        while True:
            save()
            if server.poll() is not None:
                raise RuntimeError('RDT saved checkpoint server failed to load; not a task failure')
            try:
                with urlopen(f'http://127.0.0.1:{args.port}/metadata', timeout=2) as reply:
                    metadata = json.load(reply)
                if metadata['checkpoint'] != str(CHECKPOINT) or not metadata['target_domain_finetuned']:
                    raise ValueError('Wrong server/checkpoint identity')
                break
            except (URLError, TimeoutError):
                if time.monotonic() > until:
                    raise TimeoutError('RDT startup exceeded 30 minutes')
                time.sleep(3)
        if not read(out / 'server_runtime/exact_reload.json')['all_saved_values_exact']:
            raise ValueError('RDT reload altered learned values')
        common = [SIM, '-m', 'VLA.evaluate_privileged_r33', '--model', 'rdt',
                  '--checkpoint', CHECKPOINT, '--port', str(args.port)]
        execute('actual_native8', common + ['--smoke-source', SOURCE, '--output', out / 'native8'])
        if read(out / 'native8/progress.json')['neural_controls'] != 8:
            raise ValueError('Own long-checkpoint neural controls incomplete')
        execute('development40', common + ['--trials-per-action', '10', '--output', out / 'development40'])
        check()
        report()
        stage = 'completed40_requested_actual_RDT_neural_episodes'
        save('completed40')
    except Exception as error:
        save('failed_preserve_actual_artifacts_NOT_zero_SR', f'{type(error).__name__}: {error}')
        raise
    finally:
        if active and active.poll() is None:
            while active.poll() is None:
                time.sleep(5)
        if server and server.poll() is None:
            server.terminate()
            server.wait(timeout=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5842)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--server-child', action='store_true')
    arguments = parser.parse_args()
    if arguments.server_child:
        serve(arguments)
    elif arguments.launch:
        marker = arguments.output.with_name(arguments.output.name + '_launch.json')
        if marker.exists() or arguments.output.exists():
            raise FileExistsError('Never launch a duplicate or overwrite prior trials')
        with arguments.output.with_name(arguments.output.name + '_launcher.log').open('x') as log:
            child = subprocess.Popen([str(SIM), '-m', 'VLA.run_rdt40_scene67_r42',
                '--output', str(arguments.output.resolve()), '--port', str(arguments.port)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        atomic_json(marker, {'pid': child.pid, 'output': str(arguments.output.resolve()), 'planned_trials': 40})
        print(json.dumps(read(marker)))
    else:
        main(arguments)
