"""Finite audited refinement for BOTH native VLAs, then locked120/model.

Retire only an identified IDLE same-data X20k queue after new data qualify.
Never interrupt Adapter training or R50 inference. No final-test tuning.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016'
R52 = ROOT/'VLA/audits/feedback_improvement_r52_20261016'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
TASKS = ('touch', 'pick', 'move', 'pick_place')


def read(path):
    return json.loads(Path(path).read_text())


def qualify_final_seeds(seeds):
    import numpy as np
    from VLA.audit_training_readiness import physical_scene_key
    from VLA.common.tasks import randomize_rl_scene, select_tasks
    from VLA.run_cross_vla120 import PROFILE_PROTOCOL, SEEDS
    protocol = read(PROFILE_PROTOCOL)
    reserved = read(ROOT/protocol['reserved_final_protocol'])['final_test_seeds']
    if set(seeds) & set([*SEEDS, *reserved, *range(20269001, 20269201)]):
        raise ValueError('New final cohort overlaps older/reserved/cancelled800 seed namespace')
    source = ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013'
    excluded = set()
    for name in ('qualified_train', 'validation_candidate'):
        excluded.update(physical_scene_key(r['task']) for r in read(source/name/'index.json'))
    cases = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for task in select_tasks('all'):
            task = randomize_rl_scene(task, rng)
            cases.append({'seed': seed, 'task': task.name, 'physical_scene_sha256': physical_scene_key(task.to_dict())})
    keys = [r['physical_scene_sha256'] for r in cases]
    if len(set(keys)) != len(keys) or set(keys) & excluded:
        raise ValueError('Final physical geometry overlaps TRAIN/validation or itself')
    return {'seeds': seeds, 'cases': cases, 'TRAIN_validation_geometry_overlap': 0,
            'old_or_reserved_seed_overlap': 0, 'neural_policies_or_outcomes_read': False}


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.replaces_idle_run:
        previous = read(args.replaces_idle_run/'status.json')
        failed_inactive = previous['status'] == 'failed_preserve_all_evidence_NO_blind_retry'
        if ((not failed_inactive and (previous['status'] != 'running'
                or previous['stage'] != 'waiting_R50_and_shared_server_pilots_NO_preemption'))
                or previous['active_child_pid'] is not None or previous['active_server_pid'] is not None):
            raise ValueError('Only own identified IDLE R52 queue can be replaced')
        proc = Path('/proc')/str(previous['pid'])
        atomic_json(out/'replaced_IDLE_R52_snapshot.json', previous)
        alive = proc.exists() and (proc/'stat').read_text().split(') ', 1)[1][0] != 'Z'
        if failed_inactive and alive:
            raise ValueError('Failed previous R52 must have exited; no live job preemption')
        if not failed_inactive:
            cmd = (proc/'cmdline').read_bytes()
            if b'VLA.run_feedback_refinement_r52' not in cmd or str(args.replaces_idle_run.resolve()).encode() not in cmd:
                raise ValueError('Idle R52 process identity mismatch')
            os.kill(previous['pid'], signal.SIGTERM)
            time.sleep(.5)
        if proc.exists() and (proc/'stat').read_text().split(') ', 1)[1][0] != 'Z':
            raise ValueError('Previous own queue did not exit')
        atomic_json(args.replaces_idle_run/'status.json', {**previous,
            'status': 'superseded_IDLE_by_multidirection_R52', 'replacement': str(out),
            'actual_training_interrupted': False})
    stage, child, server, results = 'waiting_complete_TRAIN_corrective_collection', None, None, []
    def status(state='running', **kw):
        atomic_json(out/'status.json', {'status': state, 'stage': stage, 'pid': os.getpid(),
            'active_child_pid': None if child is None else child.pid,
            'active_server_pid': None if server is None else server.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results, **kw})
    def run(python, module, options, label):
        nonlocal child
        with (out/(label+'.log')).open('x') as log:
            child = subprocess.Popen([str(python), '-u', '-m', module, *map(str, options)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
            status()
            while child.poll() is None:
                time.sleep(10); status()
            code = child.returncode
        child = None
        if code:
            raise RuntimeError(f'{label} integration/gate error exit{code}; evidence preserved, NOT a zero SR')
    def serve(family, cp, port, label):
        nonlocal server
        from VLA.new_vlas.train import wait_gpu
        wait_gpu(out)
        python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
        log = (out/(label+'_server.log')).open('x')
        server = subprocess.Popen([str(python), '-u', '-m', 'VLA.stable_native_r44.server',
            '--checkpoint', str(cp), '--port', str(port)], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '2',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        status()
        until = time.monotonic()+1200
        while True:
            if server.poll() is not None:
                raise RuntimeError('Own model server exited')
            try:
                health = requests.get(f'http://127.0.0.1:{port}/health', timeout=3)
                health.raise_for_status()
                if health.json()['checkpoint'] != str(cp.resolve()):
                    raise ValueError('Server checkpoint identity mismatch')
                break
            except requests.RequestException:
                if time.monotonic() > until:
                    raise TimeoutError('Own model not ready')
                time.sleep(3)
        return log
    def stop(log):
        nonlocal server
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)
        server = None
        log.close()
    def score(progress):
        p = read(progress)
        if p['completed_trials'] != p['planned_trials']:
            raise ValueError('Do not select on partial evaluation')
        return p['successes'], sum(bool(r.get('safety_violation')) for r in p['rows'])
    atomic_json(out/'plan.json', {
        'scope': 'BOTH_native_VLAs_normalizer_plus_actual_physics_recovery_refinement',
        'models': ['xvla', 'vla_adapter'], 'per_model_additional_actual_updates': 3000,
        'old_same_data_X20k_queue_replaced_ONLY_if_identified_idle_and_data_qualified': True,
        'source_candidates': str(args.candidate.resolve()), 'normalizer_floors_are_new_recipe_NOT_old_SR': True,
        'fit_gates': ['all_original_TRAIN_retained', 'new_RGB_state_action_physical_replay', 'native_FK_roundtrip',
            'new2_actual_Adam_updates', 'fresh_process_exact_reload', 'long3000_new_updates'],
        'old_and_new_variant_development': 'same10_UNSEEN_BY_PREFIX_SELECTION_seeds20268011..20268020',
        'new_variant_prefix_selection': 'matched8_development_for_4_6_and_official_native',
        'variant_selection': 'higher_complete40_success_then_fewer_abort_then_old_on_ties',
        'final_cohort': '120_per_model_30_per_action_NEW_seeds20269901..20269930',
        'final_outcomes_used_to_refit_or_retune': False,
        'all_actions_genuine_neural_VLA_not_geometric_override': True,
        'physics_robot_gripper_camera_success_horizon_unchanged': True,
        'guaranteed_success_rate': None, 'old_frozen_models_and_videos_preserved': True})
    atomic_json(out/'development_cohort.json', {'scope': 'same40_DEVELOPMENT_for_variant_selection_NOT_final_test',
        'seeds': list(range(20268011, 20268021))})
    atomic_json(out/'final_cohort.json', {'scope': 'LOCKED120_FINAL_NOT_used_for_training_or_tuning',
        'seeds': list(range(20269901, 20269931)), 'episodes_per_action': 30})
    atomic_json(out/'final_cohort_exclusion_audit.json', qualify_final_seeds(list(range(20269901, 20269931))))
    status()
    log = None
    try:
        until = time.monotonic()+4*3600
        while True:
            p = args.candidate/'status.json'
            if p.exists() and read(p)['status'].startswith('candidates_completed'):
                break
            if time.monotonic() > until:
                raise TimeoutError('TRAIN candidates not complete')
            time.sleep(10); status()
        data = out/'qualified_native_data'
        stage = 'qualifying_complete_TRAIN_bank_and_new_normalizers'; status()
        run(SIM, 'VLA.prepare_feedback_data_r52', ['--candidate', args.candidate, '--output', data], 'qualify_data')
        for family in ('xvla', 'vla_adapter'):
            stage = family+'_CPU_actual_native_image_and_state_input_audit'; status()
            python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
            run(python, 'VLA.audit_native_feedback_r52', ['--family', family, '--data', data,
                '--output', out/(family+'_input_audit')], family+'_input_audit')
        # R47 is still waiting for R50. Never terminate a live optimizer or
        # evaluator. If the ownership window changed, fail closed, not race it.
        old_root = BASE/'xvla_extended20k_r47'
        old = read(old_root/'status.json')
        inherited_retirement = (args.replaces_idle_run is not None
            and old['status'] == 'superseded_IDLE_by_R52_after_data_qualification'
            and Path(old['replacement']).resolve() == args.replaces_idle_run.resolve()
            and read(args.replaces_idle_run/'status.json').get('replacement') == str(out))
        if old['active_child_pid'] is not None or not old['stage'].startswith('waiting_'):
            raise ValueError('R47 no longer idle; no preemption or duplicate GPU trainer')
        r50_state = read(BASE/'chunk_feedback_r50/status.json')['status']
        if r50_state != 'running' and not (inherited_retirement and r50_state == 'completed'):
            raise ValueError('R50 priority window changed; re-audit before queue takeover')
        proc = Path('/proc')/str(old['pid'])
        atomic_json(out/'retired_IDLE_R47_snapshot.json', old)
        if inherited_retirement:
            if proc.exists() and (proc/'stat').read_text().split(') ', 1)[1][0] != 'Z':
                raise ValueError('Previously retired R47 unexpectedly alive')
        else:
            if not proc.exists() or b'VLA.run_xvla_exposure20k_r47' not in (proc/'cmdline').read_bytes():
                raise ValueError('Exact idle R47 PID identity required')
            os.kill(old['pid'], signal.SIGTERM)
            time.sleep(.5)
            if proc.exists() and (proc/'stat').read_text().split(') ', 1)[1][0] != 'Z':
                raise ValueError('Idle old owner did not exit; do not compete for GPU')
        atomic_json(old_root/'status.json', {**old, 'status': 'superseded_IDLE_by_R52_after_data_qualification',
            'replacement': str(out), 'actual_training_interrupted': False})
        stage = 'waiting_R50_and_shared_server_pilots_NO_preemption'; status()
        until = time.monotonic()+12*3600
        while True:
            r50 = read(BASE/'chunk_feedback_r50/status.json')
            pilot = R52/'task_schedule_pilots_v1/status.json'
            pilot_done = pilot.exists() and read(pilot)['stage'].startswith('finite_task_schedule_checks_completed')
            pilot_error = (R52/'task_schedule_pilots_v1/error.json').exists()
            if r50['status'].startswith('completed') and (pilot_done or pilot_error):
                break
            if time.monotonic() > until:
                raise TimeoutError('Existing evaluators not complete')
            time.sleep(15); status()
        from VLA.evaluate_task_chunks_r52 import choose_schedule
        for family, port in [('xvla', 5855), ('vla_adapter', 5856)]:
            python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
            model = out/family
            model.mkdir()
            parent = Path(read(ROOT/'VLA'/family/'privileged_r44_bounded67_20261016/test1/train/latest_checkpoint.json')['checkpoint'])
            if read(ROOT/'VLA'/family/'privileged_r44_bounded67_20261016/test1/train/status.json')['status'] != 'completed':
                raise ValueError('Own original fit not complete')
            old_schedule = R52/'task_schedule_pilots_v1'/(family+'_schedule.json')
            if not old_schedule.exists():
                selection = read(BASE/'chunk_feedback_r50'/(family+'_prefix_selection.json'))
                atomic_json(model/'old_schedule.json', choose_schedule([c['evidence'] for c in selection['candidates']], family))
                old_schedule = model/'old_schedule.json'
            stage = family+'_old_schedule_independent_development40'; status()
            old_dev = model/'old_development40'
            prior_dev = None if args.replaces_idle_run is None else args.replaces_idle_run/family/'old_development40'
            if prior_dev is not None and (prior_dev/'progress.json').exists():
                prior, prior_plan = read(prior_dev/'progress.json'), read(prior_dev/'plan.json')
                if (prior['checkpoint'] != str(parent) or prior['completed_trials'] != 40
                        or prior_plan['schedule_sha256'] != sha256_file(old_schedule)
                        or read(prior['cohort'])['seeds'] != read(out/'development_cohort.json')['seeds']):
                    raise ValueError('Previous development40 is not the exact same completed method/cohort')
                old_dev = prior_dev
                atomic_json(model/'reused_completed_original_development40.json', {
                    'source': str(prior_dev), 'progress_sha256': sha256_file(prior_dev/'progress.json'),
                    'exact_same_weights_schedule_and_cohort_verified': True, 'rerun_or_additional_SR_trials': False})
            else:
                log = serve(family, parent, port, family+'_old')
                run(SIM, 'VLA.evaluate_frozen_cohort_r52', ['--schedule', old_schedule, '--port', port,
                    '--output', old_dev, '--cohort', out/'development_cohort.json'], family+'_old40')
                stop(log); log = None
            warm = model/'WARM_START_ONLY_NOT_SERVE'
            stage = family+'_explicit_normalizer_warm_migration'; status()
            run(SIM, 'VLA.migrate_feedback_checkpoint_r52', ['--parent', parent, '--data', data, '--output', warm], family+'_warm')
            origin = read(warm/'config.json')['actual_optimizer_updates']
            target = origin+3000
            common = ['--family', family, '--data', data, '--total-updates', target]
            gate = model/'gate2'
            stage = family+'_new_actual_gate2'; status()
            run(python, 'VLA.fit_feedback_r52', [*common, '--mode', 'gate', '--updates', origin+2,
                '--resume', warm, '--output', gate], family+'_gate2')
            cp = Path(read(gate/'latest_checkpoint.json')['checkpoint'])
            reload = model/'fresh_reload2'
            stage = family+'_fresh_reload_gate'; status()
            run(python, 'VLA.stable_native_r44.reload_gate', ['--checkpoint', cp, '--output', reload], family+'_reload2')
            fit = model/'fit3000'
            stage = family+'_actual3000_additional_updates'; status()
            run(python, 'VLA.fit_feedback_r52', [*common, '--mode', 'long', '--updates', target,
                '--resume', cp, '--reload-proof', reload/'audit.json', '--output', fit], family+'_fit3000')
            fitted = Path(read(fit/'latest_checkpoint.json')['checkpoint'])
            stage = family+'_new_native_prefix_development_sweep'; status()
            log = serve(family, fitted, port, family+'_new')
            paths = []
            for prefix in (4, 6, 30 if family == 'xvla' else 8):
                folder = model/f'new_prefix{prefix}_pilot8'
                run(SIM, 'VLA.evaluate_native_chunks_r45', ['--family', family, '--port', port,
                    '--execution-steps', prefix, '--seed-count', 2, '--output', folder], family+f'_new_prefix{prefix}')
                paths.append(folder/'progress.json')
            schedule = model/'new_schedule.json'
            atomic_json(schedule, choose_schedule(paths, family))
            new_dev = model/'new_development40'
            stage = family+'_new_schedule_independent_development40'; status()
            run(SIM, 'VLA.evaluate_frozen_cohort_r52', ['--schedule', schedule, '--port', port,
                '--output', new_dev, '--cohort', out/'development_cohort.json'], family+'_new40')
            new_score, old_score = score(new_dev/'progress.json'), score(old_dev/'progress.json')
            use_new = (new_score[0], -new_score[1]) > (old_score[0], -old_score[1])
            selected = schedule if use_new else old_schedule
            selected_cp = fitted if use_new else parent
            atomic_json(model/'frozen_final_method.json', {'selected': 'new_recovery3000' if use_new else 'old_preserved',
                'old_development40_successes_and_aborts': old_score, 'new_development40_successes_and_aborts': new_score,
                'checkpoint': str(selected_cp), 'schedule': str(selected), 'schedule_sha256': sha256_file(selected),
                'selection_used_final_cohort': False, 'rule': 'more_complete40_success_then_fewer_abort_then_old_on_ties'})
            if not use_new:
                stop(log); log = serve(family, parent, port, family+'_selected_old')
            stage = family+'_LOCKED_final120_NO_more_tuning'; status()
            final = model/'final120'
            run(SIM, 'VLA.evaluate_frozen_cohort_r52', ['--schedule', selected, '--port', port,
                '--output', final, '--cohort', out/'final_cohort.json'], family+'_final120')
            stop(log); log = None
            final_score = score(final/'progress.json')
            results.append({'family': family, 'selected_new_recipe': use_new, 'final_successes': final_score[0],
                'final_trials': 120, 'progress': str(final/'progress.json'), 'success_over50_percent': final_score[0] > 60})
            status()
        stage = 'finite_refinement_and_locked_evaluation_completed'; status('completed')
    except Exception as error:
        status('failed_preserve_all_evidence_NO_blind_retry', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if server is not None and log is not None:
            stop(log)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--replaces-idle-run', type=Path,
        help='Exact own R52 output that must still be waiting without child/server')
    a = p.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.with_suffix('.log').open('x') as log:
            command = [sys.executable, '-u', '-m', 'VLA.run_feedback_refinement_r52',
                '--candidate', str(a.candidate.resolve()), '--output', str(a.output.resolve())]
            if a.replaces_idle_run:
                command += ['--replaces-idle-run', str(a.replaces_idle_run.resolve())]
            child = subprocess.Popen(command,
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'owner_pid': child.pid, 'output': str(a.output.resolve())}), flush=True)
    else:
        main(a)
