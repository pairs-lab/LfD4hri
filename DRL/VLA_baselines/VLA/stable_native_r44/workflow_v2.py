"""Finite native-VLA input-stability repair, fit and matched evaluation.

Preserves R43 weights/results and all robot/scoring/source data. Bounded67 is
an observation preprocessing variant, not an oracle or action correction.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.stable_native_r44.conditioning import add_contract, contract

ROOT = Path(__file__).resolve().parents[2]
N = ROOT/'VLA/audits/new_vlas_r43_20261016'
DATA = N/'native_complete_data_v4_native_contracts'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
BASE = ROOT/'VLA/audits/new_vlas_quality_r44_20261016'


def read(path):
    return json.loads(Path(path).read_text())


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage, child, results = 'qualifying_bounded_observation_contract', None, []
    def status(state='running', **extra):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': state, 'stage': stage,
            'active_child_pid': None if child is None else child.pid,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results, **extra})
    def run(python, module, arguments, label, eval_output=None):
        nonlocal child, stage
        stage = label
        with (out/(label+'.log')).open('x') as log:
            child = subprocess.Popen([str(python), '-m', module, *arguments], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            status()
            while child.poll() is None:
                if eval_output is not None and (eval_output/'progress.json').exists():
                    from VLA.watch_new_vla_diagnostics_r44 import diagnose_trace
                    progress = read(eval_output/'progress.json')
                    diagnostic = []
                    for row in progress['rows']:
                        trace = read(row['trajectory'])[row['task']]
                        diagnostic.append({**row, **diagnose_trace(trace, row['task'])})
                    atomic_json(out/(label+'_diagnostics.json'), {'scope': 'overlapping_symptoms_NOT_root_causes',
                        'observation_conditioning': contract(), 'rows': diagnostic,
                        'actual_completed_trials': len(diagnostic), 'planned_trials': progress['planned_trials'],
                        'successes': sum(r['success'] for r in diagnostic),
                        'legacy_privileged_false_annotation_is_stale_actual_policy_receives67': True})
                time.sleep(5)
            code = child.returncode
        child = None
        if code:
            raise RuntimeError(f'{label} exited{code}; actual trials retained, no fake0%')
        status()
    def evaluate(cp, family, folder, label, count=10):
        run(SIM, 'VLA.stable_native_r44.evaluate', ['--checkpoint', str(cp), '--output', str(folder),
            '--port', '5845' if family == 'xvla' else '5846', '--seed-count', str(count)], label, folder)
        p = read(folder/'progress.json')
        if p['completed_trials'] != 4*count:
            raise ValueError('Incomplete declared cohort')
        results.append({'family': family, 'label': label, 'successes': p['successes'],
            'trials': p['completed_trials'], 'progress': str(folder/'progress.json')})
        status()
    try:
        status()
        old = ROOT/'VLA/xvla/privileged_r43_native_v3_20261016/test1'
        parent = old/'train/checkpoints/step-006000'
        if read(old/'train/status.json')['status'] != 'completed':
            raise ValueError('Do not interrupt an actual trainer')
        original = read(old/'development40/progress.json')
        atomic_json(out/'retired_old_pipeline.json', {'old_coordinator_pid': 1082837,
            'retired_for': 'near_constant_q4_and_tool_rotation_normalization_amplifies_small_rollout_pose_errors',
            'old_train6000_preserved': True, 'old_evaluation': original,
            'unrun_episodes_NOT_failures': 40-original['completed_trials'],
            'old_Adapter_fit_NOT_launched': True,
            'Smol800_and_historical_waiters_NOT_resumed': True})
        rows, stats = read(DATA/'index.json'), read(DATA/'stats.json')['scene67']
        mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
        clipped, affected, total = np.zeros(67, int), 0, 0
        for row in rows:
            with np.load(DATA/row['cache'], allow_pickle=False) as z:
                raw = z['scene64']
            angle = np.stack((np.sin(raw[:, 35:38]), np.cos(raw[:, 35:38])), -1).reshape(len(raw), 6)
            encoded = np.concatenate((raw[:, :35], angle, raw[:, 38:]), axis=1)
            mask = np.abs((encoded-mean)/(std+1e-8)) > 5
            clipped += mask.sum(0); affected += int(mask.any(1).sum()); total += len(raw)
        atomic_json(out/'normalization_audit.json', {'source_frames': total,
            'contract': contract(), 'frames_affected': affected, 'fraction_affected': affected/total,
            'clipped_feature_counts': clipped.tolist(), 'original_q4_std_rad': float(std[4]),
            'RGB_labels_native_prefix_and_stats_unchanged': True,
            'normalization_scope': 'ONLY_standardized67_tail_saturation_same_at_train_and_inference',
            'data_audit_sha256': sha256_file(DATA/'audit.json')})
        atomic_json(out/'plan.json', {'order': ['xvla_same6000_weights_paired_pilot8',
            'xvla2_new_step_gate_and_reload', 'xvla_continue_to7000', 'xvla_development40',
            'adapter2_step_gate_and_reload', 'adapter_finetune6000', 'adapter_development40'],
            'observation_conditioning': contract(), 'training_sampler': __import__('VLA.stable_native_r44.fit_balanced', fromlist=['SAMPLER']).SAMPLER,
            'same_robot_camera_home_and_scoring': True,
            'additional_XVLA_optimizer_updates': 1000, 'Adapter_optimizer_updates': 6000,
            'pilot_and_development_NOT_final_test': True, 'success_rate_guarantee': None,
            'baseline_partial_trials_retained': True, 'no_oracle_action_fallback': True})
        # An inference variant with exactly the original6000 master values.
        bundle = out/'xvla_same6000_weights_bounded_bundle'
        bundle.mkdir()
        for name in ('trainable_masters.pt', 'optimizer.pt', 'torch_rng.pt', 'reload_reference.npy'):
            os.symlink(parent/name, bundle/name)
        atomic_json(bundle/'config.json', read(parent/'config.json'))
        atomic_json(bundle/'saved_values.json', read(parent/'saved_values.json'))
        add_contract(bundle, parent=parent)
        xvla = ROOT/'VLA/xvla/privileged_r44_bounded67_20261016/test1'
        xvla.mkdir(parents=True, exist_ok=True)
        pilot = xvla/'pilot8_same6000_weights'
        if pilot.exists():
            stage = 'waiting_existing_same_weights_pilot8_NO_rerun'
            status()
            until = time.monotonic()+3600
            while True:
                p = read(pilot/'progress.json')
                if p['status'].startswith('completed8'):
                    results.append({'family': 'xvla', 'label': 'existing_same6000_weights_pilot8',
                        'successes': p['successes'], 'trials': p['completed_trials'], 'progress': str(pilot/'progress.json')})
                    break
                if p['status'].startswith('failed') or time.monotonic() > until:
                    raise ValueError('Existing pilot failed; preserve it, do not reroll')
                time.sleep(5)
            # Old owner is STOPPED at its wait(child). Retire only that exact
            # coordinator, not its simulator/server or any actual trainer.
            proc = Path('/proc/1096129/cmdline')
            if proc.exists() and b'VLA.stable_native_r44.workflow' in proc.read_bytes():
                import signal
                os.kill(1096129, signal.SIGTERM)
                os.kill(1096129, signal.SIGCONT)
        else:
            evaluate(bundle, 'xvla', pilot, 'xvla_same_weights_pilot8', count=2)
        for family, short, resume, gate_steps, final_steps in (
                ('xvla', 'xvla', parent, 6002, 7000),
                ('vla_adapter', 'adapter', None, 2, 6000)):
            python = ROOT/'VLA/.venvs'/('xvla' if family == 'xvla' else 'openvla')/'bin/python'
            model = ROOT/'VLA'/family/'privileged_r44_bounded67_20261016/test1'
            model.mkdir(parents=True, exist_ok=True)
            gate, reload = out/(short+'_bounded_gate'), out/(short+'_bounded_reload')
            arguments = ['--family', family, '--mode', 'gate', '--data', str(DATA), '--output', str(gate),
                '--updates', str(gate_steps), '--total-updates', str(final_steps), '--accumulation', '4',
                '--learning-rate', '.0001']
            if resume: arguments += ['--resume', str(resume)]
            run(python, 'VLA.stable_native_r44.fit_balanced', arguments, short+'_bounded_gate')
            cp = gate/'checkpoints'/f'step-{gate_steps:06d}'
            run(python, 'VLA.stable_native_r44.reload_gate', ['--checkpoint', str(cp),
                '--output', str(reload)], short+'_bounded_reload')
            run(python, 'VLA.stable_native_r44.train', ['--family', family, '--mode', 'long',
                '--data', str(DATA), '--output', str(model/'train'), '--updates', str(final_steps),
                '--total-updates', str(final_steps), '--accumulation', '4', '--learning-rate', '.0001',
                '--resume', str(cp), '--reload-proof', str(reload/'audit.json')], short+'_bounded_finetune')
            cp = Path(read(model/'train/latest_checkpoint.json')['checkpoint'])
            evaluate(cp, family, model/'development40', short+'_bounded_development40')
        stage = 'finished_bounded_native_adaptation_review_actual_results'
        status('completed')
    except Exception as error:
        status('failed_preserve_evidence_no_blind_retry', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.output.exists() or args.output.with_name(args.output.name+'_owner.json').exists():
            raise FileExistsError('Never duplicate a pipeline')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            process = subprocess.Popen([str(SIM), '-m', 'VLA.stable_native_r44.workflow_v2',
                '--output', str(args.output.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        atomic_json(args.output.with_name(args.output.name+'_owner.json'), {'pid': process.pid})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}))
    else:
        main(args)

