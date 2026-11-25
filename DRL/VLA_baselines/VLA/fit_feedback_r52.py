"""Same native FP32 optimizer; explicit recovery rehearsal and save receipts."""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file

SAMPLER = {'schema': 'equal_task_50_original_15_existing_35_pose_recovery_PHASE_r52_v2',
    'corrective_start_first8_probability': .50, 'all_original_TRAIN_preserved': True,
    'otherwise_anchor_phase_probabilities': {'XY_over5mm': .45, 'gripper_transition': .20,
        'vertical_only_over5mm': .20, 'quiet_or_all_fallback': .15},
    'sampling_labels_and_phase_NEVER_added_to_policy_inputs': True,
    'teacher_phase_or_future_action_in_model_input': False}


def draw(data, rng):
    task = rng.choice(sorted(data.groups))
    probability = rng.random()
    kind = 'pose_corrective' if probability < .35 else 'fresh_corrective' if probability < .50 else 'complete_original'
    choices = data.groups[task][kind]
    target = rng.choice(sorted(choices))
    ei = int(rng.choice(choices[target]))
    raw, action = data.numeric(ei)['scene64'], data.numeric(ei)['canonical7']
    if kind == 'pose_corrective' and rng.random() < .50:
        fi = int(rng.integers(0, min(len(action), 8)))
    else:
        lateral = np.linalg.norm(action[:, :2], axis=1) > .005
        change = (action[:, 6] > .5) != (raw[:, 10] > .5)
        vertical = ~lateral & (abs(action[:, 2]) > .005)
        quiet = (np.linalg.norm(action[:, :3], axis=1) <= .001) & (np.max(abs(action[:, 3:6]), axis=1) <= .001)
        phase = rng.random()
        selected = lateral if phase < .45 else change if phase < .65 else vertical if phase < .85 else quiet
        ids = np.flatnonzero(selected)
        fi = int(rng.choice(ids)) if len(ids) else int(rng.integers(len(action)))
    return ei, fi


def gate_anchors(data):
    result = []
    for task in sorted(data.groups):
        for kind in ('complete_original', 'pose_corrective'):
            names = data.groups[task][kind]
            ei = names[sorted(names)[0]][0]
            result.append((ei, 0))
    return result


def main(args):
    import VLA.new_vlas.dataset as dataset
    import VLA.new_vlas.train as native
    from VLA.stable_native_r44.train import main as fit
    dataset.NativeData.draw = draw
    native.gate_anchors = gate_anchors
    # Disaggregate the actual Adapter TRAIN prediction, with no extra model
    # calls and no loss/controller alteration. A small aggregate L1 can hide
    # a near-zero XY policy when most labels are vertical or stationary.
    from collections import deque
    diagnostics = deque(maxlen=100)
    original_factory = native.create_runtime
    def factory(family, stats):
        runtime = original_factory(family, stats)
        if family == 'vla_adapter':
            from types import MethodType
            original_head, original_loss = runtime.head.predict_action, runtime.loss
            last = {}
            def head(this, features, **kwargs):
                value = original_head(features, **kwargs)
                if runtime.training_capture:
                    last['prediction'] = value.detach().float().cpu().numpy()[0]
                return value
            def loss(sample):
                value, metrics = original_loss(sample)
                prediction, target = last.pop('prediction'), sample['action']
                lo, hi = (np.asarray(stats['adapter']['action'][k]) for k in ('q01', 'q99'))
                pred_xy = .5*(prediction[:, :2]+1)*(hi[:2]-lo[:2])+lo[:2]
                true_xy = .5*(target[:, :2]+1)*(hi[:2]-lo[:2])+lo[:2]
                diagnostics.append({'normalized_XY_L1': float(abs(prediction[:, :2]-target[:, :2]).mean()),
                    'normalized_Z_L1': float(abs(prediction[:, 2]-target[:, 2]).mean()),
                    'normalized_gripper_L1': float(abs(prediction[:, 6]-target[:, 6]).mean()),
                    'mean_abs_predicted_XY_m': float(abs(pred_xy).mean()),
                    'mean_abs_supervised_XY_m': float(abs(true_xy).mean())})
                return value, metrics
            runtime.head.predict_action = MethodType(head, runtime.head)
            runtime.loss = loss
        return runtime
    native.create_runtime = factory
    if args.mode == 'gate':
        # CPU-only check of the ACTUAL sampler before optimizer admission.
        # No image bytes or development/test outcomes enter this selection.
        from collections import Counter
        from VLA.common.artifact_io import atomic_json
        data = dataset.NativeData(args.data, verify_rgb_hashes=False)
        rng, counts = np.random.default_rng(202652), Counter()
        for _ in range(4000):
            ei, fi = draw(data, rng)
            row, a = data.rows[ei], data.numeric(ei)
            command, state = a['canonical7'][fi], a['scene64'][fi]
            counts['task_'+row['task']] += 1
            counts['source_'+row['source_kind']] += 1
            counts['XY_over5mm'] += int(np.linalg.norm(command[:2]) > .005)
            counts['roll_pitch_correction'] += int(np.max(abs(command[3:5])) > 1e-5)
            counts['gripper_transition'] += int((command[6] > .5) != (state[10] > .5))
        if min(counts['task_'+t] for t in data.groups) < 700 or counts['XY_over5mm'] < 800:
            raise ValueError('Actual sampler failed task balance or minimum20percent XY support')
        atomic_json(args.output.with_suffix('.sampling_audit.json'), {'passed': True,
            'draws': 4000, 'actual_sampler': SAMPLER, 'counts': dict(counts),
            'CPU_ONLY_no_neural_success_claim': True})
    parent = json.loads((args.resume/'config.json').read_text())
    origin_updates = parent.get('r52_lineage', {}).get('parent_actual_optimizer_updates', parent['actual_optimizer_updates'])
    original_atomic = native.atomic_json
    def atomic(path, value):
        if Path(path).name == 'status.json' and diagnostics:
            value = {**value, 'r52_actual_TRAIN_diagnostics_last100_micros': {
                k: float(np.mean([r[k] for r in diagnostics])) for k in diagnostics[0]},
                'diagnostic_uses_existing_forward_only_NO_extra_model_calls': True}
        if Path(path).name in ('plan.json', 'config.json'):
            value = {**value, 'training_sampler': SAMPLER,
                'sampling': SAMPLER['schema'], 'r52_lineage': {
                    'parent_actual_optimizer_updates': origin_updates,
                    'actual_additional_updates_on_recovery_data': max(0, value.get('actual_optimizer_updates', origin_updates)-origin_updates),
                    'normalizer_recipe_changed_explicitly': True,
                    'old_checkpoint_weights_or_stats_NOT_modified': True}}
            if Path(path).name == 'config.json':
                if dataset.NativeData.draw is not draw:
                    raise ValueError('Actual recovery sampler identity changed')
                value['actual_sampler_function_identity_verified'] = True
                value['code'][str(Path(__file__).resolve())] = sha256_file(Path(__file__))
        original_atomic(path, value)
    native.atomic_json = atomic
    fit(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    p.add_argument('--mode', choices=('gate', 'long'), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--total-updates', type=int, required=True)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--reload-proof', type=Path)
    a = p.parse_args()
    before = json.loads((a.resume/'config.json').read_text())['actual_optimizer_updates']
    if a.mode == 'gate' and (a.updates-before != 2 or a.accumulation != 4):
        p.error('Gate must execute2 NEW actual updates and8 micros')
    main(a)
