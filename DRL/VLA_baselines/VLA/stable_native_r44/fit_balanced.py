"""Native fitting with explicit unheld OPEN-to-CLOSE exposure, not phase input.

R43's20 TRAIN probes got0/6 first-close decisions right although approach
and verified transport gripper signs were correct14/14. Motion weighting
underexposes the near-stationary acquisition transition. Balance this using
TRAIN gold labels ONLY for sampling. No expert phase is given to the model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAMPLER = {'schema': 'task_target_source_balanced_acquisition_transition_r44_v1',
    'equal4_tasks': True, 'corrective_probability': .2, 'reset_probability': .1,
    'nonreset_grasp_strata': {'unheld_closing': .4, 'verified_held': .35, 'general': .25},
    'unheld_closing_first_transition_probability': .5,
    'gold_labels_used_only_for_offline_sampling_NOT_policy_inputs': True}


def frame_for(data, ei, rng):
    row = data.rows[ei]
    numeric = data.numeric(ei)
    action, raw = numeric['canonical7'], numeric['scene64']
    if rng.random() < .1:
        return 0
    if row['task'] != 'touch':
        choice = rng.random()
        if choice < .4:
            ids = np.flatnonzero((action[:, 6] > .5) & (raw[:, 17] < .5))
            if len(ids):
                return int(ids[0] if rng.random() < .5 else rng.choice(ids))
        elif choice < .75:
            ids = np.flatnonzero(raw[:, 17] > .5)
            if len(ids):
                return int(rng.choice(ids))
    weights = 1.+2*(np.linalg.norm(action[:, :3], axis=1) > .001)
    return int(rng.choice(len(action), p=weights/weights.sum()))


def draw(data, rng):
    task = rng.choice(sorted(data.groups))
    kind = 'fresh_corrective' if rng.random() < .2 else 'complete_original'
    choices = data.groups[task][kind]
    target = rng.choice(sorted(choices))
    ei = int(rng.choice(choices[target]))
    return ei, frame_for(data, ei, rng)


def main(args):
    import VLA.new_vlas.dataset as dataset
    import VLA.new_vlas.train as native
    from VLA.stable_native_r44.train import main as bounded_main
    dataset.NativeData.draw = draw
    original_anchors = native.gate_anchors
    def anchors(data):
        result = []
        for ei, fi in original_anchors(data):
            values = data.numeric(ei)
            if data.rows[ei]['task'] != 'touch':
                ids = np.flatnonzero((values['canonical7'][:, 6] > .5) & (values['scene64'][:, 17] < .5))
                if not len(ids): raise ValueError('Grasp gate must exercise real unheld closing labels')
                fi = int(ids[0])
            result.append((ei, fi))
        return result
    native.gate_anchors = anchors
    original_atomic = native.atomic_json
    def atomic(path, value):
        if Path(path).name in ('plan.json', 'config.json'):
            value = {**value, 'training_sampler': SAMPLER}
            if Path(path).name == 'plan.json': value['sampling'] = SAMPLER['schema']
        original_atomic(path, value)
    native.atomic_json = atomic
    bounded_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    parser.add_argument('--mode', choices=('gate', 'long'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--total-updates', type=int, default=7000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--reload-proof', type=Path)
    args = parser.parse_args()
    start = 0 if args.resume is None else json.loads((args.resume/'config.json').read_text())['actual_optimizer_updates']
    if args.mode == 'gate' and (args.updates-start != 2 or args.accumulation != 4):
        parser.error('Gate must execute2 real new updates /8 micros')
    main(args)
