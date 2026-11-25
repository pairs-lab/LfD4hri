"""Read-only TRAIN/support and actual rollout diagnostics; never a controller.

Kept outside frozen R69/R70 sources so an audit cannot change a live run.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.common import DATA
from VLA.scene64_conditioning import encode_numeric, NUMERIC_NAMES


def read(path):
    return json.loads(Path(path).read_text())


def quantiles(values):
    return None if not len(values) else np.quantile(values, [0, .1, .5, .9, 1]).tolist()


def first_step(flag):
    found = np.flatnonzero(flag)
    return None if not len(found) else int(found[0] + 1)


def grasp_support(raw, actions):
    """Use actual Schmitt trigger (>0.2 close, <-0.2 open), not +/-0.5."""
    near = (np.linalg.norm(raw[:, 11:13], axis=1) <= .025) & (np.abs(raw[:, 13]) <= .06)
    unheld = raw[:, 31] < .5
    contact = raw[:, 28:30].any(axis=1)
    close = actions[:, 6] > .2
    opening = actions[:, 6] < -.2
    return {
        'frames': len(raw),
        'near_target_unattached': int(np.sum(near & unheld)),
        'near_target_unattached_close': int(np.sum(near & unheld & close)),
        'near_target_unattached_open': int(np.sum(near & unheld & opening)),
        'contact_unattached': int(np.sum(contact & unheld)),
        'contact_unattached_close': int(np.sum(contact & unheld & close)),
        'contact_unattached_open': int(np.sum(contact & unheld & opening)),
        'lift_over5mm_while_contact_unattached': int(np.sum(contact & unheld & (actions[:, 2] > .005))),
    }


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    rows = read(DATA / 'index.json')
    stats = read(DATA / 'stats.json')['scene67']
    mean, std = (np.asarray(stats[k], np.float64) for k in ('mean', 'std'))
    arrays = defaultdict(list)
    supports = defaultdict(Counter)
    timings = defaultdict(lambda: defaultdict(list))
    for ei, row in enumerate(rows):
        with np.load(DATA / row['cache'], allow_pickle=False) as file:
            raw, commands = file['scene64'], file['canonical7']
        key = row['task']
        arrays[key].append(raw.copy())
        supports[key].update(grasp_support(raw, commands))
        close = np.flatnonzero((commands[:, 6] > .2) & (raw[:, 10] < .5))
        if len(close):
            at = int(close[0])
            timings[key]['first_close_pad_height_fraction'].append(float(raw[at, 38]))
            timings[key]['first_close_target_relative_z_m'].append(float(raw[at, 13]))
            if row['task'] != 'touch':
                held = np.flatnonzero((raw[at:, 31] > .5) & (raw[at:, 30] < .5))
                if len(held):
                    timings[key]['first_close_to_actual_attachment_steps'].append(int(held[0]))
        if (ei + 1) % 300 == 0:
            atomic_json(a.output / 'progress.json', {'TRAIN_episodes_read': ei + 1})
    train = {}
    for task, values in arrays.items():
        raw = np.concatenate(values)
        encoded = np.asarray([encode_numeric(x) for x in raw])
        train[task] = {'raw': raw, 'encoded': encoded}
    cases = []
    cohort = read(a.cohort / 'progress.json')
    for row in cohort['rows']:
        trace = read(row['trajectory'])[row['task']]
        raw = np.asarray([x['pre_action_scene64'] for x in trace], np.float32)
        post = np.asarray([x['post_action_scene64'] for x in trace], np.float32)
        commands = np.asarray([x['action'] for x in trace], np.float32)
        native = np.asarray([x['executed_canonical_action'] for x in trace], np.float32)
        encoded = np.asarray([encode_numeric(x) for x in raw])
        z = np.abs((encoded - mean) / (std + 1e-8))
        peak = z.max(0)
        names = np.argsort(peak)[-8:][::-1]
        training = train[row['task']]['encoded']
        low, high = training.min(0), training.max(0)
        # Numeric min/max support is not an image- or task-distribution proof.
        outside = (encoded < low - 1e-5) | (encoded > high + 1e-5)
        held = np.asarray([x['after']['correct_object_held'] for x in trace])
        evidence = {
            'task': row['task'], 'seed': row['seed'], 'success': row['success'],
            'steps': len(trace), 'requested_object': trace[0]['before']['requested_object'],
            'safety_violation': row['safety_violation'], 'review': row['review'],
            'minimum_target_pad_XY_distance_m': float(np.linalg.norm(post[:, 11:13], axis=1).min()),
            'minimum_target_pad_XYZ_distance_m': float(np.linalg.norm(post[:, 11:14], axis=1).min()),
            'first_command_close_step': first_step(commands[:, 6] > .2),
            'first_measured_80percent_closed_step': first_step(post[:, 6] > .8),
            'first_correct_attachment_step': first_step(held),
            'support_counts': grasp_support(raw, commands),
            'requested_executed_max_abs_difference': np.abs(commands-native).max(0).tolist(),
            'top_numeric_zscores': [
                {'name': NUMERIC_NAMES[int(i)], 'max_abs_z': float(peak[i]),
                 'step': int(z[:, i].argmax() + 1),
                 'outside_per_action_TRAIN_range_steps': int(outside[:, i].sum())}
                for i in names],
            'latency_and_grasp_sequence_require_individual_trace_review': True,
            'terminal10_actual_controls': [
                {'step': item['step'], 'delta_z_m': item['action'][2],
                 'command_signed_close': item['action'][6],
                 'measured_closure_after': item['post_action_scene64'][6],
                 'target_relative_z_after_m': item['post_action_scene64'][13],
                 'pad_height_fraction_after': item['post_action_scene64'][38],
                 'correct_hold_after': item['after']['correct_object_held']}
                for item in trace[-10:]],
        }
        cases.append(evidence)
    result = {
        'kind': 'read_only_diagnostics_NOT_future_action_or_controller',
        'completed_rollouts_at_snapshot': cohort['completed_trials'],
        'checkpoint': cohort['checkpoint'],
        'TRAIN_episodes': len(rows), 'TRAIN_near_grasp_support': dict(supports),
        'TRAIN_timing_quantiles_min_p10_median_p90_max': {
            task: {key: {'n': len(value), 'quantiles': quantiles(value)} for key, value in fields.items()}
            for task, fields in timings.items()},
        'cases': cases,
        'warnings': [
            'Teacher close-to-attachment timing is conditional on successful physical contact, not a guaranteed motor delay.',
            'Numeric zscores alone cannot establish a camera or model failure cause.',
            'Near XY does not prove a valid body grasp: height, orientation, bilateral force and jaw geometry matter.',
            'A safety clamp/rollback is retained in measured outcome, never bypassed to increase SR.',
        ],
    }
    atomic_json(a.output / 'GRASP_TIMING_AND_INPUT_SUPPORT.json', result)
    atomic_json(a.output / 'progress.json', {'completed': True, 'actual_rollouts_diagnosed': len(cases)})
    print(json.dumps({'completed': True, 'cases': len(cases), 'TRAIN_near_grasp_support': dict(supports)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
