"""Measure target-domain supervision balance and the actual old fit exposure."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.fit_feedback_r52 import draw
from VLA.new_vlas.dataset import NativeData


def categories(actions, raw):
    return np.c_[np.linalg.norm(actions[:, :2], axis=1) > .005,
        abs(actions[:, 2]) > .005,
        (actions[:, 6] > .5) != (raw[:, 10] > .5),
        (raw[:, 6] > .8) & (raw[:, 31] < .5) & (actions[:, 6] < -.5),
        np.max(abs(actions[:, 3:5]), axis=1) > 1e-5]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = NativeData(args.data, verify_rgb_hashes=False)
    exposure = defaultdict(list)
    for line in args.exposure.read_text().splitlines():
        r = json.loads(line); exposure[r['episode']].append(r['frame'])
    total, seen, by_task = Counter(), Counter(), defaultdict(Counter)
    names = ['XY_over5mm', 'Z_over5mm', 'gripper_command_transition', 'measured_empty_close_reopen', 'roll_pitch_correction']
    for ei, row in enumerate(data.rows):
        a = data.numeric(ei)
        flags = categories(a['canonical7'], a['scene64'])
        for j, name in enumerate(names):
            total[name] += int(flags[:, j].sum())
            by_task[row['task']][name] += int(flags[:, j].sum())
            seen[name] += int(flags[exposure[ei], j].sum())
        total['frames'] += len(flags); by_task[row['task']]['frames'] += len(flags)
        seen['micros'] += len(exposure[ei])
    atomic_json(args.output/'audit.json', {'scope': 'TRAIN_support_and_actual_old_optimizer_exposure_NOT_new_SR',
        'TRAIN_counts': dict(total), 'by_task': dict(by_task), 'actual_exposure_counts': dict(seen),
        'source_exposure': str(args.exposure.resolve()),
        'categories_overlap_and_are_NOT_exclusive': True})
    print(json.dumps({'TRAIN_counts': dict(total), 'actual_exposure_counts': dict(seen)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--exposure', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
