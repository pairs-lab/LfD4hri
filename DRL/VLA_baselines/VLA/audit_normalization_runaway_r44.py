"""Read-only measured rollout OOD telemetry; correlation is not causation."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.audit_new_vla_quality_r44 import DATA, ROOT, read
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import encode_numeric, NUMERIC_NAMES


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = ROOT/'VLA/xvla/privileged_r43_native_v3_20261016/test1/development40'
    progress = read(root/'progress.json')
    stats = read(DATA/'stats.json')['scene67']
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    rows = []
    for episode in progress['rows']:
        trace = read(episode['trajectory'])[episode['task']]
        values = np.asarray([(encode_numeric(r['pre_action_scene64'])-mean)/(std+1e-8) for r in trace])
        action = np.asarray([r['action'] for r in trace])
        worst = np.abs(values).max(0)
        ids = np.argsort(worst)[-5:][::-1]
        early = [{'step': r['step'], 'q4_rad': r['pre_action_scene64'][4],
                  'q4_standardized': float(values[i, 4]), 'native_action_as_canonical': r['action']}
                 for i, r in enumerate(trace[:6])]
        rows.append({**episode, 'first6_controls': early,
            'q4_max_abs_standardized': float(worst[4]),
            'top_standardized_features': {NUMERIC_NAMES[i]: float(worst[i]) for i in ids},
            'commands_translation_over_common5cm_bound': int((np.abs(action[:, :3]) > .05).any(1).sum()),
            'common_environment_clipping_still_applied': True})
    atomic_json(args.output/'normalization_runaway_evidence.json', {
        'scope': 'actual_baseline_rollout_out_of_TRAIN_state_amplification_NOT_model_limit',
        'actual_completed_trials': progress['completed_trials'], 'successes': progress['successes'],
        'planned_but_unrun_NOT_failures': 40-progress['completed_trials'],
        'q4_TRAIN_mean_rad': float(mean[4]), 'q4_TRAIN_std_rad': float(std[4]),
        'q4_TRAIN_min_max_rad': [stats['min'][4], stats['max'][4]], 'rows': rows,
        'same_weights_bounded_observation_ablation_needed_to_assess_causality': True,
        'RGB_robot_home_scoring_safety_not_changed': True,
        'bound_files': {str(p): sha256_file(p) for p in (Path(__file__), DATA/'stats.json', root/'progress.json')}})
    print(json.dumps({'trials': len(rows), 'successes': progress['successes'],
        'max_q4_standardized': max(r['q4_max_abs_standardized'] for r in rows)}))
