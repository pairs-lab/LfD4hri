"""Read-only trace evidence, saved outside the frozen baseline artifacts."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.flower_r70.common import read, ROOT


def main(output):
    baseline = ROOT/'VLA/audits/flower_postfit_r70_20261019/final_development40/progress.json'
    result = read(baseline); counts = Counter(); cases = []
    for row in result['rows']:
        if not row.get('safety_violation'): continue
        trace = read(row['trajectory'])[row['task']]; last = trace[-1]
        raw = np.asarray(last['pre_action_scene64']); diag = last['safety_diagnostics_before_rollback']
        start = min(t['step'] for t in trace if t['query_sha256'] == last['query_sha256'])
        flags = {'linkage_abort': 'linkage' in row['safety_violation'],
                 'requested_open': last['action'][6] < -.2,
                 'pre_action_target_pad_contact': bool(raw[28:30].any()),
                 'pre_action_target_force_over_100N': raw[39] > 100,
                 'pre_action_any_attachment': bool(raw[31]),
                 'pre_action_target_within_12cm': np.linalg.norm(raw[11:14]) <= .12,
                 'pre_rollback_ground_penetration': diag['surface_clearance_m'] < 0}
        counts.update({k: int(v) for k,v in flags.items()})
        cases.append({'seed': row['seed'], 'task': row['task'],
                      'abort_step': row['steps'], 'age_within_chunk': last['step']-start,
                      'pre_action_target_force_N': float(raw[39]),
                      'pre_rollback_surface_clearance_m': diag['surface_clearance_m'],
                      'observations': {k: bool(v) for k,v in flags.items()}, 'source': row['trajectory']})
    atomic_json(output, {'baseline': str(baseline), 'actual_baseline_episodes': result['completed_trials'],
        'baseline_per_action': result['per_action'], 'actual_safety_aborts': len(cases),
        'counts': dict(counts), 'cases': cases,
        'causality_caveat': 'Stale-feedback opportunities are observed, not proof that more queries prevent a failure.',
        'fix_scope': 'Native action suffix discard and fresh observation/query ONLY; no geometry-derived action, no guard relaxation.',
        'training_caveat': 'Existing TRAIN lacks certified true drop/wrong-grasp recovery examples. Neither resampling nor cadence alone adds them.'})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
