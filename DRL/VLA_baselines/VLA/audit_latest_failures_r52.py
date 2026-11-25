"""Read-only failure evidence from completed development cohorts, no retuning."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.audit_feedback_support_r52 import analyze_rollout
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for path in args.progress:
        progress = json.loads(path.read_text())
        if progress['completed_trials'] != progress['planned_trials']:
            raise ValueError('Completed cohorts only')
        trace_cache, rows = {}, []
        for row in progress['rows']:
            source = Path(row['trajectory'])
            if source not in trace_cache:
                trace_cache[source] = json.loads(source.read_text())
            trace = trace_cache[source][row['task']]
            result, raw = analyze_rollout(row, trace)
            actions = np.asarray([f['action'] for f in trace])
            native = np.asarray([f['native_neural_action'] for f in trace])
            result.update(target_xy_error_min_m=float(np.linalg.norm(raw[:, 11:13], axis=1).min()),
                target_xy_error_final_m=float(np.linalg.norm(raw[-1, 11:13])),
                native_gripper_min_median_max=np.quantile(native[:, 9 if progress['family'] == 'xvla' else 6], [0, .5, 1]).tolist(),
                canonical_xyz_median_absolute_m=np.median(abs(actions[:, :3]), axis=0).tolist(),
                original_TRACE_sha256=sha256_file(source))
            rows.append(result)
        failed = [r for r in rows if not r['success']]
        record = {'family': progress['family'], 'checkpoint': progress['checkpoint'],
            'progress': str(path.resolve()), 'progress_sha256': sha256_file(path),
            'actual_successes': sum(r['success'] for r in rows), 'trials': len(rows),
            'failure_counts': dict(Counter(r['observed_category_NOT_model_intrinsic_cause'] for r in failed)),
            'by_task': {task: {'successes': sum(r['success'] for r in rows if r['task'] == task),
                'trials': sum(r['task'] == task for r in rows)} for task in ('touch', 'pick', 'move', 'pick_place')},
            'rows': rows, 'not_intrinsic_model_weakness_or_final_test_claim': True}
        reports.append(record)
    atomic_json(args.output/'audit.json', {'scope': 'completed_development_failure_analysis_NOT_new_neural_trials',
        'source_weights_data_controller_or_scores_changed': False, 'reports': reports})
    for r in reports:
        print(json.dumps({k: r[k] for k in ('family', 'actual_successes', 'trials', 'failure_counts', 'by_task')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
