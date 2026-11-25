"""Run the unchanged task-prefix engine with an explicit locked seed cohort."""
import argparse
import json
from pathlib import Path

from VLA.common.provenance import sha256_file


def main(args):
    import VLA.evaluate_task_chunks_r52 as runner
    plan = json.loads(args.cohort.read_text())
    if not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']):
        raise ValueError('Explicit unique frozen seeds required')
    # Process-local cohort injection. Original source, policy, actions,
    # controller and completed cohorts are not edited.
    runner.SEEDS = [None]*10 + plan['seeds']
    original_atomic = runner.atomic_json
    def atomic(path, value):
        if Path(path).name in ('plan.json', 'progress.json'):
            value = {**value, 'scope': plan['scope'], 'cohort': str(args.cohort.resolve()),
                'cohort_sha256': sha256_file(args.cohort), 'selection_matched8_not_independent_SR': False,
                'fixed_cohort_never_used_for_optimization': True}
        original_atomic(path, value)
    runner.atomic_json = atomic
    args.seed_start, args.seed_count = 10, len(plan['seeds'])
    runner.main(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    main(p.parse_args())
