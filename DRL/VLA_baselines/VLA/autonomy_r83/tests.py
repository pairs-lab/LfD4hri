"""CPU regression tests plus independent final800 action-source audit."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from VLA.autonomy_r83.common import TASKS, DEV, CONF, FINAL, ready, score, metrics, verify
from VLA.autonomy_r83.runtime import binary_evo_gripper


def audit_final(folder, method, seeds):
    from VLA.common.provenance import sha256_file
    folder = Path(folder)
    p = json.loads((folder / 'progress.json').read_text())
    expected = {(seed, task) for seed in seeds for task in TASKS}
    keys = [(r['seed'], r['task']) for r in p['rows']]
    if (p['status'] != 'completed_actual_final800' or len(keys) != 800 or
            len(set(keys)) != 800 or set(keys) != expected):
        raise ValueError('Final800 coverage/identity mismatch')
    lookup = {}
    with (folder / 'NEURAL_QUERIES.jsonl').open() as journal:
        for line in journal:
            q = json.loads(line)
            key = (q['seed'], q['task'], q['index'])
            if key in lookup or key[:2] not in expected:
                raise ValueError('Duplicate/out-of-cohort neural query')
            if q['receipt']['geometric_action_residual_or_Oracle'] or q['receipt']['future_teacher_phase_or_labels']:
                raise ValueError('Invalid neural input/action provenance')
            lookup[key] = q
    controls = 0
    for row in p['rows']:
        if type(row['success']) is not bool:
            raise ValueError('Outcome must be an actual boolean')
        traces = json.loads(Path(row['trajectory']).read_text())
        trace = traces[row['task']]
        if len(trace) != row['steps'] or not trace or trace[-1]['success'] != row['success']:
            raise ValueError('Episode outcome disagrees with trajectory')
        prefix = method['prefixes'][row['task']]
        for index, step in enumerate(trace):
            query = lookup[(row['seed'], row['task'], index // prefix)]
            expected_action = np.asarray(query['actions'][index % prefix], np.float32)
            if (not np.array_equal(expected_action, np.asarray(step['action'], np.float32)) or
                    query['input_sha256'] != step['query_sha256']):
                raise ValueError('Action is not from the actual frozen native chunk')
            controls += 1
    per = metrics(p['rows'])
    if any(v['n'] != 200 for v in per.values()) or p['per_action'] != per:
        raise ValueError('Incorrect per-action denominator')
    if controls != p['motor_controls'] or len(lookup) != p['native_queries']:
        raise ValueError('Neural/control counters disagree')
    verify(method['code'])
    verify(method['weights_sha256'])
    return {'passed': True, 'all800_unique_actual_trials': True, 'per_action': per,
        'all_executed_actions_match_frozen_native_query_chunks': True,
        'motor_controls': controls, 'native_queries': len(lookup),
        'safety_failures_retained': sum(bool(r['safety_violation']) for r in p['rows']),
        'progress_sha256': sha256_file(folder / 'progress.json'), 'five_seeds_measured': False}


class Tests(unittest.TestCase):
    def rows(self, successes):
        return [{'task': task, 'seed': seed, 'success': seed < successes,
                 'safety_violation': None} for task in TASKS for seed in range(10)]

    def test_no_empty_or_partial_readiness(self):
        with self.assertRaises(ValueError):
            ready({'rows': []})
        self.assertFalse(ready({'rows': self.rows(0)}))

    def test_explicit_threshold(self):
        self.assertTrue(ready({'rows': self.rows(5)}))
        self.assertFalse(ready({'rows': self.rows(4)}))

    def test_worst_action_required(self):
        rows = self.rows(10)
        for row in rows:
            if row['task'] == 'pick':
                row['success'] = False
        self.assertFalse(ready({'rows': rows}))

    def test_safety_failures_stay_in_denominator(self):
        rows = self.rows(5)
        rows[0].update(success=False, safety_violation='ground_penetration')
        self.assertEqual(metrics(rows)['touch']['n'], 10)
        self.assertEqual(score({'rows': rows})[2], -1)

    def test_fixed_nonoverlapping_cohorts(self):
        self.assertEqual(len(FINAL), 200)
        self.assertFalse(set(DEV) & set(CONF) or set(DEV) & set(FINAL) or set(CONF) & set(FINAL))

    def test_evo_binary_matches_official_OPEN01(self):
        actions = np.zeros((5, 7), np.float32)
        actions[:, 0] = np.arange(5) / 100
        actions[:, 6] = [-1., -.01, 0., .01, 1.]
        fixed = binary_evo_gripper(actions)
        np.testing.assert_array_equal(fixed[:, :6], actions[:, :6])
        np.testing.assert_array_equal(fixed[:, 6], [-1., -1., 1., 1., 1.])

    def test_original_actions_not_mutated(self):
        actions = np.full((2, 7), .01, np.float32)
        before = actions.copy()
        binary_evo_gripper(actions)
        np.testing.assert_array_equal(actions, before)


if __name__ == '__main__':
    unittest.main()
