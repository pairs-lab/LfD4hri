"""Temporary synthetic fixtures test the auditor; never robot experiment data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from VLA import jepa_quick40_r67 as target


class Quick40AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.r65 = self.root / 'R65'
        self.source = self.r65 / 'vla_jepa/development40'
        self.out = self.root / 'report'
        self.source.mkdir(parents=True)
        self.out.mkdir()
        self.cohort = self.root / 'cohort.json'
        self.write(self.cohort, {'seeds': list(range(1, 11)), 'role': 'development_selection'})
        self.checkpoint = str(self.r65 / 'vla_jepa/fit2500/step-002500')
        self.write(self.r65 / 'vla_jepa/fresh_reload_final.json', {'passed': True, 'actual_optimizer_updates': 2500})
        self.write(self.source / 'plan.json', {'cohort': str(self.cohort), 'prefix': 7})
        rows, journal = [], []
        for seed in range(1, 11):
            traces = {}
            for task in target.TASKS:
                digest = f'SYNTHETIC_TEST_ONLY_{seed}_{task}'
                traces[task] = [{'success': False, 'action': [0.] * 7,
                                 'neural_receipt': {'query_step': 0, 'input_sha256': digest}}]
                journal.append({'seed': seed, 'task': task, 'native_views': 3,
                    'GT_map_SHA256': 'SYNTHETIC_TEST_ONLY', 'input_sha256': digest,
                    'actual_native_input_receipt': {'actual_Qwen_image_grid_thw': [[1, 1, 1]] * 3,
                                                    'actual_DiT_state_dimensions': 67},
                    'actions': [[0.] * 7] * 7})
                rows.append({'seed': seed, 'task': task, 'steps': 1, 'success': False, 'safety_violation': None})
            self.write(self.source / f'test_seed{seed}/trajectory.json', traces)
        self.progress = {'status': 'completed_actual_cohort', 'planned_trials': 40,
            'role': 'development_selection', 'checkpoint': self.checkpoint,
            'rows': rows, 'native_queries': 40, 'motor_controls': 40}
        self.write(self.source / 'progress.json', self.progress)
        (self.source / 'NEURAL_QUERY_RECEIPTS.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in journal))
        self.patch_source = patch.object(target, 'SOURCE', self.source)
        self.patch_r65 = patch.object(target, 'R65', self.r65)
        self.patch_source.start()
        self.patch_r65.start()

    def tearDown(self):
        self.patch_source.stop()
        self.patch_r65.stop()
        self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_consistent_temporary_fixture_passes(self):
        target.audit(self.out)
        result = target.read(self.out / 'MEASURED_QUICK40.json')
        self.assertEqual(result['actual_episodes'], 40)
        self.assertTrue(result['real_neural_action_and_input_audit_passed'])
        self.assertTrue(all(r['episodes'] == 10 for r in result['by_task'].values()))

    def test_missing_episode_rejected(self):
        self.progress['rows'].pop()
        self.write(self.source / 'progress.json', self.progress)
        with self.assertRaisesRegex(ValueError, 'complete real 40'):
            target.audit(self.out)

    def test_different_action_rejected(self):
        path = self.source / 'test_seed1/trajectory.json'
        trace = target.read(path)
        trace['touch'][0]['action'][0] = 1.
        self.write(path, trace)
        with self.assertRaisesRegex(ValueError, 'not from actual JEPA'):
            target.audit(self.out)

    def test_gate_checkpoint_not_accepted_as_finetuned(self):
        self.progress['checkpoint'] = str(self.r65 / 'vla_jepa/gate2/step-000002')
        self.write(self.source / 'progress.json', self.progress)
        with self.assertRaisesRegex(ValueError, '2500-update'):
            target.audit(self.out)

    def test_wrong_cohort_role_rejected(self):
        self.progress['role'] = 'untouched_final'
        self.write(self.source / 'progress.json', self.progress)
        with self.assertRaisesRegex(ValueError, 'development cohort'):
            target.audit(self.out)

    def test_zero_of_ten_not_proof_of_zero_population_SR(self):
        lo, hi = target.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.)
        self.assertAlmostEqual(hi, .2775328, places=6)


if __name__ == '__main__':
    unittest.main()
