import unittest
from collections import deque
from types import SimpleNamespace
from VLA.evaluate_smol_chunk2_r41 import SEEDS, paired_comparison
from VLA import evaluate_smol800_r39 as cohort


class PairedTests(unittest.TestCase):
    def fixture(self):
        return [{'seed': s, 'task': t, 'success': False, 'safety_violation': None}
                for s in SEEDS for t in cohort.TASKS]

    def test_locked40_and_never_promote_automatically(self):
        ref = self.fixture()
        new = [dict(r) for r in ref]
        new[0]['success'] = True
        p = paired_comparison(ref, new)
        self.assertEqual(p['paired_improvements'], 1)
        self.assertTrue(p['candidate_for_larger_DEVELOPMENT_validation_ONLY'])
        self.assertTrue(p['NO_automatic_promotion_or800_change'])

    def test_no_promotion_for_safety_regression_or_no_gain(self):
        ref = self.fixture()
        self.assertFalse(paired_comparison(ref, ref)['candidate_for_larger_DEVELOPMENT_validation_ONLY'])
        new = [dict(r) for r in ref]
        new[0]['success'] = True
        new[1]['safety_violation'] = 'contact'
        self.assertFalse(paired_comparison(ref, new)['candidate_for_larger_DEVELOPMENT_validation_ONLY'])

    def test_no_missing_reordered_duplicate_or800_rows(self):
        ref = self.fixture()
        for wrong in (ref[:-1], list(reversed(ref)), [ref[0]] * 40,
                      [{**r, 'seed': 20269001} for r in ref]):
            with self.assertRaises(ValueError):
                paired_comparison(ref, wrong)


class OfficialQueueTests(unittest.TestCase):
    def test_official_select_consumes_first_then_second_not_last(self):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        predictions = torch.arange(35).reshape(1, 5, 7).float()
        calls = []
        dummy = SimpleNamespace(config=SimpleNamespace(n_action_steps=2),
            _queues={'action': deque(maxlen=2)}, eval=lambda: None,
            _prepare_batch=lambda b: b, _rtc_enabled=lambda: False)
        dummy._check_get_actions_condition = lambda: not dummy._queues['action']

        def chunk(batch, noise):
            calls.append(noise)
            return predictions.clone()

        dummy._get_action_chunk = chunk
        first = SmolVLAPolicy.select_action(dummy, {}, noise=torch.ones(1))
        second = SmolVLAPolicy.select_action(dummy, {}, noise=None)
        self.assertTrue(torch.equal(first, predictions[:, 0]))
        self.assertTrue(torch.equal(second, predictions[:, 1]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(dummy._queues['action']), 0)


if __name__ == '__main__':
    unittest.main()
