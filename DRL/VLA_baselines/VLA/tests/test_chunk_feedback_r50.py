import copy
import unittest

from VLA.run_chunk_feedback_r50 import TASKS, PAIRED_SEEDS, candidate, choose


def pilot(prefix=4, successes=3, aborts=0):
    return {'checkpoint': '/unused_test_checkpoint', 'native_execution_chunk': prefix, 'privileged67': True,
        'rows': [{'seed': seed, 'task': task, 'success': i < successes,
            'safety_violation': 'test-only abort' if successes <= i < successes+aborts else None}
            for i, (seed, task) in enumerate((s, t) for s in PAIRED_SEEDS for t in TASKS)]}


class ChunkFeedbackTest(unittest.TestCase):
    def qualify(self, p, prefix=4):
        return candidate(p, checkpoint='/unused_test_checkpoint', prefix=prefix, evidence='test_fixture_NOT_real_SR')

    def test_all8_real_pair_keys_required_no_partial_or_reroll(self):
        p = pilot(); p['rows'].pop()
        with self.assertRaises(ValueError): self.qualify(p)
        p = pilot(); p['rows'][-1] = copy.deepcopy(p['rows'][0])
        with self.assertRaises(ValueError): self.qualify(p)

    def test_different_checkpoint_not_an_execution_ablation(self):
        p = pilot(); p['checkpoint'] = '/different_test_checkpoint'
        with self.assertRaises(ValueError): self.qualify(p)

    def test_actual_prefix_and_privileged_contract_are_checked(self):
        with self.assertRaises(ValueError): self.qualify(pilot(6))
        p = pilot(); p['privileged67'] = False
        with self.assertRaises(ValueError): self.qualify(p)

    def test_prefix1_cannot_reintroduce_future_close_loss(self):
        with self.assertRaises(ValueError): self.qualify(pilot(1), prefix=1)

    def test_selection_cannot_trade_success_for_a_safe_looking_failure(self):
        a = self.qualify(pilot(4, successes=3, aborts=3))
        b = self.qualify(pilot(6, successes=2, aborts=0), prefix=6)
        self.assertEqual(choose([a, b], native=30)['execution_steps'], 4)

    def test_exact_ties_prefer_native_but_safety_breaks_success_tie(self):
        native = self.qualify(pilot(30, successes=3), prefix=30)
        short = self.qualify(pilot(4, successes=3))
        self.assertEqual(choose([native, short], native=30)['execution_steps'], 30)
        native['safety_aborts'] = 1
        self.assertEqual(choose([native, short], native=30)['execution_steps'], 4)

    def test_no_candidate_is_not_zero_success_rate(self):
        with self.assertRaises(ValueError): choose([], native=30)


if __name__ == '__main__':
    unittest.main()
