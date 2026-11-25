import unittest
import json
import numpy as np
from VLA.flower_feedback_r75.scheduler import FeedbackQueue


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.zeros(64, np.float32)
        self.raw[11] = .3
        self.actions = np.arange(70, dtype=np.float32).reshape(10, 7)

    def queue(self, **kwargs):
        q = FeedbackQueue(**kwargs); q.observe(self.raw, 0); q.extend(self.actions)
        return q

    def test_far_unchanged_ten_native_actions(self):
        q = self.queue()
        for i in range(10):
            np.testing.assert_array_equal(q.popleft(), self.actions[i]); q.observe(self.raw, i+1)
        self.assertFalse(q)

    def test_near_only_first_two_unchanged(self):
        self.raw[11] = .1; q = self.queue()
        np.testing.assert_array_equal(q.popleft(), self.actions[0]); q.observe(self.raw, 1)
        self.assertEqual(len(q), 9)
        np.testing.assert_array_equal(q.popleft(), self.actions[1]); q.observe(self.raw, 2)
        self.assertFalse(q)

    def test_contact_change_discards_suffix_before_next_action(self):
        q = self.queue(); q.popleft(); self.raw[28] = 1; q.observe(self.raw, 1)
        self.assertFalse(q)

    def test_drop_change_discards_suffix(self):
        self.raw[31] = self.raw[17] = self.raw[28] = self.raw[29] = 1
        q = self.queue(); q.popleft(); self.raw[[17, 28, 29, 31]] = 0; q.observe(self.raw, 1)
        self.assertFalse(q)

    def test_near_held_transport_keeps_native_chunk(self):
        self.raw[11] = .01; self.raw[31] = 1; q = self.queue()
        for i in range(10):
            np.testing.assert_array_equal(q.popleft(), self.actions[i]); q.observe(self.raw, i+1)
        self.assertFalse(q)

    def test_static_control_matches_original_even_near_and_touching(self):
        self.raw[11] = .01; q = self.queue(enabled=False)
        for i in range(10):
            np.testing.assert_array_equal(q.popleft(), self.actions[i])
            self.raw[28] = i % 2; q.observe(self.raw, i+1)
        self.assertFalse(q)

    def test_new_query_resets_contact_reference(self):
        q = self.queue(); q.popleft(); self.raw[28] = 1; q.observe(self.raw, 1)
        q.extend(self.actions); q.popleft(); q.observe(self.raw, 2)
        self.assertEqual(len(q), 9)

    def test_repeated_observation_does_not_consume_budget(self):
        self.raw[11] = .01; q = self.queue(); q.popleft()
        for _ in range(3): q.observe(self.raw, 1)
        self.assertEqual(len(q), 9)

    def test_no_future_or_alias_modification(self):
        q = self.queue(); saved = self.actions.copy(); self.raw[11] = .01
        q.popleft(); q.observe(self.raw, 1); q.popleft(); q.observe(self.raw, 2)
        np.testing.assert_array_equal(self.actions, saved)
        self.assertEqual(float(q.query_observation[11]), float(np.float32(.3)))

    def test_reject_malformed_chunk(self):
        q = FeedbackQueue(); q.observe(self.raw, 0)
        with self.assertRaises(ValueError): q.extend(np.zeros((9, 7)))

    def test_actual_event_receipt_is_json_serializable(self):
        events=[];q=self.queue(log=events.append);q.popleft()
        self.raw[28]=1;q.observe(self.raw,1)
        self.assertEqual(json.loads(json.dumps(events))[0]['discarded_neural_actions'],9)


if __name__ == '__main__': unittest.main()
