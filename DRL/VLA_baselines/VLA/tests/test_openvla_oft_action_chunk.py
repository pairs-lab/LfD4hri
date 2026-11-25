from collections import deque
import math
import unittest

from VLA.openvla_oft.action_chunk import consume_chunk_prefix


class OpenVLAOFTChunkPrefixTest(unittest.TestCase):
    def chunk(self):
        return [[step + channel / 10 for channel in range(7)] for step in range(8)]

    def test_first_one_of_eight_not_bounded_deque_last(self):
        chunk = self.chunk()
        upstream_queue = deque(maxlen=1)
        upstream_queue.extend(chunk)
        self.assertEqual(upstream_queue[0], chunk[-1])
        self.assertEqual(consume_chunk_prefix(chunk, 1), [chunk[0]])
        self.assertNotEqual(consume_chunk_prefix(chunk, 1)[0], upstream_queue[0])

    def test_ordered_prefix_and_full_horizon(self):
        chunk = self.chunk()
        self.assertEqual(consume_chunk_prefix(chunk, 3), chunk[:3])
        self.assertEqual(consume_chunk_prefix(chunk, 8), chunk)

    def test_returns_independent_copy(self):
        chunk = self.chunk()
        result = consume_chunk_prefix(chunk, 1)
        result[0][0] = 999
        self.assertEqual(chunk[0][0], 0)

    def test_no_gripper_inversion_scale_or_clipping(self):
        chunk = [[.01, -.02, .03, -.04, .05, -.06, -1],
                 [.01, -.02, .03, -.04, .05, -.06, 1]]
        self.assertEqual(consume_chunk_prefix(chunk, 2), chunk)

    def test_replan_bounds_and_integer_type(self):
        for value in (0, -1, 9, 1.0, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                consume_chunk_prefix(self.chunk(), value)

    def test_empty_and_non_chunk_rejected(self):
        for value in ([], 4, None, "actions"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                consume_chunk_prefix(value, 1)

    def test_ragged_or_non_vector_action_rejected(self):
        for row in ([0] * 6, [0] * 8, 4):
            with self.subTest(row=row), self.assertRaises(ValueError):
                consume_chunk_prefix([row], 1)

    def test_nonfinite_and_non_numeric_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, "0.1", 1j):
            chunk = self.chunk()
            chunk[0][3] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                consume_chunk_prefix(chunk, 1)

    def test_invalid_unused_future_action_is_rejected(self):
        chunk = self.chunk()
        chunk[-1][-1] = math.nan
        with self.assertRaises(ValueError):
            consume_chunk_prefix(chunk, 1)

    def test_float_overflow_rejected_as_invalid_action(self):
        chunk = self.chunk()
        chunk[0][0] = 10 ** 1000
        with self.assertRaises(ValueError):
            consume_chunk_prefix(chunk, 1)

    def test_iterable_input_keeps_order(self):
        chunk = self.chunk()
        result = consume_chunk_prefix((tuple(row) for row in chunk), 2)
        self.assertEqual(result, chunk[:2])


if __name__ == "__main__":
    unittest.main()
