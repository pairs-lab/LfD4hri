import unittest
import json
from VLA.audit_r33_neural_video import frame_mapping, transport_metadata, TASKS


class NativeVideoTests(unittest.TestCase):
    def sample(self):
        rows = [dict(seed=1, task=t, steps=i+2, success=False) for i,t in enumerate(TASKS)]
        traces = {r['task']: [dict(success=False)]*r['steps'] for r in rows}
        return rows, traces

    def test_no_legacy_intro_or_extra_terminal_frame(self):
        count, mapping = frame_mapping(*self.sample())
        self.assertEqual(count, 14)
        self.assertEqual([r['offset'] for r in mapping], [0, 2, 5, 9])
        self.assertEqual(mapping[-1]['samples'][-1], 13)

    def test_reject_partial_seed_and_changed_outcome(self):
        rows, traces = self.sample()
        with self.assertRaises(ValueError): frame_mapping(rows[:-1], traces)
        traces['pick'][-1] = dict(success=True)
        with self.assertRaisesRegex(ValueError, 'outcome'): frame_mapping(rows, traces)

    def test_unknown_stream_estimate_does_not_corrupt_JSON_or_override_actual_count(self):
        result = transport_metadata({'fps':10., 'duration':56., 'nframes':float('inf')}, 560)
        self.assertEqual(result['exact_decoded_frame_count'], 560)
        self.assertTrue(result['imageio_estimated_nframes_unknown'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__': unittest.main()
