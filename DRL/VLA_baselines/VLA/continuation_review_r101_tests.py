"""Synthetic observer tests; these are never counted as robot episodes."""
from pathlib import Path
import tempfile
import unittest

from VLA.admission_r100.common import atomic_json
from VLA.continuation_review_r101 import trace_diagnosis


class ObserverTests(unittest.TestCase):
    def case(self, root, *, attachment, bilateral, diagnostic=None):
        raw = [0.] * 64
        raw[31] = float(attachment)
        raw[28] = float(bilateral)
        raw[29] = float(attachment)
        raw[17] = float(attachment and bilateral)
        if diagnostic is None:
            diagnostic = attachment
        sample = {'pre_action_scene64': raw, 'post_action_scene64': raw,
            'action': [0., 0., 0., 0., 0., 0., 1.], 'success': False,
            'before': {'correct_object_held': diagnostic},
            'after': {'correct_object_held': diagnostic}}
        trajectory = root / 'SYNTHETIC_NOT_ROBOT_EPISODE.json'
        atomic_json(trajectory, {'move': [sample]})
        return {'rows': [{'task': 'move', 'trajectory': str(trajectory),
            'steps': 1, 'success': False, 'safety_violation': None}]}

    def test_attachment_identity_is_not_bilateral_hold(self):
        with tempfile.TemporaryDirectory(prefix='r101_SYNTHETIC_') as tmp:
            progress = self.case(Path(tmp), attachment=True, bilateral=False)
            result = trace_diagnosis(progress)['move']
        self.assertEqual(result['failed_without_correct_hold'], 1)
        self.assertEqual(result['failed_still_correctly_attached_at_end'], 1)
        self.assertEqual(result['failed_still_correctly_held_at_end'], 0)

    def test_bilateral_failure_partition(self):
        with tempfile.TemporaryDirectory(prefix='r101_SYNTHETIC_') as tmp:
            progress = self.case(Path(tmp), attachment=True, bilateral=True)
            result = trace_diagnosis(progress)['move']
        self.assertEqual(result['failed_without_correct_hold'], 0)
        self.assertEqual(result['failed_still_correctly_held_at_end'], 1)

    def test_actual_attachment_disagreement_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix='r101_SYNTHETIC_') as tmp:
            progress = self.case(Path(tmp), attachment=False, bilateral=False, diagnostic=True)
            with self.assertRaises(ValueError):
                trace_diagnosis(progress)


if __name__ == '__main__':
    unittest.main()
