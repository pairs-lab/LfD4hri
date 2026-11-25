import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.prepare_paired_grounding_followup import wait_for_complete


class PairedFollowupTest(unittest.TestCase):
    def timer(self):
        elapsed=[0.]
        def sleep(seconds): elapsed[0]+=seconds
        return lambda:elapsed[0],sleep

    def test_transient_partial_index_not_misclassified_as_bad_data(self):
        clock,sleep=self.timer()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'index.json').touch()
            reads=iter(['{"unfinished"', '[{"success":true,"safety_violation":null}]'])
            def read(path,*args,**kwargs):
                return next(reads) if path.name=='index.json' else '{}'
            with patch.object(Path,'read_text',read), patch.object(Path,'exists',lambda p:True), \
                 patch('VLA.prepare_paired_grounding_followup.validate_completed',return_value={'pick':38}) as completed:
                self.assertEqual(wait_for_complete(root,30,clock=clock,sleep=sleep),{'pick':38})
                self.assertEqual(clock(),10)
                completed.assert_called_once()

    def test_failed_episode_never_retried_or_promoted(self):
        clock,sleep=self.timer()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            # Test fixtures use framework helpers, not production collection.
            with patch.object(Path,'exists',lambda p:p.name=='index.json'), \
                 patch.object(Path,'read_text',return_value=json.dumps([{'success':False,'safety_violation':None}])):
                with self.assertRaisesRegex(ValueError,'Failed candidate'):
                    wait_for_complete(root,30,clock=clock,sleep=sleep)
                self.assertEqual(clock(),0)

    def test_collector_failure_with_no_index_stops_immediately(self):
        clock,sleep=self.timer()
        with patch.object(Path,'exists',lambda p:p.name=='collection_state.json'), \
             patch.object(Path,'read_text',return_value='{"status":"stopped_error_NOT_training","error":"no safe startup"}'):
            with self.assertRaisesRegex(ValueError,'no safe startup'):
                wait_for_complete('/unused',30,clock=clock,sleep=sleep)
            self.assertEqual(clock(),0)

    def test_permanent_partial_json_is_bounded_and_never_validated(self):
        clock,sleep=self.timer()
        with patch.object(Path,'exists',return_value=True), patch.object(Path,'read_text',return_value='{'), \
             patch('VLA.prepare_paired_grounding_followup.validate_completed') as complete:
            with self.assertRaises(TimeoutError):wait_for_complete('/unused',25,clock=clock,sleep=sleep)
            self.assertEqual(clock(),25)
            complete.assert_not_called()


if __name__=='__main__':unittest.main()
