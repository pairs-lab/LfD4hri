import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from VLA.audit_v9_data_support import feedback_counts
from VLA.resume_repaired_train_fit import ready_for_reviewed_continuation


class ReviewedContinuationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def state(self, status):
        (self.root / "workflow_status.json").write_text(json.dumps({"status": status, "candidate": "/not-executed-candidate"}))

    def test_active_preparation_cannot_launch(self):
        self.state("running")
        self.assertFalse(ready_for_reviewed_continuation(self.root))

    def test_unreviewed_source_cannot_launch(self):
        self.state("awaiting_actual_final_source_visual_review_no_training")
        self.assertFalse(ready_for_reviewed_continuation(self.root))

    def test_failed_preparation_cannot_launch(self):
        self.state("stopped_no_silent_retry_no_training")
        with self.assertRaises(ValueError):
            ready_for_reviewed_continuation(self.root)

    def test_invalid_receipt_is_not_trusted(self):
        self.state("awaiting_actual_final_source_visual_review_no_training")
        (self.root / "visual_review_receipt.json").write_text("{}")
        with patch("VLA.resume_repaired_train_fit.require_visual_receipt", side_effect=ValueError("stale")) as check:
            with self.assertRaises(ValueError):
                ready_for_reviewed_continuation(self.root)
            check.assert_called_once()


class MeasuredFeedbackSupportTest(unittest.TestCase):
    def test_counts_measured_feedback_separately_from_requested_closure(self):
        arrays = {"canonical_actions": np.array([[0, 0, 0, 0, 0, 0, -1], [0, 0, .01, 0, 0, 0, 1.]]),
            "gripper_closure": np.array([.8, .8]), "next_grasped_object": np.array([False, True]),
            "next_verified_grasp": np.array([False, True]), "next_gripper_closure": np.array([.2, .7]),
            "next_requested_gripper_closure": np.array([0., 1.])}
        result = feedback_counts(arrays)
        self.assertEqual(result["initial_closed_without_correct_target"], 1)
        self.assertEqual(result["reopen_labels_from_those_states"], 1)
        self.assertEqual(result["closed_upward_labels_from_those_states"], 1)
        self.assertEqual(result["final_verified_bilateral_target_grasp"], 1)
        self.assertAlmostEqual(result["max_measured_next_vs_requested_closure_gap"], .3)


if __name__ == "__main__":
    unittest.main()
