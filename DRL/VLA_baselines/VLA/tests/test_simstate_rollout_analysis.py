import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from VLA.analyze_simstate120 import analyze, audit_trace


def fixture():
    before = {"pad_grasp_center_xyz_m": [.1,-.4,.2], "object_center_xyz_m": [.2,-.3,.04],
        "goal_xyz_m": [.25,-.25,.1], "object_base_xyz_m": [.2,-.3,.02],
        "correct_object_held": False, "held_object": None}
    pad, target, goal, base = (np.asarray(before[k], dtype=np.float32) for k in
        ("pad_grasp_center_xyz_m", "object_center_xyz_m", "goal_xyz_m", "object_base_xyz_m"))
    state = np.r_[np.zeros(7), pad, 0., target-pad, goal-base, 0.].astype(np.float32)
    step = {"step": 1, "action": [0,0,.01,0,0,0,1], "pre_action_PRIVILEGED_state18": state.tolist(),
        "interaction_before_audit": before, "interaction_after_audit": copy.deepcopy(before), "success": False}
    row = {"seed": 1, "task": "pick", "steps": 1, "success": False, "safety_violation": None}
    return row, [step]


class SimStateRolloutAnalysisTests(unittest.TestCase):
    def test_failed_close_and_lift_does_not_imply_success(self):
        row, trace = fixture()
        report = audit_trace(row, trace)
        self.assertIn("never_held_requested_object", report["failure_signatures"])
        self.assertIn("close_and_lift_without_hold_after_step", report["failure_signatures"])
        self.assertEqual(report["closing_lift_requests_with_pre_action_feedback_zero"], 1)
        self.assertFalse(report["success"])

    def test_future_or_misaligned_position_input_is_rejected(self):
        row, trace = fixture()
        trace[0]["pre_action_PRIVILEGED_state18"][11] += .01
        with self.assertRaises(ValueError):
            audit_trace(row, trace)

    def test_wrong_object_positive_feedback_is_rejected(self):
        row, trace = fixture()
        trace[0]["pre_action_PRIVILEGED_state18"][17] = 1.
        with self.assertRaises(ValueError):
            audit_trace(row, trace)

    def test_neural_partial_close_matches_actual_controller_threshold(self):
        row,trace=fixture();trace[0]["action"][6]=.3
        report=audit_trace(row,trace)
        self.assertEqual(report["close_requests"],1)
        self.assertEqual(report["canonical_close_request_threshold"],.2)
        self.assertNotIn("no_close_request_in_grasp_task",report["failure_signatures"])

    def test_incomplete_results_cannot_be_reported_as_SR120(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "progress.json")
            path.write_text(json.dumps({"status": "running", "rows": []}))
            with self.assertRaises(ValueError):
                analyze(path, Path(directory, "analysis"))

    def test_synthetic_complete_fixture_retains_privileged_and_unreviewed_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for seed in range(30):
                traces = {}
                for task in ("touch", "pick", "move", "pick_place"):
                    row, trace = fixture()
                    row.update(seed=seed, task=task, trajectory=str(root/f"trace{seed}.json"))
                    rows.append(row)
                    traces[task] = trace
                (root/f"trace{seed}.json").write_text(json.dumps(traces))
            path = root / "progress.json"
            path.write_text(json.dumps({"status": "completed120_privileged_input_VLA", "rows": rows}))
            result = analyze(path, root/"analysis")
            self.assertTrue(result["complete120"])
            self.assertTrue(result["privileged_model_inputs"])
            self.assertTrue(result["NOT_plain_VLA_or_autonomous_visual_grounding"])
            self.assertFalse(result["all_video_frames_visually_reviewed"])
            self.assertFalse(result["stable_high_success_established"])
            self.assertEqual(result["trials"], 120)
            self.assertEqual(result["successes"], 0)


if __name__ == "__main__":
    unittest.main()
