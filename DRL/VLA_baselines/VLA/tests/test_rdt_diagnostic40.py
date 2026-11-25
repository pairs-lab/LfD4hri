import copy
import json
from pathlib import Path
import tempfile
import unittest

from VLA.run_rdt_diagnostic40 import SEEDS, TASKS, analyze_requests, result_rows


class RDTDiagnostic40Tests(unittest.TestCase):
    def test_same_budget_and_seeds_as_shared_development_protocol(self):
        self.assertEqual(SEEDS, list(range(20267001, 20267011)))
        self.assertEqual(set(TASKS), {"touch", "pick", "move", "pick_place"})
        self.assertEqual(len(SEEDS) * len(TASKS), 40)

    def fixture(self, root):
        path = root / "results.json"
        path.write_text(json.dumps({"model": "rdt", "tasks": [
            {"name": task, "success": False, "error": None} for task in TASKS]}))
        (root / "manifest.json").write_text(json.dumps({"seed": SEEDS[0], "replan_steps": 1,
            "privileged_object_pose_given_to_vla": False}))
        return path

    def test_all_failures_remain_in_the_results(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = result_rows(self.fixture(Path(directory)), SEEDS[0], True)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["success"] is False for row in rows))
            self.assertTrue(all(row["video"].endswith("rollout.mp4") for row in rows))

    def test_integration_failure_and_identity_mismatch_are_not_scored_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            with self.assertRaises(ValueError): result_rows(path, SEEDS[1], False)
            result = json.loads(path.read_text())
            result["tasks"][0]["error"] = "server down"
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError): result_rows(path, SEEDS[0], False)

    def test_continuous_close_uses_actual_controller_request_threshold_not_point5(self):
        result = {"name": "pick", "prompt": "Pick the red block up", "success": False, "steps": 1}
        trace = [{"step": 1, "action": [0, 0, .01, 0, 0, 0, .21],
            "interaction_after_audit_only": {"privileged_policy_input": False, "held_object": None,
                "correct_object_held": False, "pad_grasp_center_xyz_m": [0, 0, .1],
                "object_center_xyz_m": [0, 0, .01]}}]
        original = copy.deepcopy(trace)
        report = analyze_requests(result, trace)
        self.assertEqual(report["close_requests"], 1)
        self.assertIn("close_and_lift_without_hold_after_step", report["failure_signatures"])
        self.assertEqual(trace, original)
        trace[0]["action"][6] = .2
        self.assertEqual(analyze_requests(result, trace)["close_requests"], 0)


if __name__ == "__main__":
    unittest.main()
