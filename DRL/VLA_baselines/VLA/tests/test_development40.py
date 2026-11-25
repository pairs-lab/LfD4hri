import json
from pathlib import Path
import tempfile
import unittest

from VLA.run_diagnostic40 import SEEDS, TASKS, rows_from_result
from VLA.analyze_diagnostic40 import analyze


class Development40Tests(unittest.TestCase):
    def test_exact_budget_and_fixed_seed_prefix(self):
        self.assertEqual(len(SEEDS), 10)
        self.assertEqual(len(TASKS) * len(SEEDS), 40)
        self.assertEqual(SEEDS, list(range(20267001, 20267011)))

    def make_trial(self, root):
        tasks = [{"name": task, "success": False, "error": None} for task in TASKS]
        p = root / "results.json"
        p.write_text(json.dumps({"model": "smolvla", "tasks": tasks}))
        (root / "manifest.json").write_text(json.dumps({"seed": SEEDS[0], "replan_steps": 1,
            "privileged_object_pose_given_to_vla": False}))
        return p, tasks

    def test_all_task_failures_retained_not_rerolled(self):
        with tempfile.TemporaryDirectory() as folder:
            p, _ = self.make_trial(Path(folder))
            rows = rows_from_result(p, SEEDS[0], True)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(not r["success"] for r in rows))
            self.assertEqual({r["task"] for r in rows}, set(TASKS))

    def test_integration_errors_and_missing_tasks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p, tasks = self.make_trial(Path(folder))
            for bad in (tasks[:3], [{**tasks[0], "error": "adapter failed"}, *tasks[1:]]):
                p.write_text(json.dumps({"model": "smolvla", "tasks": bad}))
                with self.assertRaises(ValueError):
                    rows_from_result(p, SEEDS[0], False)

    def test_seed_mismatch_and_privileged_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p, _ = self.make_trial(root)
            with self.assertRaises(ValueError):
                rows_from_result(p, SEEDS[1], False)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["privileged_object_pose_given_to_vla"] = True
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                rows_from_result(p, SEEDS[0], False)

    def test_analysis_does_not_call_partial_or_wrong_seed_set_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / "progress.json"
            for rows in ([], [{"seed": s, "task": t} for s in SEEDS for t in TASKS][:-1],
                         [{"seed": s + 1, "task": t} for s in SEEDS for t in TASKS]):
                p.write_text(json.dumps({"status": "completed40_development_trials", "rows": rows}))
                with self.assertRaises(ValueError):
                    analyze(p, root / "analysis")
                self.assertFalse((root / "analysis").exists())


if __name__ == "__main__":
    unittest.main()
