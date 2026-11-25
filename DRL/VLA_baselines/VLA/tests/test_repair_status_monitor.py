import json
from pathlib import Path
import tempfile
import unittest

from VLA.monitor_repair_cycles import model_snapshot


class RepairMonitorTests(unittest.TestCase):
    def test_pending_fit_is_not_reported_as_zero_percent_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text(json.dumps({"status":"running","stage":"train","updates":5000}))
            (root / "logs").mkdir()
            (root / "logs/train.log").write_text("Training: 1% | 160/80000 [00:20]\r")
            report = model_snapshot(root,"smolvla")
            self.assertIsNone(report["completed_new_development_success_rate"])
            self.assertEqual(report["approximate_optimizer_updates_from_log"],10)
            self.assertTrue(report["log_counter_NOT_saved_Adam_counter"])

    def test_completed_rdt_reports_actual_success_not_trial_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text(json.dumps({"status":"completed_fit_test_remaining_errors_require_review","stage":"done","updates":5000}))
            (root / "development40").mkdir()
            (root / "development40/progress.json").write_text(json.dumps({"completed_trials":40,"counts":{"touch":10,"pick":10,"move":10,"pick_place":10}}))
            report = model_snapshot(root,"rdt")
            self.assertIsNone(report["completed_new_development_success_rate"])
            (root / "development40/analysis.json").write_text(json.dumps({"complete40":True,"successes":12,"statistics":{"by_task":{"touch":{"trials":10,"successes":3}}}}))
            report = model_snapshot(root,"rdt")
            self.assertEqual(report["completed_new_development_success_rate"],.3)
            self.assertEqual(report["completed_development_statistics"]["by_task"]["touch"]["successes"],3)


if __name__ == "__main__":
    unittest.main()
