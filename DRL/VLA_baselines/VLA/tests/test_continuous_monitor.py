from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.monitor_v6 import jobs, training_progress


class ContinuousMonitorTest(unittest.TestCase):
    def progress(self, model, content):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text(content)
            with patch.dict("VLA.monitor_v6.TRAIN_LOGS", {model: log}), patch.dict("VLA.monitor_v6.RUNS", {model: root}):
                return training_progress(model)

    def test_smol_microsteps_are_not_optimizer_updates(self):
        status = self.progress("smolvla", "Training: 10%| 4001/40000 [1:00]\rTraining: 10%| 4004/40000 [1:01]")
        self.assertEqual(status["optimizer_updates"], 1001)
        self.assertEqual(status["microstep"], 4004)

    def test_numerical_warnings_are_detected_in_both_formats(self):
        for model, log in (("rdt", '{"step": 25, "loss": NaN}'),
                           ("rdt", '{"step": 25, "loss": 0.2, "grad_norm": Infinity}'),
                           ("smolvla", "Training: 200/40000 loss:0.1 grdn:inf")):
            self.assertTrue(self.progress(model, log)["numerical_or_traceback_warning"])

    def test_normal_metrics_are_not_false_warnings(self):
        status = self.progress("rdt", '{"step": 50, "loss": 0.0005, "grad_norm": 0.2}')
        self.assertEqual(status["optimizer_updates"], 50)
        self.assertFalse(status["numerical_or_traceback_warning"])

    def test_diagnostics_do_not_include_final_checkpoints(self):
        self.assertEqual(len({job["id"] for job in jobs()}), len(jobs()))
        for job in jobs():
            self.assertLess(job["step"], 20000 if job["model"] == "rdt" else 40000)


if __name__ == "__main__":
    unittest.main()
