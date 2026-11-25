import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.analyze_cross_vla120 import analyze
from VLA.run_priority_postfit_cycle import validate_handoff
from VLA.run_cross_vla120 import SEEDS, TASKS


class PriorityPostfitTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        previous = root / "previous"; previous.mkdir()
        (previous / "status.json").write_text(json.dumps({"pid": 123, "status": "running", "stage": "project_validation"}))
        for name in ("smol_precision", "rdt_precision", "rdt_reload", "smol_actual_input_parity"):
            (previous / (name + ".json")).write_text('{"passed":true}')
        for pid, module, output in ((123, "VLA.recover_interrupted_cycles", previous),
                                   (456, "VLA.resume_simstate_projection", previous / "validation")):
            proc = root / "proc" / str(pid)
            proc.mkdir(parents=True)
            (proc / "cmdline").write_bytes(b"python\0-m\0" + module.encode() + b"\0--output\0" + str(output).encode() + b"\0validation\0")
            (proc / "cwd").symlink_to(root)
            (proc / "task" / str(pid)).mkdir(parents=True)
            (proc / "task" / str(pid) / "children").write_text("456" if pid == 123 else "")
        def fake_path(value):
            path = Path(value)
            return root / "proc" / str(path).removeprefix("/proc/") if str(path).startswith("/proc/") else path
        return root, previous, fake_path

    def test_only_passive_parent_with_exact_CPU_child_is_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root, previous, fake = self.fixture(directory)
            with patch("VLA.run_priority_postfit_cycle.Path", side_effect=fake), patch("VLA.run_priority_postfit_cycle.ROOT", root), patch("VLA.run_priority_postfit_cycle.pid_alive", return_value=True), patch("VLA.run_priority_postfit_cycle.queue.gpu_state", return_value={"compute_pids":[]}):
                state, child = validate_handoff(previous)
                self.assertEqual((state["pid"], child), (123, 456))

    def test_active_neural_job_prevents_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root, previous, fake = self.fixture(directory)
            with patch("VLA.run_priority_postfit_cycle.Path", side_effect=fake), patch("VLA.run_priority_postfit_cycle.ROOT", root), patch("VLA.run_priority_postfit_cycle.pid_alive", return_value=True), patch("VLA.run_priority_postfit_cycle.queue.gpu_state", return_value={"compute_pids":["789"]}):
                with self.assertRaises(ValueError): validate_handoff(previous)

    def test_unexpected_second_child_prevents_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root, previous, fake = self.fixture(directory)
            (root / "proc/123/task/123/children").write_text("456 789")
            with patch("VLA.run_priority_postfit_cycle.Path", side_effect=fake), patch("VLA.run_priority_postfit_cycle.ROOT", root), patch("VLA.run_priority_postfit_cycle.pid_alive", return_value=True):
                with self.assertRaises(ValueError): validate_handoff(previous)

    def test_failed_saved_gate_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, previous, fake = self.fixture(directory)
            (previous / "rdt_precision.json").write_text('{"passed":false}')
            with patch("VLA.run_priority_postfit_cycle.Path", side_effect=fake), patch("VLA.run_priority_postfit_cycle.ROOT", root), patch("VLA.run_priority_postfit_cycle.pid_alive", return_value=True), patch("VLA.run_priority_postfit_cycle.queue.gpu_state", return_value={"compute_pids":[]}):
                with self.assertRaises(ValueError): validate_handoff(previous)

    def test_partial_model_is_not_a_zero_percent_120_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "summary.json"
            path.write_text(json.dumps({"status":"integration_error_no_reroll", "rows":[]}))
            with self.assertRaises(ValueError): analyze(path, root / "analysis")
            self.assertFalse((root / "analysis").exists())

    def test_duplicate_seed_cannot_replace_missing_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"seed": s, "task": t, "success": False} for s in SEEDS for t in TASKS]
            rows[-1] = rows[0].copy()
            path = root / "summary.json"
            path.write_text(json.dumps({"status":"completed120", "rows":rows}))
            with self.assertRaises(ValueError): analyze(path, root / "analysis")


if __name__ == "__main__":
    unittest.main()
