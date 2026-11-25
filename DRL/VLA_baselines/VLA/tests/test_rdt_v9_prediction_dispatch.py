"""Keep offline RDT audits on their source's controller and exact reset context."""
import ast
from pathlib import Path
import unittest


class RDTV9PredictionDispatchTest(unittest.TestCase):
    def test_source_dispatch_and_reset_are_used_in_prediction_audit(self):
        path = Path(__file__).resolve().parents[1] / "rdt" / "audit_checkpoint_predictions.py"
        tree = ast.parse(path.read_text())
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("audit_environment", calls)
        self.assertIn("reset_source_episode", calls)
        self.assertNotIn("ManipulationEnv", calls)
        self.assertIn("verify_source_payload_for_profile", calls)
        self.assertIn("verify_environment_runtime_fingerprint", calls)


if __name__ == "__main__":
    unittest.main()
