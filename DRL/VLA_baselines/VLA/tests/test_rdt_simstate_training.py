import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from VLA.rdt.train_simstate import require_new_qualification


class PrivilegedRDTTrainingGateTests(unittest.TestCase):
    def test_two_update_pipeline_probe_permitted(self):
        require_new_qualification(SimpleNamespace(steps=2), None, Path("/unused"))

    def test_unqualified_long_fit_blocked_before_gpu(self):
        with self.assertRaisesRegex(ValueError, "OWN real CUDA"):
            require_new_qualification(SimpleNamespace(steps=5000), None, Path("/unused"))

    def test_generic_old_qualification_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            proof = Path(folder)/"old.json"
            proof.write_text('{"passed":true,"scope":"RDT_FP32_signal_two_update_integration_qualification_NOT_task_success"}')
            with self.assertRaisesRegex(ValueError, "Incomplete/wrong"):
                require_new_qualification(SimpleNamespace(steps=1000), proof, Path(folder))


if __name__ == "__main__":
    unittest.main()
