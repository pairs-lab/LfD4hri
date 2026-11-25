import json
import os
from pathlib import Path
import tempfile
import unittest

from VLA.evaluate_openpi_development import process_identity, validate_checkpoint_contract


class OpenPIDevelopmentDriverTest(unittest.TestCase):
    def test_current_process_identity_and_missing_process(self):
        self.assertEqual(process_identity(os.getpid()), process_identity(os.getpid()))
        self.assertIsNotNone(process_identity(os.getpid()))
        self.assertIsNone(process_identity(999999999))

    def test_checkpoint_source_and_accumulation_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "source"
            contract = {"model": "pi0", "repo_id": "train_only", "source": {"source": str(source)},
                        "optimization": {"partial_accumulation_microsteps": 0}}
            path = folder / "vla_checkpoint_manifest.json"
            path.write_text(json.dumps(contract))
            self.assertEqual(validate_checkpoint_contract(folder, "pi0", "train_only", source), contract)
            for model, repo, src in (("pi0_fast", "train_only", source), ("pi0", "other", source),
                                      ("pi0", "train_only", folder / "validation")):
                with self.assertRaises(RuntimeError):
                    validate_checkpoint_contract(folder, model, repo, src)
            contract["optimization"]["partial_accumulation_microsteps"] = 1
            path.write_text(json.dumps(contract))
            with self.assertRaises(RuntimeError):
                validate_checkpoint_contract(folder, "pi0", "train_only", source)


if __name__ == "__main__":
    unittest.main()
