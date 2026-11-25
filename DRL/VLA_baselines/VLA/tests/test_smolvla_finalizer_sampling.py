import copy
import json
from pathlib import Path
import tempfile
import unittest

from VLA.smolvla.balanced_sampling import sampling_contract
from VLA.smolvla.finalize_checkpoint import validate_sampling_contract


class SmolVLAFinalizerSamplingTest(unittest.TestCase):
    def make_source(self, root):
        # Uneven within-task target counts distinguish target-balanced from
        # merely task/episode-balanced sampling.
        rows = []
        for name in ("touch", "pick", "move", "pick_place"):
            for target in ("red apple", "red apple", "blue block"):
                rows.append({"task": {"name": name, "pick_name": target}, "steps": 3})
        (root / "index.json").write_text(json.dumps(rows))
        return rows

    def test_target_balanced_intermediate_contract_is_not_reconstructed_as_false(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            recorded = sampling_contract(root, .30, target_balanced=True)
            self.assertNotEqual(recorded, sampling_contract(root, .30, target_balanced=False))
            self.assertEqual(validate_sampling_contract(root, recorded), recorded)

    def test_task_episode_only_contract_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            recorded = sampling_contract(root, .30, target_balanced=False)
            self.assertEqual(validate_sampling_contract(root, recorded), recorded)

    def test_stale_source_and_malformed_mode_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_source(root)
            recorded = sampling_contract(root, .30, target_balanced=True)
            malformed = copy.deepcopy(recorded)
            malformed["target_balanced"] = "false"
            with self.assertRaisesRegex(RuntimeError, "must be boolean"):
                validate_sampling_contract(root, malformed)
            rows[0]["steps"] = 4
            (root / "index.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(RuntimeError, "differs"):
                validate_sampling_contract(root, recorded)


if __name__ == "__main__":
    unittest.main()
