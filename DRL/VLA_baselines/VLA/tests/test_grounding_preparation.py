import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from VLA.common.provenance import sha256_file
from VLA.prepare_v9_candidate_fit import collection_ready


class GroundingPreparationTests(unittest.TestCase):
    def test_nonuniform_counts_are_explicit_and_exact(self):
        counts = {"touch": 2, "pick": 3, "move": 4, "pick_place": 5}
        with TemporaryDirectory() as directory:
            source = Path(directory)
            rows = [{"task": {"name": name}, "success": True, "safety_violation": None}
                    for name, count in counts.items() for _ in range(count)]
            (source/"index.json").write_text(json.dumps(rows))
            (source/"collection_state.json").write_text(json.dumps({
                "status": "completed_candidate_not_qualified", "completed_episodes": len(rows),
                "index_sha256": sha256_file(source/"index.json")}))
            self.assertTrue(collection_ready(source, counts))
            with self.assertRaises(RuntimeError): collection_ready(source, 2)
            with self.assertRaises(RuntimeError): collection_ready(source, {**counts, "pick": 2})

    def test_zero_and_missing_counts_are_not_automatic_acceptance(self):
        for expected in (0, {}, {"touch": 1}):
            with self.assertRaises(ValueError): collection_ready(Path("unused"), expected)


if __name__ == "__main__":
    unittest.main()
