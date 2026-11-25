from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.benchmark import _checkpoint_fingerprint
from VLA.fingerprint_checkpoint import REPORT_NAME, layout_records, layout_sha256
from VLA.fingerprint_checkpoint import main as fingerprint_main


class CheckpointFingerprintTest(unittest.TestCase):
    def call_cli(self, root, *options):
        with patch("sys.argv", ["fingerprint", str(root), *options]):
            fingerprint_main()

    def test_verify_existing_reuses_valid_report_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "params.bin").write_bytes(b"checkpoint")
            self.call_cli(root, "--verify-existing")
            report = root / REPORT_NAME
            before, mtime = report.read_bytes(), report.stat().st_mtime_ns
            self.call_cli(root, "--verify-existing")
            self.assertEqual(report.read_bytes(), before)
            self.assertEqual(report.stat().st_mtime_ns, mtime)
            with self.assertRaises(FileExistsError):
                self.call_cli(root)

    def test_verify_existing_rejects_corruption_even_with_identical_layout(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "params.bin"
            artifact.write_bytes(b"checkpoint")
            self.call_cli(root)
            stat = artifact.stat()
            before = (root / REPORT_NAME).read_bytes()
            artifact.write_bytes(b"badpoints!")
            os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaises(RuntimeError):
                self.call_cli(root, "--verify-existing")
            self.assertEqual((root / REPORT_NAME).read_bytes(), before)

    def test_verify_existing_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.call_cli(Path(directory), "--verify-existing", "--overwrite")

    def test_cached_full_fingerprint_requires_unchanged_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "params.bin"
            artifact.write_bytes(b"checkpoint")
            records = layout_records(root)
            content_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            tree = hashlib.sha256()
            tree.update(b"params.bin\0")
            tree.update(content_digest.encode("ascii"))
            tree.update(b"\n")
            (root / REPORT_NAME).write_text(
                json.dumps(
                    {
                        "version": 1,
                        "checkpoint": str(root.resolve()),
                        "files": 1,
                        "bytes": len(b"checkpoint"),
                        "layout_sha256": layout_sha256(records),
                        "content_tree_sha256": tree.hexdigest(),
                    }
                )
            )

            result = _checkpoint_fingerprint(str(root))
            self.assertTrue(result["layout_verified"])
            self.assertEqual(result["full_content_fingerprint"], tree.hexdigest())

            artifact.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(RuntimeError, "changed after full fingerprint"):
                _checkpoint_fingerprint(str(root))


if __name__ == "__main__":
    unittest.main()
