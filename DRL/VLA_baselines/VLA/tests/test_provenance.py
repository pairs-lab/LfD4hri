from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from VLA.common.provenance import (
    build_directory_fingerprint,
    verify_directory_fingerprint,
)


class DirectoryProvenanceTest(unittest.TestCase):
    def test_fast_and_full_verification_with_explicit_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "episode.parquet").write_bytes(b"payload")
            (root / "conversion_manifest.json").write_text("excluded\n")

            fingerprint = build_directory_fingerprint(
                root,
                exclude_paths=("conversion_manifest.json",),
            )

            self.assertEqual(fingerprint["files"], 1)
            self.assertEqual(
                verify_directory_fingerprint(root, fingerprint)["mode"],
                "fast_layout",
            )
            self.assertEqual(
                verify_directory_fingerprint(root, fingerprint, full=True)["mode"],
                "full",
            )

    def test_full_mode_detects_same_size_same_mtime_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"abcdef")
            fingerprint = build_directory_fingerprint(root)
            original = payload.stat()
            payload.write_bytes(b"ABCDEF")
            os.utime(payload, ns=(original.st_atime_ns, original.st_mtime_ns))

            self.assertTrue(verify_directory_fingerprint(root, fingerprint)["verified"])
            with self.assertRaisesRegex(RuntimeError, "content differs"):
                verify_directory_fingerprint(root, fingerprint, full=True)

    def test_fast_mode_detects_layout_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"abc")
            fingerprint = build_directory_fingerprint(root)
            payload.write_bytes(b"abcd")

            with self.assertRaisesRegex(RuntimeError, "layout differs"):
                verify_directory_fingerprint(root, fingerprint)


if __name__ == "__main__":
    unittest.main()
