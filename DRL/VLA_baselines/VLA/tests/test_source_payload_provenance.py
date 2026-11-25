from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from VLA.common.source_payload_provenance import (
    source_payload_fingerprint,
    verify_source_payload_fingerprint,
)


class SourcePayloadProvenanceTest(unittest.TestCase):
    def make_source(self, root: Path) -> list[dict]:
        (root / "nested").mkdir()
        (root / "a.npz").write_bytes(b"abcdef")
        (root / "nested/b.npz").write_bytes(b"second payload")
        rows = [{"file": "a.npz", "task": {"name": "touch"}}, {"file": "nested/b.npz"}]
        (root / "index.json").write_text(json.dumps(rows))
        return rows

    def test_default_explicit_rows_and_index_path_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_source(root)
            saved = source_payload_fingerprint(root)
            self.assertEqual(saved, source_payload_fingerprint(root, rows))
            self.assertEqual(saved, source_payload_fingerprint(root, root / "index.json"))
            self.assertEqual(set(saved["payloads"]), {"a.npz", "nested/b.npz"})
            self.assertEqual(saved["files"], 2)
            self.assertTrue(verify_source_payload_fingerprint(saved, root)["verified"])

    def test_unlisted_files_and_row_order_do_not_change_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_source(root)
            saved = source_payload_fingerprint(root)
            (root / "unlisted.npz").write_bytes(b"not a training episode")
            (root / "audit.json").write_text("arbitrary audit contents")
            self.assertEqual(saved, source_payload_fingerprint(root, list(reversed(rows))))
            self.assertTrue(verify_source_payload_fingerprint(saved, root)["verified"])

    def test_changed_payload_with_unchanged_index_size_and_mtime_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            index_bytes = (root / "index.json").read_bytes()
            saved = source_payload_fingerprint(root)
            payload = root / "a.npz"
            original = payload.stat()
            payload.write_bytes(b"ABCDEF")
            os.utime(payload, ns=(original.st_atime_ns, original.st_mtime_ns))
            self.assertEqual(index_bytes, (root / "index.json").read_bytes())
            with self.assertRaisesRegex(RuntimeError, "content/hash differs.*a.npz"):
                verify_source_payload_fingerprint(saved, root)

    def test_filename_change_with_identical_payload_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_source(root)
            saved = source_payload_fingerprint(root)
            (root / "renamed.npz").write_bytes((root / "a.npz").read_bytes())
            rows[0]["file"] = "renamed.npz"
            with self.assertRaisesRegex(RuntimeError, "filenames differ"):
                verify_source_payload_fingerprint(saved, root, rows)

    def test_missing_fingerprint_version_role_records_and_summary_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            saved = source_payload_fingerprint(root)
            for invalid in (None, {}, {**saved, "version": 999}, {**saved, "version": True},
                            {**saved, "role": "another artifact"}, {**saved, "payloads": {}},
                            {**saved, "payload_tree_sha256": "0" * 64}, {**saved, "files": 3}):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(RuntimeError):
                        verify_source_payload_fingerprint(invalid, root)
            corrupt = copy.deepcopy(saved)
            corrupt["payloads"]["a.npz"]["sha256"] = "not a hash"
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                verify_source_payload_fingerprint(corrupt, root)

    def test_duplicate_filename_and_normalized_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            for duplicate in ("a.npz", "./a.npz"):
                with self.subTest(duplicate=duplicate):
                    with self.assertRaisesRegex(ValueError, "Duplicate"):
                        source_payload_fingerprint(root, [{"file": "a.npz"}, {"file": duplicate}])

    def test_absolute_parent_traversal_non_npz_and_malformed_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            invalid_names = (str(root / "a.npz"), "../a.npz", "nested/../../a.npz", "index.json", "", "a\\b.npz")
            for name in invalid_names:
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        source_payload_fingerprint(root, [{"file": name}])
            for index in ({"file": "a.npz"}, [], [None], [{}]):
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        source_payload_fingerprint(root, index)

    def test_missing_and_nonfile_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root)
            with self.assertRaisesRegex(FileNotFoundError, "Missing indexed"):
                source_payload_fingerprint(root, [{"file": "missing.npz"}])
            (root / "directory.npz").mkdir()
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                source_payload_fingerprint(root, [{"file": "directory.npz"}])
            saved = source_payload_fingerprint(root)
            (root / "a.npz").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing indexed"):
                verify_source_payload_fingerprint(saved, root)

    def test_symlink_escape_and_duplicate_internal_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            self.make_source(root)
            target = Path(outside) / "outside.npz"
            target.write_bytes(b"outside")
            (root / "escape.npz").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "outside the source"):
                source_payload_fingerprint(root, [{"file": "escape.npz"}])
            (root / "alias.npz").symlink_to(root / "a.npz")
            with self.assertRaisesRegex(ValueError, "Duplicate.*alias"):
                source_payload_fingerprint(root, [{"file": "a.npz"}, {"file": "alias.npz"}])


if __name__ == "__main__":
    unittest.main()
