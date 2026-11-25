import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest

from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.pi0.codec_provenance import verify_fast_codec_binding


class FastCodecProvenanceTest(unittest.TestCase):
    def test_replay_checks_codec_binding_before_and_after_execution(self):
        path = Path(__file__).resolve().parents[1] / "pi0" / "audit_closed_loop_replay.py"
        tree = ast.parse(path.read_text())
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertEqual(calls.count("verify_fast_codec_binding"), 2)
        self.assertIn("source_payload_fingerprint", calls)
        self.assertIn("verify_source_payload_fingerprint", calls)

    def fixtures(self, root):
        source, codec = root / "source", root / "codec"
        source.mkdir(); codec.mkdir()
        profile = {"gripper_controller": "body_contact_v9"}
        (source / "dataset_manifest.json").write_text(json.dumps({"observation_profile": profile}))
        (source / "index.json").write_text(json.dumps([{"file": "a.npz"}]))
        (source / "a.npz").write_bytes(b"source npz bytes")
        (codec / "a.npz").write_bytes(b"decoded npz bytes")
        manifest = {"selected_files": ["a.npz"],
                    "source_payload_fingerprint": source_payload_fingerprint(source),
                    "codec_payload_fingerprint": source_payload_fingerprint(codec, [{"file": "a.npz"}]),
                    "environment_runtime_fingerprint": environment_runtime_fingerprint(profile)}
        return source, codec, manifest

    def test_exact_v9_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            source, codec, manifest = self.fixtures(Path(directory))
            self.assertTrue(verify_fast_codec_binding(manifest, source, codec)["verified"])

    def test_changed_decoded_target_rejected_with_unchanged_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, codec, manifest = self.fixtures(Path(directory))
            (codec / "a.npz").write_bytes(b"mutated npz bytes")
            with self.assertRaisesRegex(RuntimeError, "content/hash differs"):
                verify_fast_codec_binding(manifest, source, codec)

    def test_changed_source_rejected_with_unchanged_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, codec, manifest = self.fixtures(Path(directory))
            (source / "a.npz").write_bytes(b"changed source")
            with self.assertRaisesRegex(RuntimeError, "content/hash differs"):
                verify_fast_codec_binding(manifest, source, codec)

    def test_missing_v9_source_codec_or_runtime_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, codec, manifest = self.fixtures(Path(directory))
            for key in ("source_payload_fingerprint", "codec_payload_fingerprint", "environment_runtime_fingerprint"):
                invalid = copy.deepcopy(manifest); invalid.pop(key)
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    verify_fast_codec_binding(invalid, source, codec)

    def test_unknown_codec_filename_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, codec, manifest = self.fixtures(Path(directory))
            manifest["selected_files"] = ["unknown.npz"]
            with self.assertRaisesRegex(RuntimeError, "unknown source episode"):
                verify_fast_codec_binding(manifest, source, codec)

    def test_legacy_missing_new_fields_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "dataset_manifest.json").write_text(json.dumps({}))
            self.assertIsNone(verify_fast_codec_binding({}, source, source))


if __name__ == "__main__":
    unittest.main()
