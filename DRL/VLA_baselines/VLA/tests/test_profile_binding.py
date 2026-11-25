import json
from pathlib import Path
import tempfile
import unittest

from VLA.common.observation_profile import baseline_profile, profile_hash
from VLA.common.profile_binding import checkpoint_profile, resolve_profile
from VLA.common.provenance import sha256_file


class ProfileBindingTest(unittest.TestCase):
    def fixture(self, root, layout="flat"):
        source, checkpoint = root/"source", root/"checkpoint"
        source.mkdir(); checkpoint.mkdir()
        profile = baseline_profile()
        (source/"dataset_manifest.json").write_text(json.dumps({"observation_profile": profile,
            "observation_profile_sha256": profile_hash(profile)}))
        lineage = {"source": str(source), "source_manifest_sha256": sha256_file(source/"dataset_manifest.json")}
        metadata = lineage if layout == "flat" else {"lineage": lineage} if layout == "rdt" else {"source": {"source": str(source), "manifest_sha256": lineage["source_manifest_sha256"]}}
        (checkpoint/"vla_checkpoint_manifest.json").write_text(json.dumps(metadata))
        return source, checkpoint, json.loads(json.dumps(profile))

    def test_native_lineages_auto_restore_exact_sensor_profile(self):
        for layout in ("flat", "rdt", "openpi"):
            with tempfile.TemporaryDirectory() as temp:
                _, checkpoint, profile = self.fixture(Path(temp), layout)
                self.assertEqual(checkpoint_profile(checkpoint), profile)
                self.assertEqual(resolve_profile("smolvla", checkpoint), profile)

    def test_modified_source_is_rejected_before_deployment(self):
        with tempfile.TemporaryDirectory() as temp:
            source, checkpoint, _ = self.fixture(Path(temp))
            (source/"dataset_manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                checkpoint_profile(checkpoint)

    def test_old_checkpoint_cannot_silently_accept_new_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"profile.json"
            path.write_text(json.dumps(baseline_profile()))
            with self.assertRaises(ValueError):
                resolve_profile("openvla", "pretrained/unbound", path)


if __name__ == "__main__":
    unittest.main()
