import copy
from pathlib import Path
import tempfile
import unittest

from VLA.common.runtime_provenance import (
    LEGACY_SOURCES,
    V9_ADDITIONAL_SOURCES,
    environment_runtime_fingerprint,
    verify_environment_runtime_fingerprint,
)


class RuntimeProvenanceTest(unittest.TestCase):
    V9_PROFILE = {"gripper_controller": "body_contact_v9"}

    def fixture(self, root):
        for path in LEGACY_SOURCES + V9_ADDITIONAL_SOURCES:
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# fixture: " + path + "\n")

    def test_legacy_dependency_list_remains_explicit_and_stable(self):
        expected = (
            "VLA/common/environment.py",
            "VLA/common/observation_profile.py",
            "VLA/common/robotiq.py",
        )
        self.assertEqual(LEGACY_SOURCES, expected)
        for profile in (None, {}, {"name": "historical_camera_profile"}):
            fingerprint = environment_runtime_fingerprint(profile)
            self.assertEqual(tuple(fingerprint["sources_sha256"]), expected)
            self.assertEqual(fingerprint["controller_contract"], "legacy")
            self.assertEqual(verify_environment_runtime_fingerprint(fingerprint, profile), fingerprint)

    def test_v9_includes_all_body_controller_safety_and_dispatch_sources(self):
        fingerprint = environment_runtime_fingerprint(self.V9_PROFILE)
        self.assertEqual(tuple(fingerprint["sources_sha256"]), LEGACY_SOURCES + (
            "VLA/common/body_environment.py",
            "VLA/common/body_gripper.py",
            "VLA/common/physics_safety.py",
            "VLA/common/environment_factory.py",
        ))
        self.assertEqual(fingerprint["controller_contract"], "body_contact_v9")
        self.assertEqual(verify_environment_runtime_fingerprint(fingerprint, self.V9_PROFILE), fingerprint)

    def test_each_v9_source_change_invalidates_saved_replay_fingerprint(self):
        for changed in LEGACY_SOURCES + V9_ADDITIONAL_SOURCES:
            with self.subTest(path=changed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture(root)
                fingerprint = environment_runtime_fingerprint(self.V9_PROFILE, project_root=root)
                (root / changed).write_text("# changed implementation\n")
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    verify_environment_runtime_fingerprint(fingerprint, self.V9_PROFILE, project_root=root)

    def test_wrong_contract_and_dependency_downgrade_are_rejected(self):
        fingerprint = environment_runtime_fingerprint(self.V9_PROFILE)
        for key, value in (("role", "dataset"), ("version", True),
                           ("version", 2), ("controller_contract", "legacy"),
                           ("sources_sha256", {path: fingerprint["sources_sha256"][path]
                                               for path in LEGACY_SOURCES})):
            with self.subTest(key=key, value=value):
                altered = copy.deepcopy(fingerprint)
                altered[key] = value
                with self.assertRaises(RuntimeError):
                    verify_environment_runtime_fingerprint(altered, self.V9_PROFILE)
        with self.assertRaises(RuntimeError):
            verify_environment_runtime_fingerprint(None, self.V9_PROFILE)
        with self.assertRaises(RuntimeError):
            verify_environment_runtime_fingerprint(environment_runtime_fingerprint(None), self.V9_PROFILE)

    def test_unknown_controller_never_falls_back_to_legacy(self):
        with self.assertRaises(ValueError):
            environment_runtime_fingerprint({"gripper_controller": "unrecognized"})


if __name__ == "__main__":
    unittest.main()
