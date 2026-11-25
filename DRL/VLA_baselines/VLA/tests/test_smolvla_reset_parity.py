import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.evaluate_smolvla_v9_source import reset_observation_errors, validate_reset_observation, verify_source_audit


class SmolVLAResetParityTest(unittest.TestCase):
    def sensors(self):
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        obs = SimpleNamespace(joint_positions=np.zeros(6), front_rgb=rgb.copy(), wrist_rgb=rgb.copy())
        recorded = {"joint_positions": np.zeros((1, 6)), "gripper_closure": np.array([.97]),
                    "front_rgb": rgb[None].copy(), "wrist_rgb": rgb[None].copy()}
        return obs, recorded

    def test_exact_observation_and_documented_one_lsb_rounding_pass(self):
        obs, recorded = self.sensors()
        obs.front_rgb[0, 0] = 1  # One changed pixel, not three separately counted channels.
        errors = reset_observation_errors(obs, .97, recorded)
        self.assertAlmostEqual(errors["front_rgb_changed_pixel_fraction"], 1 / 1600)
        self.assertTrue(validate_reset_observation(errors)["passed"])

    def test_real_image_changes_and_broad_one_lsb_drift_fail(self):
        for camera, broad in (("front", False), ("wrist", False), ("front", True)):
            with self.subTest(camera=camera, broad=broad):
                obs, recorded = self.sensors()
                actual = getattr(obs, camera + "_rgb")
                if broad:
                    actual[:] = 1
                else:
                    actual[0, 0] = 2
                with self.assertRaisesRegex(ValueError, camera):
                    validate_reset_observation(reset_observation_errors(obs, .97, recorded))

    def test_state_drift_missing_and_nonfinite_metrics_fail(self):
        obs, recorded = self.sensors()
        for closure, joint in ((.970002, 0), (.97, .000002), (float("nan"), 0)):
            obs.joint_positions[0] = joint
            with self.assertRaises(ValueError):
                validate_reset_observation(reset_observation_errors(obs, closure, recorded))
        with self.assertRaises(ValueError):
            validate_reset_observation({})

    def test_shape_and_float_rgb_cannot_silently_broadcast(self):
        obs, recorded = self.sensors()
        obs.front_rgb = obs.front_rgb.astype(np.float32)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            reset_observation_errors(obs, .97, recorded)
        obs, recorded = self.sensors()
        obs.joint_positions = np.zeros((1, 6))
        with self.assertRaisesRegex(ValueError, "joint"):
            reset_observation_errors(obs, .97, recorded)

    def test_missing_or_stale_source_runtime_rejects_even_passing_camera_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            profile = {"gripper_controller": "body_contact_v9"}
            fingerprint = environment_runtime_fingerprint(profile)
            manifest = {"observation_profile": profile, "required_success_steps": 1,
                        "privileged_pose_in_policy_input": False, "environment_runtime_fingerprint": fingerprint}
            (source / "index.json").write_text("[]")
            for proof in (None, {**fingerprint, "sources_sha256": {key: "stale" for key in fingerprint["sources_sha256"]}}, fingerprint):
                manifest["environment_runtime_fingerprint"] = proof
                (source / "dataset_manifest.json").write_text(json.dumps(manifest))
                audit = {"passed": True, "failures": [], "dataset": str(source),
                         "index_sha256": sha256_file(source / "index.json"),
                         "manifest_sha256": sha256_file(source / "dataset_manifest.json")}
                path = source / "audit.json"
                path.write_text(json.dumps(audit))
                with patch("VLA.evaluate_smolvla_v9_source.verify_source_payload_for_profile"):
                    if proof is fingerprint:
                        self.assertEqual(verify_source_audit(source, path)[1], manifest)
                    else:
                        with self.assertRaises(RuntimeError):
                            verify_source_audit(source, path)


if __name__ == "__main__":
    unittest.main()
