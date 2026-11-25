import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from VLA.audit_saved_gripper_thresholds import rescore_rows
from VLA.smolvla.audit_checkpoint_predictions import summarize


class SavedGripperThresholdTests(unittest.TestCase):
    def fixture(self, path, *, v9=True, startup=None, requested=True):
        manifest = {"observation_profile": {"gripper_controller": "body_contact_v9" if v9 else "legacy"}}
        (path / "dataset_manifest.json").write_text(json.dumps(manifest))
        row = {"file": "episode.npz"}
        if startup:
            row["recovery_start_actions_not_training_labels"] = startup
        (path / "index.json").write_text(json.dumps([row]))
        arrays = {"gripper_closure": np.array([.97, .3]), "next_grasped_object": np.array([False, True])}
        if requested:
            arrays["next_requested_gripper_closure"] = np.array([1., 1.])
        np.savez(path / "episode.npz", **arrays)
        return {"source": str(path), "rows": [{"episode": "episode.npz", "frame": 1,
            "predicted": [0, 0, 0, 0, 0, 0, -.1], "expected": [0, 0, 0, 0, 0, 0, 1]}]}

    def test_partial_aperture_does_not_erase_close_request(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = rescore_rows(self.fixture(Path(directory)))
            self.assertEqual(rows[0]["commanded_closure_before_action"], 1.)
            self.assertEqual(rows[0]["measured_closure_before_action"], .3)
            self.assertEqual(summarize(rows)["gripper_applied_command_agreement"], 1.)

    def test_recovery_initial_state_uses_startup_not_aperture(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.fixture(Path(directory), startup=[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1]])
            report["rows"][0]["frame"] = 0
            self.assertEqual(rescore_rows(report)[0]["commanded_closure_before_action"], 0.)

    def test_missing_v9_request_history_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyError):
                rescore_rows(self.fixture(Path(directory), requested=False))

    def test_legacy_command_state_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = rescore_rows(self.fixture(Path(directory), v9=False, requested=False))
            self.assertEqual(rows[0]["commanded_closure_before_action"], .3)

    def test_invalid_frame_is_not_silently_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.fixture(Path(directory))
            for frame in (-1, 2, 1.5):
                report["rows"][0]["frame"] = frame
                with self.assertRaises(ValueError):
                    rescore_rows(report)


if __name__ == "__main__":
    unittest.main()
