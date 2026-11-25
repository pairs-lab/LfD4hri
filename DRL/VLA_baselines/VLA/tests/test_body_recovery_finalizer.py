"""Candidate-only finalizer contracts; immutable real TRAIN fixtures are copied."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from VLA.finalize_body_recovery_candidate import finalize


PROJECT = Path(__file__).resolve().parents[2]
NEAR_SOURCE = PROJECT/"VLA/datasets/candidates/body_v9_near_object_missed_close_recovery_pilot_20261009"
WORKSPACE_SOURCE = PROJECT/"VLA/datasets/candidates/body_v9_closed_empty_recovery_pilot_20261008"
NEAR_KIND = "closed_empty_near_object_missed_close"


@unittest.skipUnless((NEAR_SOURCE/"dataset_manifest.json").exists()
                     and (WORKSPACE_SOURCE/"dataset_manifest.json").exists(), "Audited TRAIN fixtures not installed")
class BodyRecoveryFinalizerTest(unittest.TestCase):
    def fixture(self, root, source=NEAR_SOURCE):
        for filename in ("index.json", "audit.json", "collection_state.json"):
            shutil.copy2(source/filename, root/filename)
        rows = json.loads((root/"index.json").read_text())
        for row in rows:
            shutil.copy2(source/row["file"], root/row["file"])
        return rows

    def finish(self, root, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return finalize(root, **kwargs)

    def test_workspace301_default_backwards_compatible_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, WORKSPACE_SOURCE)
            self.assertEqual(self.finish(root)["frames"], 301)
            before = (root/"dataset_manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.finish(root)
            self.assertEqual(before, (root/"dataset_manifest.json").read_bytes())

    def test_near_miss275_explicit_kind_binds_actual_anchor_and_candidate_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = self.finish(root, recovery_kind=NEAR_KIND, expected_per_task=2, expected_transitions=275)
            manifest = json.loads((root/"dataset_manifest.json").read_text())
            self.assertEqual(result["frames"], 275)
            self.assertEqual(manifest["near_object_startup_unique_anchor_names"], ["green pear"])
            self.assertEqual(manifest["near_object_startup_unique_anchor_count"], 1)
            self.assertTrue(manifest["near_object_deliberate_missed_close_pilot_audited"])
            self.assertFalse(manifest["training_certified"])
            self.assertFalse(manifest["near_object_missed_grasp_recovery_certified"])

    def test_default_kind_does_not_silently_accept_near_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Recovery kind"):
                self.finish(root)
            self.assertFalse((root/"dataset_manifest.json").exists())

    def test_declared_counts_and_transitions_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "episode count"):
                self.finish(root, recovery_kind=NEAR_KIND, expected_per_task=4)
            with self.assertRaisesRegex(ValueError, "276 corrective transitions"):
                self.finish(root, recovery_kind=NEAR_KIND, expected_transitions=276)
            self.assertFalse((root/"dataset_manifest.json").exists())

    def test_nonfinite_or_unheld_claim_with_contacts_fails_before_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            for value, message in ((float("nan"), "invalid physical feedback"), (.80, "physically safe/closed/empty")):
                changed = copy.deepcopy(rows)
                changed[0]["near_object_recovery_replay_evidence"]["actual_state"]["actual_closure"] = value
                (root/"index.json").write_text(json.dumps(changed))
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, message):
                        self.finish(root, recovery_kind=NEAR_KIND)
            changed = copy.deepcopy(rows)
            changed[0]["near_object_recovery_replay_evidence"]["actual_state"]["gripper_contact_count"] = 1
            (root/"index.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "physically safe/closed/empty"):
                self.finish(root, recovery_kind=NEAR_KIND)
            self.assertFalse((root/"dataset_manifest.json").exists())

    def test_false_verified_carry_and_changed_observation_feedback_are_rejected(self):
        for mutation, message in (("carry", "without verified bilateral grasp"),
                                  ("feedback", "differs from recorded t0 observation")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = self.fixture(root)
                row = next(row for row in rows if row["task"]["name"] == "pick")
                path = root/row["file"]
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {name: saved[name] for name in saved.files}
                if mutation == "carry":
                    arrays["next_verified_grasp"][-1] = False
                else:
                    arrays["gripper_closure"][0] -= .002
                np.savez_compressed(path, **arrays)
                with self.assertRaisesRegex(ValueError, message):
                    self.finish(root, recovery_kind=NEAR_KIND)
                self.assertFalse((root/"dataset_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
