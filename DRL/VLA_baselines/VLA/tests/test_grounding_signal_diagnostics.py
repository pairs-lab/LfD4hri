import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from VLA.audit_grounding_pair_signal import main
from VLA.audit_localization_handoff_visibility import summary


class GroundingSignalDiagnosticsTests(unittest.TestCase):
    def fixture(self, root, duplicate_prompt=False):
        rows = []
        for task in ("touch", "pick", "move", "pick_place"):
            for variant in (0, 1):
                file = f"{task}_{variant}.npz"
                actions = np.zeros((10, 7), np.float32)
                actions[:, 2] = -.01
                actions[:, 6] = -1
                actions[6:, 0] = .01 if variant == 0 else -.01
                np.savez(root / file, canonical_actions=actions,
                    joint_positions=np.zeros((10, 6), np.float32), gripper_closure=np.zeros(10))
                rows.append({"file": file, "steps": 10, "canonical_source": "/offline/train",
                    "paired_scene_source": task+"_scene.npz", "task": {"name": task,
                    "pick_name": "red" if variant == 0 else "blue",
                    "prompt": task + (" red" if variant == 0 or duplicate_prompt else " blue")}})
        (root / "index.json").write_text(json.dumps(rows))
        (root / "dataset_manifest.json").write_text(json.dumps({"episodes": len(rows)}))

    def test_shared_prefix_does_not_count_initial_anchor_as_grounding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / "report.json"
            with patch("sys.argv", ["audit", "--source", str(root), "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads(output.read_text())
            self.assertEqual(report["pairs"], 4)
            self.assertEqual(report["pairs_with_first_label_contrast"], 0)
            self.assertEqual(report["pairs_with_initial_horizon_contrast"], 0)
            self.assertEqual(report["first_substantive_contrast_histogram"], {"6": 4})
            self.assertAlmostEqual(report["paired_initial_anchor_probability_mass"], .1)
            self.assertAlmostEqual(report["paired_contrast_horizon_anchor_probability_mass"], .8)
            self.assertFalse(report["training_recipe_changed"])

    def test_same_prompt_cannot_be_counted_as_text_intervention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, duplicate_prompt=True)
            with patch("sys.argv", ["audit", "--source", str(root), "--output", str(root / "report.json")]):
                with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
                    main()

    def test_dual_view_visibility_is_not_front_only(self):
        rows = [
            {"first_xy_frame": 17, "initial_pixels": {"front": 100, "wrist": 100}, "handoff_pixels": {"front": 0, "wrist": 80}},
            {"first_xy_frame": 15, "initial_pixels": {"front": 100, "wrist": 0}, "handoff_pixels": {"front": 0, "wrist": 0}},
            {"first_xy_frame": 19, "initial_pixels": {"front": 0, "wrist": 10}, "handoff_pixels": {"front": 30, "wrist": 0}},
        ]
        report = summary(rows)
        self.assertEqual(report["handoff_front_fully_occluded"], 2)
        self.assertEqual(report["handoff_wrist_fully_occluded"], 2)
        self.assertEqual(report["handoff_both_views_fully_occluded"], 1)
        self.assertEqual(report["handoff_best_view_below64_pixels"], 2)
        self.assertEqual(report["initial_best_view_at_least64_but_handoff_below64"], 1)
        self.assertEqual(report["first_xy_frame_quantiles"][1], 17)


if __name__ == "__main__":
    unittest.main()
