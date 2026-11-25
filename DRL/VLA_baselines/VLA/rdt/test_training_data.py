from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from VLA.rdt.training_data import (
    RDTBatchCollator,
    PyBulletRDTDataset,
    sha256,
    validate_lineage,
)


class _FakeProcessor:
    image_mean = [0.5, 0.5, 0.5]

    def preprocess(self, images, return_tensors="pt"):
        del return_tensors
        if not isinstance(images, list):
            images = [images]
        arrays = []
        for image in images:
            value = np.asarray(image, dtype=np.float32) / 255.0
            arrays.append(torch.from_numpy(value).permute(2, 0, 1))
        return {"pixel_values": torch.stack(arrays)}


class RDTTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        length = 3
        front = np.stack(
            [np.full((384, 384, 3), 10 + step, dtype=np.uint8) for step in range(length)]
        )
        wrist = np.stack(
            [np.full((384, 384, 3), 20 + step, dtype=np.uint8) for step in range(length)]
        )
        joints = np.array(
            [
                [0.0, 0.1, 0.2, 0.3, 1.5 * np.pi, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.5 * np.pi, 1.1],
                [1.2, 1.3, 1.4, 1.5, 1.5 * np.pi, 1.7],
            ],
            dtype=np.float32,
        )
        targets = joints + np.float32(0.01)
        np.savez_compressed(
            self.root / "touch_00000.npz",
            front_rgb=front,
            wrist_rgb=wrist,
            joint_positions=joints,
            gripper_open=np.array([1.0, 1.0, 0.0], dtype=np.float32),
            next_joint_targets=targets,
            next_gripper_closure=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            prompt=np.asarray("Touch the red block"),
        )
        self.index = [
            {
                "file": "touch_00000.npz",
                "steps": length,
                "success": True,
                "task": {"name": "touch", "prompt": "Touch the red block"},
            }
        ]
        (self.root / "index.json").write_text(json.dumps(self.index))
        self.manifest = {
            "robot": "ur5e",
            "gripper": "Robotiq 2F-85",
            "control_hz": 10,
            "image_size": 384,
            "training_view": "vla_one_shot_success",
            "required_success_steps": 1,
            "privileged_pose_in_policy_input": False,
        }
        (self.root / "dataset_manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_sparse_absolute_joint_contract_and_padding(self) -> None:
        dataset = PyBulletRDTDataset(self.root, samples_per_episode=3, seed=7)
        item = dataset[0]
        np.testing.assert_array_equal(item.step_ids, [0, 1, 2])
        self.assertEqual(item.states.shape, (3, 1, 128))
        self.assertEqual(item.actions.shape, (3, 64, 128))
        np.testing.assert_array_equal(np.flatnonzero(item.state_mask[0]), [0, 1, 2, 3, 4, 5, 10])
        self.assertAlmostEqual(float(item.states[0, 0, 4]), -0.5 * np.pi, places=5)
        np.testing.assert_allclose(item.actions[0, :3, 10], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            item.actions[0, 3:], np.repeat(item.actions[0, 2:3], 61, axis=0)
        )
        np.testing.assert_allclose(
            item.actions[2, 1:], np.repeat(item.actions[2, 0:1], 63, axis=0)
        )
        np.testing.assert_array_equal(item.action_valid_lengths, [3, 2, 1])
        # Reset history repeats t=0; subsequent samples use the actual t-1 frame.
        self.assertEqual(int(item.images[0, 0, 0, 0, 0]), 10)
        self.assertEqual(int(item.images[0, 2, 0, 0, 0]), 10)
        self.assertEqual(int(item.images[1, 0, 0, 0, 0]), 10)
        self.assertEqual(int(item.images[1, 2, 0, 0, 0]), 11)

    def test_collator_orders_six_camera_slots(self) -> None:
        dataset = PyBulletRDTDataset(self.root, samples_per_episode=3, seed=7)
        embeddings = {"Touch the red block": torch.ones((5, 4096), dtype=torch.bfloat16)}
        batch = RDTBatchCollator(_FakeProcessor(), embeddings)([dataset[0]])
        self.assertEqual(tuple(batch["images"].shape), (3, 6, 3, 384, 384))
        observed = [round(float(batch["images"][1, view, 0, 0, 0] * 255)) for view in range(6)]
        self.assertEqual(observed, [10, 20, 127, 11, 21, 127])
        self.assertEqual(tuple(batch["lang_tokens"].shape), (3, 5, 4096))
        self.assertTrue(bool(batch["lang_attn_mask"].all()))
        self.assertTrue(bool(torch.all(batch["ctrl_freqs"] == 10)))

    def test_lineage_binds_audit_to_index_and_manifest(self) -> None:
        audit_path = self.root / "audit.json"
        audit = {
            "dataset": str(self.root.resolve()),
            "index_sha256": sha256(self.root / "index.json"),
            "manifest_sha256": sha256(self.root / "dataset_manifest.json"),
            "passed": True,
            "failures": [],
            "episodes": 1,
            "frames": 3,
            "required_success_steps": 1,
            "all_oracle_success": True,
            "temporal_alignment": {
                "next_joint_to_next_observation_max_abs_error": 0.0,
                "next_gripper_to_next_observation_max_abs_error": 0.0,
            },
        }
        audit_path.write_text(json.dumps(audit))
        result = validate_lineage(self.root, audit_path)
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["frames"], 3)
        self.manifest["control_hz"] = 5
        (self.root / "dataset_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(RuntimeError):
            validate_lineage(self.root, audit_path)


if __name__ == "__main__":
    unittest.main()
