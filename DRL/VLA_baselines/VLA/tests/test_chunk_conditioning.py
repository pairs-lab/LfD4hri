from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from VLA.common.tasks import TASKS
from VLA.smolvla.audit_conditioning import audit_pair, pair_chunks


class ChunkConditioningTest(unittest.TestCase):
    def test_shared_first_action_does_not_hide_later_conditioning(self):
        policy = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(chunk_size=3)))
        actions = [np.zeros((3, 7), dtype=np.float32), np.zeros((3, 7), dtype=np.float32)]
        actions[0][1:, 0] = .03
        actions[1][1:, 0] = -.03
        with TemporaryDirectory() as folder:
            source = Path(folder)
            items = []
            for i, action in enumerate(actions):
                task = {**TASKS["touch"].to_dict(), "prompt": f"target{i}"}
                file = f"variant{i}.npz"
                np.savez(source / file, front_rgb=np.zeros((3, 2, 2, 3), dtype=np.uint8),
                         wrist_rgb=np.zeros((3, 2, 2, 3), dtype=np.uint8),
                         joint_positions=np.zeros((3, 6)), gripper_closure=np.zeros(3),
                         gripper_open=np.ones(3), ee_position=np.zeros((3, 3)),
                         ee_rotation_6d=np.zeros((3, 6)), canonical_actions=action)
                items.append({"task": task, "file": file, "variant": i})
            def predict(task, observation, closure, seed):
                return actions[1 if task.prompt == "target1" else 0]
            rows = audit_pair(policy, predict, "touch", "scene", items, source)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["teacher_text_pair_translation_difference_m"], 0)
        self.assertGreater(rows[0]["teacher_chunk_text_translation_difference_rmse_m"], .01)
        self.assertGreater(rows[0]["model_chunk_text_translation_difference_rmse_m"], .01)

    def test_chunk_comparison_handles_different_episode_lengths(self):
        a, b = pair_chunks(np.zeros((3, 7)), np.zeros((2, 7)))
        self.assertEqual(a.shape, (2, 7))
        self.assertEqual(b.shape, (2, 7))
        with self.assertRaises(ValueError):
            pair_chunks([], [])


if __name__ == "__main__":
    unittest.main()
