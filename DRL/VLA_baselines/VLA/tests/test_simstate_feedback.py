import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from VLA.simstate_feedback import FEATURE_NAMES, feedback_state, state_stats
from VLA.smolvla.simstate_dataset import SimStateDataset, load_projection


def fixture(held=True, bilateral=True, command=1.):
    grip = SimpleNamespace(grasped_object=5 if held else None, _bilateral=lambda _body: bilateral)
    env = SimpleNamespace(pick_id=5, gripper=grip, normalized_gripper_closure=lambda: .6,
        commanded_gripper_closure=lambda: command,
        debug_state=lambda: {"grasp_center": [.1, -.4, .2], "pick_center": [.2, -.3, .04],
                            "pick_position": [.2, -.3, .02], "goal_position": [.25, -.25, .1]})
    obs = SimpleNamespace(joint_positions=np.arange(6, dtype=np.float32))
    return env, obs


class SimStateFeedbackTests(unittest.TestCase):
    def test_18_causal_features_no_future_action_or_teacher_phase(self):
        env, obs = fixture()
        values = feedback_state(env, obs)
        self.assertEqual(values.shape, (18,))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(len(FEATURE_NAMES), 18)
        self.assertFalse(any("next_" in name or "teacher" in name for name in FEATURE_NAMES))
        np.testing.assert_allclose(values[11:14], [.1, .1, -.16], atol=1e-7)
        np.testing.assert_allclose(values[14:17], [.05, .05, .08], atol=1e-7)

    def test_command_is_not_thresholded_encoder_width(self):
        env, obs = fixture(command=0.)
        values = feedback_state(env, obs)
        self.assertAlmostEqual(float(values[6]), .6)
        self.assertEqual(values[10], 0.)

    def test_wrong_body_or_unilateral_contact_is_not_verified_hold(self):
        env, obs = fixture(bilateral=False)
        self.assertEqual(feedback_state(env, obs)[17], 0.)
        env.gripper.grasped_object = 9
        env.gripper._bilateral = lambda _: True
        self.assertEqual(feedback_state(env, obs)[17], 0.)

    def test_correct_bilateral_is_feedback_not_a_fake_action(self):
        env, obs = fixture()
        self.assertEqual(feedback_state(env, obs)[17], 1.)
        self.assertEqual(feedback_state(env, obs).shape, (18,))

    def test_numeric_validation_rejects_nonfinite_input(self):
        env, obs = fixture()
        obs.joint_positions[0] = np.nan
        with self.assertRaises(ValueError):
            feedback_state(env, obs)

    def test_stats_use_complete_float64_train_values(self):
        values = np.arange(54, dtype=np.float32).reshape(3, 18)
        stats = state_stats(values)
        self.assertEqual(stats["count"], [3])
        np.testing.assert_array_equal(stats["mean"], values.astype(np.float64).mean(0))

    def test_pilot_projection_cannot_silently_train(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "audit.json").write_text(json.dumps({"passed": True, "pilot_only": True}))
            with self.assertRaises(ValueError):
                load_projection(directory)

    def test_wrapper_keeps_base_pixels_labels_meta_and_indices_unchanged(self):
        values = np.zeros((2, 18), dtype=np.float32)
        values[:, :6] = np.arange(6)
        class Base:
            meta = SimpleNamespace(info=SimpleNamespace(features={"observation.state": {"shape": [7]}}),
                                   stats={"observation.state": {"mean": np.zeros(7)}})
            def __len__(self): return 2
            def __getitem__(self, index):
                return {"index": torch.tensor(index), "observation.state": torch.from_numpy(values[index, :7]),
                        "action": torch.ones(5, 7), "pixels": "unchanged"}
        base = Base()
        with patch("VLA.smolvla.simstate_dataset.load_projection", return_value=(values, {}, state_stats(values))):
            wrapped = SimStateDataset(base, ".")
        self.assertEqual(base.meta.info.features["observation.state"]["shape"], [7])
        self.assertEqual(wrapped.meta.info.features["observation.state"]["shape"], [18])
        self.assertEqual(wrapped[0]["pixels"], "unchanged")
        self.assertEqual(wrapped[0]["action"].shape, (5, 7))
        self.assertEqual(wrapped[0]["observation.state"].shape, (18,))

    def test_actual_dataloader_cannot_delegate_to_the_state7_fast_path(self):
        values = np.arange(54, dtype=np.float32).reshape(3, 18)
        class Base:
            meta = SimpleNamespace(info=SimpleNamespace(features={"observation.state": {"shape": [7]}}),
                                   stats={"observation.state": {"mean": np.zeros(7)}})
            fast_calls = 0
            def __len__(self): return 3
            def __getitem__(self, index):
                return {"index":torch.tensor(index), "observation.state":torch.from_numpy(values[index,:7]).unsqueeze(0),
                        "action":torch.ones(5,7)*index}
            def __getitems__(self, indices):
                self.fast_calls += 1
                return [self[index] for index in indices]
        base = Base()
        with patch("VLA.smolvla.simstate_dataset.load_projection", return_value=(values, {}, state_stats(values))):
            wrapped = SimStateDataset(base, ".")
        batches = list(torch.utils.data.DataLoader(wrapped,batch_size=2,shuffle=False,num_workers=0))
        self.assertEqual(base.fast_calls,0)
        self.assertEqual(batches[0]["observation.state"].shape,(2,1,18))
        torch.testing.assert_close(batches[0]["observation.state"][:,0],torch.from_numpy(values[:2]))
        self.assertEqual(batches[0]["action"].shape,(2,5,7))
        self.assertEqual(batches[1]["observation.state"].shape,(1,1,18))


if __name__ == "__main__":
    unittest.main()
