from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from VLA.pi0.adapter import OpenPiClientPolicy
from VLA.pi0.server import _RequestNoisePolicy


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict] = []

    def infer(self, request: dict) -> dict:
        self.calls += 1
        self.requests.append(request)
        offset = 10 * self.calls
        return {
            "actions": np.asarray(
                [
                    [offset + 1] * 6 + [0.1],
                    [offset + 2] * 6 + [0.2],
                    [offset + 3] * 6 + [0.3],
                ],
                dtype=np.float32,
            )
        }


class _FakeEnv:
    robot_spec = SimpleNamespace(name="ur5e")

    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []

    def normalized_gripper_closure(self) -> float:
        return 0.0

    def joint_targets_to_canonical(self, targets: np.ndarray, closure: float) -> np.ndarray:
        self.targets.append(np.r_[targets, closure])
        return np.r_[targets[:6], closure].astype(np.float32)


def _observation(step: int = 0):
    return SimpleNamespace(
        front_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        joint_positions=np.zeros(6, dtype=np.float32),
        prompt="Touch the red block",
        step=step,
    )


class OpenPiChunkQueueTest(unittest.TestCase):
    def test_runtime_horizon_controls_diffusion_noise_shape(self) -> None:
        policy = OpenPiClientPolicy(_FakeEnv(), "pi0", replan_steps=2)
        policy._validate_server_metadata({
            "protocol": "pybullet_ur5e_robotiq_openpi_v1", "model": "pi0",
            "target_domain": False, "action_horizon": 5, "action_dim": 32,
        })
        client = _FakeClient()
        policy.client = client
        policy.predict(_observation())
        self.assertEqual(client.requests[0]["__policy_noise__"].shape, (5, 32))

    def test_wrong_model_and_undeclared_target_checkpoint_are_rejected(self) -> None:
        policy = OpenPiClientPolicy(_FakeEnv(), "pi0")
        metadata = {
            "protocol": "pybullet_ur5e_robotiq_openpi_v1", "model": "pi0_fast",
            "target_domain": False, "action_horizon": 10, "action_dim": 7,
        }
        with self.assertRaises(RuntimeError):
            policy._validate_server_metadata(metadata)
        metadata.update(model="pi0", target_domain=True, action_dim=32)
        with self.assertRaises(RuntimeError):
            policy._validate_server_metadata(metadata)

    def test_native_chunk_targets_are_converted_lazily(self) -> None:
        env = _FakeEnv()
        policy = OpenPiClientPolicy(env, "pi0", replan_steps=2)
        client = _FakeClient()
        policy.client = client

        first = policy.predict(_observation(0))
        second = policy.predict(_observation(1))
        third = policy.predict(_observation(2))

        self.assertEqual(client.calls, 2)
        np.testing.assert_allclose(first, [11, 11, 11, 11, 11, 11, 0.1])
        np.testing.assert_allclose(second, [12, 12, 12, 12, 12, 12, 0.2])
        np.testing.assert_allclose(third, [21, 21, 21, 21, 21, 21, 0.1])
        np.testing.assert_allclose(env.targets[1], [12, 12, 12, 12, 12, 12, 0.2])
        self.assertEqual(client.requests[0]["__policy_noise__"].shape, (10, 32))

    def test_reset_discards_old_native_chunk(self) -> None:
        env = _FakeEnv()
        policy = OpenPiClientPolicy(env, "pi0", replan_steps=3)
        client = _FakeClient()
        policy.client = client

        policy.predict(_observation())
        policy.reset(SimpleNamespace())
        action = policy.predict(_observation())

        self.assertEqual(client.calls, 2)
        np.testing.assert_allclose(action, [21, 21, 21, 21, 21, 21, 0.1])

    def test_diffusion_noise_repeats_after_episode_reset(self) -> None:
        env = _FakeEnv()
        policy = OpenPiClientPolicy(env, "pi0", replan_steps=1, seed=123)
        client = _FakeClient()
        policy.client = client

        policy.predict(_observation())
        first_noise = client.requests[-1]["__policy_noise__"].copy()
        policy.reset(SimpleNamespace())
        policy.predict(_observation())
        second_noise = client.requests[-1]["__policy_noise__"]

        np.testing.assert_array_equal(first_noise, second_noise)


class _FakeServerPolicy:
    metadata = {"name": "fake"}

    def __init__(self) -> None:
        self.observation = None
        self.noise = None

    def infer(self, observation: dict, *, noise=None) -> dict:
        self.observation = observation
        self.noise = noise
        return {"actions": np.zeros((10, 7), dtype=np.float32)}


class RequestNoisePolicyTest(unittest.TestCase):
    def test_noise_is_not_exposed_as_a_model_observation(self) -> None:
        underlying = _FakeServerPolicy()
        policy = _RequestNoisePolicy(underlying)
        noise = np.ones((10, 32), dtype=np.float32)

        result = policy.infer({"prompt": "Touch the red block", "__policy_noise__": noise})

        self.assertEqual(policy.metadata, {"name": "fake"})
        self.assertEqual(set(underlying.observation), {"prompt"})
        np.testing.assert_array_equal(underlying.noise, noise)
        self.assertEqual(result["actions"].shape, (10, 7))


if __name__ == "__main__":
    unittest.main()
