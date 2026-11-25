from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from VLA.rdt.adapter import RDTPolicy


class _Env:
    control_hz = 10

    def __init__(self):
        self.measured = np.zeros(6, dtype=np.float32)

    def joint_targets_to_canonical(self, joints, closure):
        return np.r_[joints - self.measured, 2 * closure - 1].astype(np.float32)


def _observation(step):
    return SimpleNamespace(
        front_rgb=np.full((8, 8, 3), step, dtype=np.uint8),
        wrist_rgb=np.full((8, 8, 3), step + 10, dtype=np.uint8),
        proprio=np.zeros(16, dtype=np.float32),
        prompt="Touch the red block", step=step,
    )


class RDTAdapterTest(unittest.TestCase):
    def test_queue_uses_measured_joints_and_actual_previous_control_frame(self):
        env = _Env()
        policy = RDTPolicy(env, replan_steps=2, seed=123)
        policy.server_metadata = {"already_verified": True}
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"joint_actions": [[1] * 6 + [1], [2] * 6 + [0]]},
        )
        with patch("VLA.rdt.adapter.requests.post", return_value=response) as post:
            first = policy.predict(_observation(0))
            env.measured[:] = 1
            second = policy.predict(_observation(1))
            policy.predict(_observation(2))
        np.testing.assert_array_equal(first, [1] * 6 + [-1])
        np.testing.assert_array_equal(second, [1] * 6 + [1])
        self.assertEqual(post.call_count, 2)
        request = post.call_args.kwargs["json"]
        previous = np.asarray(Image.open(BytesIO(base64.b64decode(request["previous_front_rgb_png"]))))
        np.testing.assert_array_equal(previous, np.ones((8, 8, 3), dtype=np.uint8))
        self.assertEqual(request["seed"], 123)

    def test_reset_discards_native_queue_and_camera_history(self):
        env = _Env()
        policy = RDTPolicy(env, replan_steps=2)
        policy.server_metadata = {"already_verified": True}
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"joint_actions": [[1] * 6 + [1], [2] * 6 + [0]]},
        )
        with patch("VLA.rdt.adapter.requests.post", return_value=response) as post:
            policy.predict(_observation(0))
            policy.reset(None)
            policy.predict(_observation(0))
        self.assertEqual(post.call_count, 2)
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["previous_front_rgb_png"], request["front_rgb_png"])


if __name__ == "__main__":
    unittest.main()
