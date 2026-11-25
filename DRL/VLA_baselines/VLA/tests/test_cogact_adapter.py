from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from VLA.cogact.adapter import CogACTPolicy


def metadata():
    return {
        "protocol": "pybullet_ur5e_robotiq_cogact_v1", "model": "cogact",
        "checkpoint": "CogACT/CogACT-Small", "unnorm_key": "pybullet",
        "action_dim": 7, "action_horizon": 16,
        "gripper_convention": "1=open,0=closed",
    }


def observation():
    return SimpleNamespace(front_rgb=np.full((8, 8, 3), 73, dtype=np.uint8),
                           prompt="Pick the red block", step=0)


class CogACTContractTest(unittest.TestCase):
    def test_wrong_checkpoint_normalization_and_convention_are_rejected(self):
        policy = CogACTPolicy(checkpoint="CogACT/CogACT-Small", unnorm_key="pybullet")
        for key, value in (("checkpoint", "other"), ("unnorm_key", "berkeley"),
                           ("gripper_convention", "1=closed"), ("action_dim", 8),
                           ("action_horizon", 0), ("model", "rdt")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                policy._validate_server_metadata({**metadata(), key: value})

    def test_chunk_gripper_reset_and_no_privileged_inputs(self):
        policy = CogACTPolicy(checkpoint="CogACT/CogACT-Small", unnorm_key="pybullet",
                              replan_steps=2, seed=17)
        policy._validate_server_metadata(metadata())
        sent = []

        def post(url, files, timeout):
            del url, timeout
            query = json.loads(files["json"][1].getvalue())
            sent.append(query)
            np.testing.assert_array_equal(
                np.asarray(Image.open(BytesIO(files["images"][1].getvalue()))),
                observation().front_rgb)
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: [[.01, .02, .03, 0, 0, 0, 1], [.04, 0, 0, 0, 0, 0, 0]])

        with patch("VLA.cogact.adapter.requests.post", side_effect=post):
            first = policy.predict(observation())
            second = policy.predict(observation())
            self.assertEqual(len(sent), 1)
            self.assertEqual(first[6], -1)
            self.assertEqual(second[6], 1)
            self.assertAlmostEqual(float(first[0]), .01)
            policy.reset(None)
            policy.predict(observation())
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(set(sent[0]), {"task_description", "seed", "step", "inference_index", "unnorm_key"})

    def test_nan_and_short_chunk_are_rejected(self):
        policy = CogACTPolicy(replan_steps=2)
        policy._validate_server_metadata(metadata())
        for native in ([[0] * 6 + [float("nan")]] * 2, [[0] * 7], [[0] * 8] * 2):
            response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: native)
            with patch("VLA.cogact.adapter.requests.post", return_value=response), self.assertRaises(ValueError):
                policy.predict(observation())


if __name__ == "__main__":
    unittest.main()
