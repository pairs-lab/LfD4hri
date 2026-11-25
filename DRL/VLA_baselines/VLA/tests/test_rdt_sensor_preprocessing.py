import unittest

import numpy as np
import torch

from VLA.rdt.adapter import _encode_png
from VLA.rdt.server import prepare_sensor_tensors


class FakeProcessor:
    image_mean = [.5, .5, .5]

    def preprocess(self, image, return_tensors):
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return {"pixel_values": pixels.unsqueeze(0)}


class RDTSensorPreprocessingTest(unittest.TestCase):
    def payload(self):
        return {"front_rgb_png": _encode_png(np.full((2, 2, 3), 20, dtype=np.uint8)),
                "wrist_rgb_png": _encode_png(np.full((2, 2, 3), 40, dtype=np.uint8)),
                "previous_front_rgb_png": _encode_png(np.full((2, 2, 3), 60, dtype=np.uint8)),
                "previous_wrist_rgb_png": _encode_png(np.full((2, 2, 3), 80, dtype=np.uint8)),
                "proprio": [1., 2., 3., 4., 5., 6., .6] + [999.] * 9,
                "prompt": "Touch the red block", "step": 5}

    def test_true_camera_history_order_and_sparse_state(self):
        embeddings = {"Touch the red block": torch.ones((3, 8), dtype=torch.bfloat16)}
        batch = prepare_sensor_tensors(self.payload(), FakeProcessor(), embeddings, target_domain=True)
        values = batch["images"][0, :, 0, 0, 0] * 255
        torch.testing.assert_close(values, torch.tensor([60., 80., 127., 20., 40., 127.]))
        self.assertEqual(batch["states"].shape, (1, 1, 128))
        self.assertEqual(batch["states"][0, 0, 10].item(), np.float32(.6))
        self.assertEqual(batch["state_elem_mask"].sum().item(), 7)
        self.assertEqual(batch["ctrl_freqs"].item(), 10)
        self.assertTrue(batch["lang_attn_mask"].all())

    def test_eef_fields_not_advertised_as_incompatible_rdt_state(self):
        payload = self.payload()
        embeddings = {payload["prompt"]: torch.ones((3, 8))}
        before = prepare_sensor_tensors(payload, FakeProcessor(), embeddings, target_domain=True)
        payload["proprio"][7:] = [-123.] * 9
        after = prepare_sensor_tensors(payload, FakeProcessor(), embeddings, target_domain=True)
        torch.testing.assert_close(before["states"], after["states"], rtol=0, atol=0)

    def test_wrong_proprio_shape_fails_closed(self):
        payload = self.payload()
        payload["proprio"] = [0.] * 7
        with self.assertRaises(ValueError):
            prepare_sensor_tensors(payload, FakeProcessor(), {payload["prompt"]: torch.ones((3, 8))}, target_domain=True)


if __name__ == "__main__":
    unittest.main()
