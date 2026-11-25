import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from VLA.benchmark import _observation_telemetry, _save_observation_snapshot
from VLA.common.observation import VLAObservation


class ObservationSnapshotTest(unittest.TestCase):
    def test_snapshot_is_exact_sensor_rgb(self):
        rng = np.random.default_rng(4)
        observation = VLAObservation(
            front_rgb=rng.integers(0, 256, (12, 12, 3), dtype=np.uint8),
            wrist_rgb=rng.integers(0, 256, (12, 12, 3), dtype=np.uint8),
            joint_positions=np.arange(6, dtype=np.float32), gripper_width=.04,
            gripper_open=.5, ee_position=np.zeros(3), ee_rotation_6d=np.zeros(6),
            prompt="Pick the red block up", step=20,
        )
        env = SimpleNamespace(normalized_gripper_closure=lambda: .75)
        with tempfile.TemporaryDirectory() as directory:
            _save_observation_snapshot(Path(directory), "pick", observation, env)
            root = Path(directory) / "observations/pick"
            for camera in ("front", "wrist"):
                pixels = np.asarray(Image.open(root / f"step_0020_{camera}.png"))
                np.testing.assert_array_equal(pixels, getattr(observation, camera + "_rgb"))
            metadata = json.loads((root / "step_0020.json").read_text())
            self.assertEqual(metadata["gripper_command_closure"], .75)
            self.assertEqual(metadata["prompt"], observation.prompt)
            self.assertNotIn("goal_position", metadata)
            self.assertNotIn("object_position", metadata)
        self.assertEqual(_observation_telemetry(observation, env)["step"], 20)


if __name__ == "__main__":
    unittest.main()
