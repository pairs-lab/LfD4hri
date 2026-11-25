"""Thin client for CogACT's official Flask deployment server."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import requests

from VLA.common.observation import VLAObservation
from VLA.common.policy import Policy


class CogACTPolicy(Policy):
    name = "cogact"
    input_modalities = ("front_rgb", "text")

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        checkpoint: str | None = None,
        unnorm_key: str | None = None,
        replan_steps: int = 1,
        seed: int = 0,
        **_kwargs,
    ) -> None:
        self.url = f"http://{host}:{5500 if port is None else port}/api/inference"
        self.checkpoint = checkpoint
        self.unnorm_key = unnorm_key
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.replan_steps = int(replan_steps)
        self.seed = int(seed)
        self._queue: list[np.ndarray] = []
        self._inference_index = 0
        self.server_metadata: dict = {}

    def _validate_server_metadata(self, metadata: dict) -> None:
        if metadata.get("protocol") != "pybullet_ur5e_robotiq_cogact_v1" or metadata.get("model") != "cogact":
            raise RuntimeError("Incompatible CogACT policy server")
        if metadata.get("action_dim") != 7 or metadata.get("gripper_convention") != "1=open,0=closed":
            raise RuntimeError("CogACT server action contract differs from the benchmark")
        served = metadata.get("checkpoint")
        if self.checkpoint is not None:
            expected = self.checkpoint
            if Path(expected).is_file():
                expected = str(Path(expected).resolve())
            if served != expected:
                raise RuntimeError("CogACT server checkpoint differs from evaluation checkpoint")
        served_key = metadata.get("unnorm_key")
        if not isinstance(served_key, str) or not served_key:
            raise RuntimeError("CogACT server did not declare action normalization")
        if self.unnorm_key is not None and self.unnorm_key != served_key:
            raise RuntimeError("CogACT server normalization differs from evaluation normalization")
        if int(metadata.get("action_horizon", 0)) < self.replan_steps:
            raise RuntimeError("CogACT replan steps exceed the served action horizon")
        self.server_metadata = metadata

    def _connect(self) -> None:
        if not self.server_metadata:
            response = requests.get(self.url.rsplit("/", 1)[0] + "/metadata", timeout=30)
            response.raise_for_status()
            self._validate_server_metadata(response.json())

    def reset(self, task) -> None:
        del task
        self._queue.clear()
        self._inference_index = 0

    def predict(self, observation: VLAObservation) -> np.ndarray:
        self._connect()
        if self._queue:
            return self._queue.pop(0)
        image_buffer = BytesIO()
        Image.fromarray(observation.front_rgb).save(image_buffer, format="PNG")
        image_buffer.seek(0)
        query = BytesIO(
            json.dumps({
                "task_description": observation.prompt,
                "seed": self.seed,
                "step": observation.step,
                "inference_index": self._inference_index,
                "unnorm_key": self.server_metadata["unnorm_key"],
            }).encode("utf-8")
        )
        response = requests.post(
            self.url,
            files={
                "images": ("front.png", image_buffer, "image/png"),
                "json": ("query.json", query, "application/json"),
            },
            timeout=600,
        )
        response.raise_for_status()
        self._inference_index += 1
        native = np.asarray(response.json(), dtype=np.float32)
        if native.ndim == 1:
            native = native[None, :]
        if native.ndim != 2 or native.shape[1] != 7:
            raise ValueError(f"CogACT returned {native.shape}, expected (H, 7)")
        if not len(native) or not np.isfinite(native).all():
            raise ValueError("CogACT returned empty/non-finite native actions")
        if len(native) < self.replan_steps:
            raise ValueError("CogACT action chunk is shorter than requested replan steps")
        canonical = native[:, :7].copy()
        # CogACT follows OXE/SIMPLER: 1=open, 0=closed.
        canonical[:, 6] = 1.0 - 2.0 * canonical[:, 6]
        self._queue.extend(action.copy() for action in canonical[:self.replan_steps])
        return self._queue.pop(0)
