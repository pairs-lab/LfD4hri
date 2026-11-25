"""Client for an isolated RDT inference server.

RDT requires its own pinned PyTorch/Transformers/DeepSpeed environment and very
large T5/SigLIP encoders.  The server contract intentionally returns canonical
delta-EEF actions so simulator code remains model-independent.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image
import requests

from VLA.common.observation import VLAObservation
from VLA.common.environment import ManipulationEnv
from VLA.common.policy import Policy


def _encode_png(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RDTPolicy(Policy):
    name = "rdt"
    input_modalities = (
        "front_rgb",
        "wrist_rgb",
        "joint_positions",
        "gripper_state",
        "text",
    )

    def __init__(
        self,
        env: ManipulationEnv,
        host: str = "127.0.0.1",
        port: int | None = None,
        checkpoint: str | None = None,
        unnorm_key: str | None = None,
        replan_steps: int = 1,
        seed: int = 0,
        **_kwargs,
    ) -> None:
        self.env = env
        self.url = f"http://{host}:{5600 if port is None else port}/act"
        self.checkpoint = checkpoint
        self.unnorm_key = unnorm_key
        if replan_steps < 1:
            raise ValueError("replan_steps must be at least one")
        self.replan_steps = int(replan_steps)
        self._native_queue: list[np.ndarray] = []
        self.seed = int(seed)
        self.server_metadata: dict = {}
        self._previous_observation: VLAObservation | None = None

    def _verify_server(self) -> None:
        if self.server_metadata:
            return
        from pathlib import Path
        from VLA.fingerprint_checkpoint import verify_fingerprint

        response = requests.get(self.url.rsplit("/", 1)[0] + "/metadata", timeout=30)
        response.raise_for_status()
        metadata = response.json()
        if metadata.get("protocol") != "pybullet_ur5e_robotiq_rdt_v1" or metadata.get("model") != "rdt":
            raise RuntimeError("RDT server model/contract mismatch")
        if metadata.get("target_domain"):
            if self.checkpoint is None:
                raise RuntimeError("Target-domain RDT evaluation requires --checkpoint")
            root = Path(self.checkpoint).resolve()
            fingerprint = verify_fingerprint(root)
            served = metadata.get("checkpoint", {})
            if (
                Path(served.get("path", "")).resolve() != root
                or served.get("content_tree_sha256") != fingerprint["content_tree_sha256"]
                or metadata.get("control_hz") != self.env.control_hz
            ):
                raise RuntimeError("RDT server checkpoint/control frequency differs from evaluation")
        self.server_metadata = metadata

    def reset(self, task) -> None:
        del task
        self._native_queue.clear()
        self._previous_observation = None

    def build_sensor_request(self, observation, previous) -> dict:
        """Exact deployed sensor payload, also reused by offline input audits."""
        return {
            "front_rgb_png": _encode_png(observation.front_rgb),
            "wrist_rgb_png": _encode_png(observation.wrist_rgb),
            "previous_front_rgb_png": _encode_png(previous.front_rgb),
            "previous_wrist_rgb_png": _encode_png(previous.wrist_rgb),
            "proprio": observation.proprio.tolist(),
            "prompt": observation.prompt, "step": observation.step, "seed": self.seed,
        }

    def predict(self, observation: VLAObservation) -> np.ndarray:
        self._verify_server()
        previous = self._previous_observation or observation
        self._previous_observation = observation
        if self._native_queue:
            action = self._native_queue.pop(0)
            return self.env.joint_targets_to_canonical(action[:6], 1.0 - action[6])
        response = requests.post(
            self.url,
            json=self.build_sensor_request(observation, previous),
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        if "joint_actions" in payload:
            joint_actions = np.asarray(payload["joint_actions"], dtype=np.float32)
            if joint_actions.ndim == 1:
                joint_actions = joint_actions[None, :]
            if joint_actions.ndim != 2 or joint_actions.shape[1] != 7 or not np.isfinite(joint_actions).all():
                raise ValueError(
                    f"RDT server returned invalid joint actions {joint_actions.shape}, expected finite (H, 7)"
                )
            # Server convention is six absolute joints plus gripper-open in
            # [0,1]; the shared converter consumes gripper closure.
            if not np.isfinite(joint_actions).all() or len(joint_actions) < 1:
                raise ValueError("RDT returned empty/non-finite native actions")
            count = min(self.replan_steps, len(joint_actions))
            self._native_queue.extend(action.copy() for action in joint_actions[:count])
            action = self._native_queue.pop(0)
            return self.env.joint_targets_to_canonical(action[:6], 1.0 - action[6])
        actions = np.asarray(payload.get("actions", payload), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"RDT server returned {actions.shape}, expected (H, 7)")
        return actions
