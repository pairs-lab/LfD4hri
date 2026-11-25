"""openpi websocket client for the shared UR5e PyBullet benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from VLA.common.environment import ManipulationEnv
from VLA.common.observation import VLAObservation
from VLA.common.policy import Policy
from VLA.common.tasks import TaskSpec


class OpenPiClientPolicy(Policy):
    input_modalities = ("front_rgb", "wrist_rgb", "joint_positions", "gripper_state", "text")
    def __init__(
        self,
        env: ManipulationEnv,
        model_name: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        checkpoint: str | None = None,
        unnorm_key: str | None = None,
        replan_steps: int = 1,
        seed: int = 0,
        **_kwargs,
    ) -> None:
        if env.robot_spec.name != "ur5e":
            raise ValueError("Official openpi UR5 normalization is only defensible with --robot ur5e")
        self.env = env
        self.name = model_name
        self.host = host
        self.port = 8000 if port is None else port
        self.checkpoint = checkpoint
        self.unnorm_key = unnorm_key
        if replan_steps < 1:
            raise ValueError("replan_steps must be at least one")
        self.replan_steps = int(replan_steps)
        self.seed = int(seed)
        self.client = None
        self.task: TaskSpec | None = None
        self._native_queue: list[np.ndarray] = []
        self._inference_index = 0
        self.action_horizon = 10
        self.action_dim = 32
        self.server_metadata: dict = {}

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        self._native_queue.clear()
        self._inference_index = 0

    def _connect(self) -> None:
        if self.client is None:
            from openpi_client import websocket_client_policy

            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=self.host, port=self.port
            )
            self._validate_server_metadata(self.client.get_server_metadata())

    def _validate_server_metadata(self, metadata: dict) -> None:
        if (
            metadata.get("protocol") != "pybullet_ur5e_robotiq_openpi_v1"
            or metadata.get("model") != self.name
        ):
            raise RuntimeError("openpi server model/contract does not match this benchmark")
        horizon = int(metadata.get("action_horizon", 0))
        action_dim = int(metadata.get("action_dim", 0))
        if horizon < 1 or action_dim < 7 or self.replan_steps > horizon:
            raise RuntimeError("openpi server action shape is incompatible with replan_steps")
        if self.unnorm_key is not None and metadata.get("repo_id") != self.unnorm_key:
            raise RuntimeError("openpi server serves a different target dataset/normalization")
        if metadata.get("target_domain") and self.checkpoint is None:
            raise RuntimeError("Target-domain evaluation must declare --checkpoint")
        if self.checkpoint is not None:
            expected_path = Path(self.checkpoint)
            served = metadata.get("checkpoint", {})
            if expected_path.is_dir():
                from VLA.fingerprint_checkpoint import layout_records, layout_sha256, verify_fingerprint

                root = expected_path.resolve()
                fingerprint_path = root / "benchmark_fingerprint.json"
                if not fingerprint_path.is_file():
                    raise FileNotFoundError(fingerprint_path)
                fingerprint = verify_fingerprint(root)
                if (
                    Path(served.get("path", "")).resolve() != root
                    or fingerprint.get("layout_sha256")
                    != layout_sha256(layout_records(root))
                    or served.get("content_tree_sha256")
                    != fingerprint.get("content_tree_sha256")
                ):
                    raise RuntimeError("openpi server weights differ from the declared checkpoint")
            elif str(served.get("path")) != self.checkpoint:
                raise RuntimeError("openpi server checkpoint does not match the requested base model")
        self.action_horizon = horizon
        self.action_dim = action_dim
        self.server_metadata = dict(metadata)

    def predict(self, observation: VLAObservation) -> np.ndarray:
        self._connect()
        from openpi_client import image_tools

        if not self._native_queue:
            base = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation.front_rgb, 224, 224)
            )
            wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation.wrist_rgb, 224, 224)
            )
            request = {
                    "base_rgb": base,
                    "wrist_rgb": wrist,
                    # PyBullet may report equivalent revolute angles outside
                    # [-pi, pi] (the home wrist joint is 3*pi/2).  openpi's
                    # UR5e state statistics use the wrapped convention.
                    "joints": (
                        (observation.joint_positions + np.pi) % (2.0 * np.pi) - np.pi
                    ).astype(np.float32),
                    "gripper": np.array(
                        (self.env.normalized_gripper_closure(),), dtype=np.float32
                    ),
                    "prompt": observation.prompt,
                }
            if self.name == "pi0":
                # Flow matching samples from standard Gaussian noise.  Send a
                # deterministic draw so rerunning the same benchmark seed is
                # bit-reproducible and independent of requests previously
                # handled by the long-lived server.  This changes only the RNG
                # realization, not its distribution or any policy input.
                prompt_key = int.from_bytes(
                    hashlib.sha256(observation.prompt.encode("utf-8")).digest()[:8],
                    byteorder="little",
                )
                rng = np.random.default_rng(
                    np.random.SeedSequence((self.seed, prompt_key, self._inference_index))
                )
                request["__policy_noise__"] = rng.standard_normal(
                    (self.action_horizon, self.action_dim), dtype=np.float32
                )
            response = self.client.infer(request)
            self._inference_index += 1
            native_actions = np.asarray(response["actions"], dtype=np.float32)
            if native_actions.ndim == 1:
                native_actions = native_actions[None, :]
            if native_actions.ndim != 2 or native_actions.shape[1] < 7:
                raise ValueError(f"openpi returned {native_actions.shape}, expected (H, >=7)")
            if not np.isfinite(native_actions).all():
                raise ValueError("openpi returned non-finite native actions")
            count = min(self.replan_steps, len(native_actions))
            self._native_queue.extend(action.copy() for action in native_actions[:count])

        # openpi returns *absolute* future joint targets after its output
        # transform.  Convert only the target being executed, using the latest
        # measured joints.  Converting an entire chunk at prediction time
        # applies every future target relative to the same stale state and can
        # turn a correct chunk into cumulative overshoot.
        native_action = self._native_queue.pop(0)
        return self.env.joint_targets_to_canonical(native_action[:6], native_action[6])
