"""Run a target-domain SmolVLA checkpoint in the common PyBullet action space."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from VLA.common.environment import ManipulationEnv
from VLA.common.observation import VLAObservation
from VLA.common.policy import Policy
from VLA.common.tasks import TaskSpec


class SmolVLAPolicyAdapter(Policy):
    name = "smolvla"
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
        checkpoint: str | None = None,
        replan_steps: int = 1,
        seed: int = 0,
        **_kwargs,
    ) -> None:
        if checkpoint is None:
            raise ValueError("SmolVLA evaluation requires --checkpoint pointing to a fine-tuned checkpoint")
        self.env = env
        self.checkpoint = checkpoint
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.replan_steps = int(replan_steps)
        self.seed = int(seed)
        self.model = None
        self.preprocess = None
        self.postprocess = None
        self._task_prompt: str | None = None
        self._noise_generator = None
        self._fp32_masters = False

    def _task_noise_seed(self, prompt: str) -> int:
        """Stable per-task seed without mutating PyTorch's global RNG."""

        payload = f"smolvla\0{self.seed}\0{prompt}".encode("utf-8")
        # torch.Generator.manual_seed accepts signed 64-bit-compatible values.
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)

    def _reset_noise_generator(self) -> None:
        if self.model is None or self._task_prompt is None:
            self._noise_generator = None
            return
        import torch

        device = next(self.model.parameters()).device
        self._noise_generator = torch.Generator(device=device)
        self._noise_generator.manual_seed(self._task_noise_seed(self._task_prompt))

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        self._fp32_masters = (Path(self.checkpoint) / "fp32_master_contract.json").is_file()
        if not self._fp32_masters:
            from safetensors import safe_open

            with safe_open(Path(self.checkpoint) / "model.safetensors", framework="np") as saved:
                anchor = "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
                if saved.get_slice(anchor).get_dtype() == "F32":
                    raise RuntimeError("Finalize the FP32 master contract before inference; refusing a lossy BF16 reload")
        if self._fp32_masters:
            from VLA.smolvla.precision import fp32_policy_class

            contract = json.loads((Path(self.checkpoint) / "fp32_master_contract.json").read_text())
            if not contract.get("passed") or not contract.get("saved_weights_and_adam_moments_fp32"):
                raise RuntimeError("SmolVLA FP32 master contract failed")
            policy_class = fp32_policy_class(contract.get("train_visual_connector", False))
        else:
            # Preserve historical BF16 diagnostics; do not silently change
            # their inference numerics after discovering the training defect.
            policy_class = SmolVLAPolicy
        self.model = policy_class.from_pretrained(self.checkpoint).to("cuda").eval()
        if self._fp32_masters and any(p.dtype != torch.float32 for p in self.model.parameters() if p.requires_grad):
            raise RuntimeError("SmolVLA inference loader downcast learned FP32 weights")
        # The released config defaults to consuming all 50 predicted actions
        # open-loop.  Our demonstrations are only 18--36 steps long and the
        # common benchmark replans every control step for other VLAs.  Limit
        # the action queue so the policy uses the same 10 Hz visual feedback.
        self.model.config.n_action_steps = self.replan_steps
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self._reset_noise_generator()

    def reset(self, task: TaskSpec) -> None:
        self._task_prompt = task.prompt
        if self.model is not None:
            self.model.reset()
        if self.preprocess is not None:
            self.preprocess.reset()
        if self.postprocess is not None:
            self.postprocess.reset()
        self._reset_noise_generator()

    def build_sensor_batch(self, observation: VLAObservation) -> dict:
        """Build the deployed sensor batch; also reused by train/infer audits."""
        import torch

        state = np.concatenate(
            (
                observation.joint_positions.astype(np.float32),
                np.asarray((self.env.normalized_gripper_closure(),), dtype=np.float32),
            )
        )
        return {
            # The official base checkpoint names its visual slots camera1/2/3.
            # Training maps our semantic front/wrist names to the first two;
            # camera3 is absent and SmolVLA ignores that unpopulated slot.
            "observation.images.camera1": torch.from_numpy(observation.front_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0),
            "observation.images.camera2": torch.from_numpy(observation.wrist_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0),
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [observation.prompt],
        }

    def predict(self, observation: VLAObservation) -> np.ndarray:
        self._load()
        import torch

        batch = self.build_sensor_batch(observation)
        precision_context = torch.autocast("cuda", dtype=torch.bfloat16) if self._fp32_masters else nullcontext()
        with torch.inference_mode(), precision_context:
            processed = self.preprocess(batch)
            # SmolVLA is a flow-matching policy and otherwise samples fresh
            # global-RNG noise for every predicted chunk.  Use a private,
            # benchmark-seeded generator so repeated protocol runs are exactly
            # reproducible without perturbing any other torch consumer.  Do not
            # advance it while queued open-loop actions are being consumed.
            noise = None
            if self.model._check_get_actions_condition():
                if self._noise_generator is None:
                    self._task_prompt = observation.prompt
                    self._reset_noise_generator()
                state = processed["observation.state"]
                noise = torch.randn(
                    (
                        state.shape[0],
                        self.model.config.chunk_size,
                        self.model.config.max_action_dim,
                    ),
                    dtype=torch.float32,
                    device=state.device,
                    generator=self._noise_generator,
                )
            action = self.model.select_action(processed, noise=noise)
            action = self.postprocess(action)
        values = action.squeeze(0).float().cpu().numpy()
        if values.shape != (7,):
            raise ValueError(f"SmolVLA returned {values.shape}; expected (7,)")
        return values.astype(np.float32)
