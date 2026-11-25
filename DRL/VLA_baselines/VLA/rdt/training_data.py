"""Audited PyBullet-to-RDT training contract.

RDT represents many embodiments in one sparse 128-D state/action vector.  For
this single-arm UR5e benchmark we deliberately advertise only six absolute arm
joint positions and one absolute gripper-open value.  The RGB order matches the
official RDT contract exactly::

    [front(t-1), right_wrist(t-1), unavailable_left_wrist(t-1),
     front(t),   right_wrist(t),   unavailable_left_wrist(t)]

The unavailable view is the SigLIP processor-mean image.  No object pose,
target coordinate, segmentation mask, reward, or success label is returned to
the learner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure, copy_requested_gripper_field
import torch
from PIL import Image
from torch.utils.data import Dataset


STATE_DIM = 128
ACTION_HORIZON = 64
IMAGE_HISTORY = 2
NUM_CAMERAS = 3
ARM_INDICES = tuple(range(6))
GRIPPER_OPEN_INDEX = 10
VALID_INDICES = ARM_INDICES + (GRIPPER_OPEN_INDEX,)
CONTROL_HZ = 10

REQUIRED_ARRAYS = {
    "front_rgb",
    "wrist_rgb",
    "joint_positions",
    "gripper_open",
    "next_joint_targets",
    "next_gripper_closure",
    "prompt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_joints(joints: np.ndarray) -> np.ndarray:
    """Use one continuous, documented joint convention for train and serve.

    PyBullet reports the fifth UR5e joint around ``3*pi/2`` while the same
    physical angle is conventionally represented as ``-pi/2``.  RDT consumes
    physical joint angles without normalization, so wrapping every revolute
    joint to ``[-pi, pi)`` avoids a gratuitous 2*pi distribution shift.
    """

    values = np.asarray(joints, dtype=np.float32)
    return ((values + np.float32(np.pi)) % np.float32(2.0 * np.pi) - np.float32(np.pi)).astype(
        np.float32, copy=False
    )


def validate_lineage(
    source: Path, audit_report: Path, action_contract_report: Path | None = None
) -> dict:
    """Fail closed unless ``source`` is the immutable, audited one-shot view."""

    source = source.resolve()
    audit_report = audit_report.resolve()
    index_path = source / "index.json"
    manifest_path = source / "dataset_manifest.json"
    for path in (index_path, manifest_path, audit_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    index_hash = sha256(index_path)
    manifest_hash = sha256(manifest_path)
    index = json.loads(index_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_report.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, source)
    if not isinstance(index, list) or not index:
        raise RuntimeError("RDT source index must be a non-empty list")
    if any(not item.get("success") for item in index):
        raise RuntimeError("RDT training source contains a failed demonstration")
    expected_frames = sum(int(item["steps"]) for item in index)
    expected_files = [str(item["file"]) for item in index]
    if len(expected_files) != len(set(expected_files)):
        raise RuntimeError("RDT training source contains duplicate episode paths")
    if (
        manifest.get("robot") != "ur5e"
        or manifest.get("gripper") != "Robotiq 2F-85"
        or int(manifest.get("control_hz", -1)) != CONTROL_HZ
        or int(manifest.get("image_size", -1)) != 384
        or manifest.get("training_view") != "vla_one_shot_success"
        or int(manifest.get("required_success_steps", -1)) != 1
        or manifest.get("privileged_pose_in_policy_input") is not False
    ):
        raise RuntimeError("RDT requires the locked UR5e/Robotiq 10 Hz one-shot VLA source")
    if (
        not audit.get("passed")
        or audit.get("failures")
        or Path(audit.get("dataset", "")).resolve() != source
        or audit.get("index_sha256") != index_hash
        or audit.get("manifest_sha256") != manifest_hash
        or int(audit.get("episodes", -1)) != len(index)
        or int(audit.get("frames", -1)) != expected_frames
        or int(audit.get("required_success_steps", -1)) != 1
        or audit.get("all_oracle_success") is not True
        or audit.get("temporal_alignment", {}).get(
            "next_joint_to_next_observation_max_abs_error"
        )
        != 0.0
        or audit.get("temporal_alignment", {}).get(
            "next_gripper_to_next_observation_max_abs_error"
        )
        != 0.0
    ):
        raise RuntimeError("RDT source camera/data audit is failed, incomplete, or stale")
    missing = [name for name in expected_files if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} RDT source episodes; first={missing[0]}")
    result = {
        "source": str(source),
        "audit_report": str(audit_report),
        "episodes": len(index),
        "frames": expected_frames,
        "source_index_sha256": index_hash,
        "source_manifest_sha256": manifest_hash,
        "audit_report_sha256": sha256(audit_report),
        "control_hz": CONTROL_HZ,
        "image_size": 384,
        "action_horizon": ACTION_HORIZON,
        "valid_state_action_indices": list(VALID_INDICES),
        "joint_convention": "wrap_to_minus_pi_pi",
    }
    if action_contract_report is not None:
        action_contract_report = action_contract_report.resolve()
        if not action_contract_report.is_file():
            raise FileNotFoundError(action_contract_report)
        action_audit = json.loads(action_contract_report.read_text())
        from VLA.common.runtime_provenance import verify_source_environment_runtime
        verify_source_environment_runtime(action_audit, source)
        contract_path = Path(__file__).resolve()
        environment_path = contract_path.parents[1] / "common" / "environment.py"
        if (
            not action_audit.get("passed")
            or action_audit.get("model") != "rdt"
            or Path(action_audit.get("source", "")).resolve() != source
            or action_audit.get("source_index_sha256") != index_hash
            or action_audit.get("source_manifest_sha256") != manifest_hash
            or int(action_audit.get("frames", -1)) != expected_frames
            or action_audit.get("joint_convention") != "wrap_to_minus_pi_pi"
            or action_audit.get("training_contract_sha256") != sha256(contract_path)
            or action_audit.get("environment_sha256") != sha256(environment_path)
            or float(action_audit.get("gripper_max_abs_error", float("inf"))) != 0.0
        ):
            raise RuntimeError("RDT native-action contract audit is failed or stale")
        result["action_contract_report"] = str(action_contract_report)
        result["action_contract_report_sha256"] = sha256(action_contract_report)
    return result


def validate_language_embeddings(
    embeddings_path: Path,
    manifest_path: Path,
    source_index_path: Path,
    prompts: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load T5 embeddings and bind them to the exact training index by hash."""

    for path in (embeddings_path, manifest_path, source_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(manifest_path.read_text())
    if (
        metadata.get("embeddings_sha256") != sha256(embeddings_path)
        or metadata.get("source_index_sha256") != sha256(source_index_path)
        or int(metadata.get("training_prompt_count", metadata.get("prompt_count", -1))) != len(set(prompts))
        or metadata.get("dtype") != "torch.bfloat16"
        or int(metadata.get("embedding_dim", -1)) != 4096
    ):
        raise RuntimeError("RDT language-embedding manifest is missing or stale")
    loaded = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("RDT language embeddings must be a prompt-to-tensor dictionary")
    missing = sorted(set(prompts) - set(loaded))
    extras = sorted(set(loaded) - set(prompts))
    declared_extras = metadata.get("additional_prompts_scope") == "declared_task_vocabulary_no_observations"
    if declared_extras:
        from VLA.rdt.prompt_vocabulary import declared_prompt_vocabulary, vocabulary_sha256
        if (set(loaded) != set(declared_prompt_vocabulary())
                or metadata.get("declared_prompt_vocabulary_sha256") != vocabulary_sha256()):
            raise RuntimeError("RDT extra prompts are not the locked public task grammar")
    if missing or (extras and not declared_extras) or metadata.get("prompt_count") != len(loaded):
        raise RuntimeError(
            f"RDT prompt embedding vocabulary mismatch: missing={len(missing)}, extras={len(extras)}"
        )
    normalized: dict[str, torch.Tensor] = {}
    for prompt, value in loaded.items():
        tensor = torch.as_tensor(value, dtype=torch.bfloat16, device="cpu")
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2 or tensor.shape[1] != 4096 or tensor.shape[0] < 1:
            raise ValueError(f"Invalid T5 embedding for {prompt!r}: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite T5 embedding for {prompt!r}")
        normalized[prompt] = tensor.contiguous()
    return normalized, metadata


@dataclass(frozen=True)
class EpisodeSamples:
    """Several independently sampled transitions decoded from one NPZ open."""

    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    state_mask: np.ndarray
    action_valid_lengths: np.ndarray
    prompts: tuple[str, ...]
    episode_file: str
    step_ids: np.ndarray


class PyBulletRDTDataset(Dataset):
    """Episode-balanced map dataset with deterministic per-epoch sampling."""

    def __init__(
        self,
        source: Path,
        *,
        samples_per_episode: int = 4,
        seed: int = 0,
        max_episodes: int | None = None,
        action_horizon: int = ACTION_HORIZON,
    ) -> None:
        if samples_per_episode < 1:
            raise ValueError("samples_per_episode must be positive")
        self.source = Path(source).resolve()
        self.index = json.loads((self.source / "index.json").read_text())
        if max_episodes is not None:
            if max_episodes < 1:
                raise ValueError("max_episodes must be positive")
            self.index = self.index[:max_episodes]
        self.samples_per_episode = int(samples_per_episode)
        self.seed = int(seed)
        self.epoch = 0
        if action_horizon < 1 or action_horizon > ACTION_HORIZON:
            raise ValueError("RDT action_horizon must lie in [1,64]")
        self.action_horizon = int(action_horizon)

    @property
    def prompts(self) -> list[str]:
        return [str(item["task"]["prompt"]) for item in self.index]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> EpisodeSamples:
        item = self.index[index]
        episode_path = self.source / item["file"]
        with np.load(episode_path, allow_pickle=False) as archive:
            episode = {key: archive[key] for key in REQUIRED_ARRAYS}
            copy_requested_gripper_field(archive, episode)
            missing = REQUIRED_ARRAYS - set(episode)
            if missing:
                raise KeyError(f"{episode_path} is missing {sorted(missing)}")
            length = int(item["steps"])
            prompt = str(np.asarray(episode["prompt"]).item())
            if prompt != str(item["task"]["prompt"]):
                raise RuntimeError(f"Prompt mismatch in {episode_path}")
            keys_with_time = (
                "front_rgb",
                "wrist_rgb",
                "joint_positions",
                "gripper_open",
                "next_joint_targets",
                "next_gripper_closure",
            )
            if any(len(episode[key]) != length for key in keys_with_time):
                raise RuntimeError(f"Temporal length mismatch in {episode_path}")
            front = np.asarray(episode["front_rgb"], dtype=np.uint8)
            wrist = np.asarray(episode["wrist_rgb"], dtype=np.uint8)
            joints = np.asarray(episode["joint_positions"], dtype=np.float32)
            gripper_open = np.asarray(episode["gripper_open"], dtype=np.float32)
            targets = np.asarray(episode["next_joint_targets"], dtype=np.float32)
            next_open = 1.0 - np.asarray(native_gripper_closure(episode), dtype=np.float32)

        if (
            front.ndim != 4
            or wrist.shape != front.shape
            or front.shape[1:] != (384, 384, 3)
            or joints.shape != (length, 6)
            or targets.shape != (length, 6)
            or gripper_open.shape != (length,)
            or next_open.shape != (length,)
        ):
            raise ValueError(f"Unexpected RDT source shapes in {episode_path}")
        numeric = (joints, targets, gripper_open, next_open)
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError(f"Non-finite RDT source values in {episode_path}")
        if np.any((gripper_open < 0.0) | (gripper_open > 1.0)) or np.any(
            (next_open < 0.0) | (next_open > 1.0)
        ):
            raise ValueError(f"Gripper-open value outside [0,1] in {episode_path}")

        # One deterministic generator per (epoch, episode) makes resumes and
        # multi-worker loading reproducible without sharing mutable RNG state.
        seed_sequence = np.random.SeedSequence((self.seed, self.epoch, int(index)))
        rng = np.random.default_rng(seed_sequence)
        replace = length < self.samples_per_episode
        step_ids = np.sort(
            rng.choice(length, size=self.samples_per_episode, replace=replace).astype(np.int64)
        )

        sample_count = len(step_ids)
        images = np.empty((sample_count, 4, 384, 384, 3), dtype=np.uint8)
        states = np.zeros((sample_count, 1, STATE_DIM), dtype=np.float32)
        actions = np.zeros((sample_count, self.action_horizon, STATE_DIM), dtype=np.float32)
        state_mask = np.zeros((sample_count, STATE_DIM), dtype=np.float32)
        state_mask[:, VALID_INDICES] = 1.0
        valid_lengths = np.empty(sample_count, dtype=np.int64)
        wrapped_joints = wrap_joints(joints)
        wrapped_targets = wrap_joints(targets)

        for sample_id, step_id in enumerate(step_ids):
            previous = max(int(step_id) - 1, 0)
            # Four real views; the collator inserts the two unavailable camera
            # slots using the exact processor-mean background.
            images[sample_id] = np.stack(
                (front[previous], wrist[previous], front[step_id], wrist[step_id]), axis=0
            )
            states[sample_id, 0, ARM_INDICES] = wrapped_joints[step_id]
            states[sample_id, 0, GRIPPER_OPEN_INDEX] = gripper_open[step_id]
            stop = min(int(step_id) + self.action_horizon, length)
            valid = stop - int(step_id)
            valid_lengths[sample_id] = valid
            # The six arm slots are contiguous.  A slice preserves the
            # (time, joint) axis order; NumPy advanced indexing with the tuple
            # ``ARM_INDICES`` would transpose this selection.
            actions[sample_id, :valid, :6] = wrapped_targets[step_id:stop]
            actions[sample_id, :valid, GRIPPER_OPEN_INDEX] = next_open[step_id:stop]
            if valid < self.action_horizon:
                actions[sample_id, valid:] = actions[sample_id, valid - 1]

        return EpisodeSamples(
            images=images,
            states=states,
            actions=actions,
            state_mask=state_mask,
            action_valid_lengths=valid_lengths,
            prompts=(prompt,) * sample_count,
            episode_file=str(item["file"]),
            step_ids=step_ids,
        )


class RDTBatchCollator:
    """Apply the official SigLIP preprocessor and pad variable T5 sequences."""

    def __init__(self, image_processor, language_embeddings: Mapping[str, torch.Tensor]) -> None:
        self.image_processor = image_processor
        self.language_embeddings = language_embeddings
        background_rgb = tuple(int(value * 255) for value in image_processor.image_mean)
        background = Image.new("RGB", (384, 384), background_rgb)
        self.background = image_processor.preprocess(
            background, return_tensors="pt"
        )["pixel_values"][0]

    def __call__(self, episodes: Sequence[EpisodeSamples]) -> dict[str, object]:
        if not episodes:
            raise ValueError("Cannot collate an empty RDT batch")
        states = torch.from_numpy(np.concatenate([item.states for item in episodes], axis=0))
        actions = torch.from_numpy(np.concatenate([item.actions for item in episodes], axis=0))
        state_mask = torch.from_numpy(
            np.concatenate([item.state_mask for item in episodes], axis=0)
        )
        valid_lengths = torch.from_numpy(
            np.concatenate([item.action_valid_lengths for item in episodes], axis=0)
        )
        prompts = [prompt for item in episodes for prompt in item.prompts]
        real_images = np.concatenate([item.images for item in episodes], axis=0)
        batch_size = real_images.shape[0]
        flattened = [
            Image.fromarray(real_images[sample, view], mode="RGB")
            for sample in range(batch_size)
            for view in range(4)
        ]
        processed = self.image_processor.preprocess(
            flattened, return_tensors="pt"
        )["pixel_values"].reshape(batch_size, 4, 3, 384, 384)
        background = self.background.view(1, 3, 384, 384).expand(batch_size, -1, -1, -1)
        # [front_prev, wrist_prev, bg_prev, front_now, wrist_now, bg_now]
        images = torch.stack(
            (
                processed[:, 0],
                processed[:, 1],
                background,
                processed[:, 2],
                processed[:, 3],
                background,
            ),
            dim=1,
        )

        language = [self.language_embeddings[prompt] for prompt in prompts]
        lengths = torch.tensor([tensor.shape[0] for tensor in language], dtype=torch.int64)
        lang_tokens = torch.nn.utils.rnn.pad_sequence(language, batch_first=True)
        positions = torch.arange(lang_tokens.shape[1]).unsqueeze(0)
        lang_attn_mask = positions < lengths.unsqueeze(1)
        return {
            "images": images,
            "states": states,
            "actions": actions,
            "state_elem_mask": state_mask,
            "ctrl_freqs": torch.full((batch_size,), CONTROL_HZ, dtype=torch.int64),
            "lang_tokens": lang_tokens,
            "lang_attn_mask": lang_attn_mask,
            "action_valid_lengths": valid_lengths,
            "prompts": prompts,
            "episode_files": [item.episode_file for item in episodes],
            "step_ids": torch.from_numpy(
                np.concatenate([item.step_ids for item in episodes], axis=0)
            ),
        }
