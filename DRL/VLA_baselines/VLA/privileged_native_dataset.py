"""Lossless source NPZ view with a qualified causal feedback prompt.

No PNG re-encoding, action relabelling, resampling of failed scenes or future
feedback. Native action conversions remain separate from conditioning.
"""
from bisect import bisect_right
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from VLA.common.native_action_fields import native_gripper_closure, copy_requested_gripper_field
from VLA.privileged_prompt import augment_prompt
from VLA.smolvla.simstate_dataset import load_projection


class PrivilegedNativeDataset:
    def __init__(self, projection, action_horizon=10):
        self.states, self.audit, self.stats = load_projection(projection)
        self.source = Path(self.audit["source"])
        self.rows = json.loads((self.source / "index.json").read_text())
        self.ends = np.cumsum([int(r["steps"]) for r in self.rows]).tolist()
        self.horizon = int(action_horizon)
        if self.horizon < 1 or self.ends[-1] != len(self.states):
            raise ValueError("Projection/source frame count or horizon differs")

    def __len__(self):
        return len(self.states)

    @lru_cache(maxsize=1)
    def episode(self, episode_index):
        with np.load(self.source / self.rows[episode_index]["file"], allow_pickle=False) as archive:
            episode = {key: archive[key] for key in (
                "front_rgb", "wrist_rgb", "joint_positions", "gripper_closure",
                "next_joint_targets", "next_gripper_closure", "canonical_actions", "prompt")}
            copy_requested_gripper_field(archive, episode)
        return episode

    def address(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.ends, index)
        offset = 0 if episode_index == 0 else self.ends[episode_index - 1]
        return episode_index, index - offset

    def __getitem__(self, index):
        ei, fi = self.address(index)
        episode = self.episode(ei)
        state = self.states[int(index)]
        ordinary = np.r_[episode["joint_positions"][fi], episode["gripper_closure"][fi]].astype(np.float32)
        if not np.allclose(ordinary, state[:7], atol=1e-6, rtol=0):
            raise ValueError("Feedback did not originate from this pre-action source frame")
        ids = np.minimum(fi + np.arange(self.horizon), len(episode["canonical_actions"]) - 1)
        native = np.c_[episode["next_joint_targets"][ids], native_gripper_closure(episode)[ids]].astype(np.float32)
        native[:, :6] = (native[:, :6] + np.pi) % (2 * np.pi) - np.pi
        canonical = episode["canonical_actions"][ids].astype(np.float32, copy=True)
        language_action = canonical.copy()
        language_action[:, 6] = (1 - canonical[:, 6]) / 2
        return {"index": int(index), "episode_index": ei, "frame_index": fi,
            "image": episode["front_rgb"][fi].copy(), "wrist_image": episode["wrist_rgb"][fi].copy(),
            "joints": ((ordinary[:6] + np.pi) % (2 * np.pi) - np.pi).astype(np.float32),
            "gripper": ordinary[6:].copy(), "actions": native,
            "language_actions": language_action, "canonical_actions": canonical,
            "original_prompt": str(episode["prompt"].item()),
            "prompt": augment_prompt(str(episode["prompt"].item()), state),
            "pre_action_state18": state.copy()}

    def __getitems__(self, indices):
        # No inherited fast path is allowed to discard the feedback overlay.
        return [self[int(index)] for index in indices]
