"""Observation type exposed to VLA adapters.

No object pose, target coordinate, or reward-internal state is exposed.  The
only task-level input is natural language.  Images and robot proprioception are
standard VLA sensor inputs, not privileged task annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VLAObservation:
    front_rgb: np.ndarray
    wrist_rgb: np.ndarray
    joint_positions: np.ndarray
    gripper_width: float
    gripper_open: float
    ee_position: np.ndarray
    ee_rotation_6d: np.ndarray
    prompt: str
    step: int

    @property
    def proprio(self) -> np.ndarray:
        return np.concatenate(
            (
                self.joint_positions.astype(np.float32),
                np.array([self.gripper_open], dtype=np.float32),
                self.ee_position.astype(np.float32),
                self.ee_rotation_6d.astype(np.float32),
            )
        )
