"""Canonical action contract shared by every policy adapter.

The benchmark deliberately separates a model's native action representation from
the simulator.  Every adapter must emit a delta end-effector pose plus one
Robotiq command in the following order:

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

Translations are metres in the PyBullet world frame, rotations are radians in
the end-effector local frame, and ``gripper`` is in [-1, 1] (negative=open,
positive=close).  This keeps embodiment conversion identical for all models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
ACTION_DIM = len(ACTION_LABELS)


@dataclass(frozen=True)
class CanonicalAction:
    values: np.ndarray

    @classmethod
    def from_array(
        cls,
        values: Iterable[float],
        translation_limit: float = 0.05,
        rotation_limit: float = 0.25,
    ) -> "CanonicalAction":
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape != (ACTION_DIM,):
            raise ValueError(f"Expected a {ACTION_DIM}D action, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Action contains NaN/Inf: {array}")
        array = array.copy()
        array[:3] = np.clip(array[:3], -translation_limit, translation_limit)
        array[3:6] = np.clip(array[3:6], -rotation_limit, rotation_limit)
        array[6] = np.clip(array[6], -1.0, 1.0)
        return cls(array)

    @property
    def translation(self) -> np.ndarray:
        return self.values[:3]

    @property
    def rotation(self) -> np.ndarray:
        return self.values[3:6]

    @property
    def gripper(self) -> float:
        return float(self.values[6])

