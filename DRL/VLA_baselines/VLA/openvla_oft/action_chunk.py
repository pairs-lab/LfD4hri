"""Dependency-free, ordered chunk-prefix selection; not a robot/model adapter.

Do not extend a bounded deque with a full chunk: deque(maxlen=1).extend(chunk)
keeps the last action. Receding-horizon control must execute the FIRST actions.
This utility validates numeric shape/finiteness only; it does not unnormalize,
clip, change coordinate frames, invert a LIBERO gripper, or guarantee safety.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Iterable


ACTION_DIM = 7


def consume_chunk_prefix(actions: Iterable[Iterable[Real]], replan_steps: int) -> list[list[float]]:
    """Copy the first ``replan_steps`` of a finite, horizon-by-7 action chunk.

The whole chunk, including unused future actions, must be valid. Returned
values keep their original order, sign and scale; semantics must be audited
by a future model-specific adapter before anything is sent to the controller.
"""
    if type(replan_steps) is not int or replan_steps < 1:
        raise ValueError("replan_steps must be an integer >= 1")
    try:
        rows = list(actions)
    except TypeError as exc:
        raise ValueError("Expected an iterable action chunk") from exc
    if not rows or replan_steps > len(rows):
        raise ValueError("replan_steps must be between 1 and the chunk horizon")
    validated = []
    for row_number, action in enumerate(rows):
        try:
            values = list(action)
        except TypeError as exc:
            raise ValueError(f"Action {row_number} must be a 7D numeric vector") from exc
        if len(values) != ACTION_DIM:
            raise ValueError(f"Action {row_number} has {len(values)} channels; expected 7")
        converted = []
        for column, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Action {row_number}, channel {column} must be a real number")
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"Action {row_number}, channel {column} cannot be represented as a finite float") from exc
            if not math.isfinite(value):
                raise ValueError(f"Action {row_number}, channel {column} is non-finite")
            converted.append(value)
        validated.append(converted)
    return validated[:replan_steps]
