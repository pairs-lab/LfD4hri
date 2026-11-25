"""Common environment, action contract, policies, and recording utilities."""

from .action import ACTION_DIM, ACTION_LABELS, CanonicalAction
from .tasks import TASKS, TaskSpec


def __getattr__(name: str):
    # Model training/conversion environments intentionally do not install
    # PyBullet.  Numeric provenance helpers and task definitions must remain
    # importable there; load the simulator only when explicitly requested.
    if name == "ManipulationEnv":
        from .environment import ManipulationEnv

        return ManipulationEnv
    raise AttributeError(name)

__all__ = [
    "ACTION_DIM",
    "ACTION_LABELS",
    "CanonicalAction",
    "ManipulationEnv",
    "TASKS",
    "TaskSpec",
]
