"""Policy protocol and model registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .observation import VLAObservation
from .tasks import TaskSpec

if TYPE_CHECKING:
    from .environment import ManipulationEnv


class Policy(ABC):
    name = "policy"
    comparable = True
    input_modalities: tuple[str, ...] = ()

    def reset(self, task: TaskSpec) -> None:
        del task

    @abstractmethod
    def predict(self, observation: VLAObservation) -> np.ndarray:
        """Return one canonical action or an ``(H, 7)`` action chunk."""

    def close(self) -> None:
        pass


def load_policy(name: str, env: "ManipulationEnv", **kwargs) -> Policy:
    if name == "oracle":
        from .scripted_policy import ScriptedPolicy

        return ScriptedPolicy(env)
    if name == "openvla":
        from VLA.openvla.adapter import OpenVLAPolicy

        return OpenVLAPolicy(**kwargs)
    if name in {"pi0", "pi0_fast"}:
        from VLA.pi0.adapter import OpenPiClientPolicy

        return OpenPiClientPolicy(env=env, model_name=name, **kwargs)
    if name == "cogact":
        from VLA.cogact.adapter import CogACTPolicy

        return CogACTPolicy(**kwargs)
    if name == "rdt":
        from VLA.rdt.adapter import RDTPolicy

        return RDTPolicy(env=env, **kwargs)
    if name == "smolvla":
        from VLA.smolvla.adapter import SmolVLAPolicyAdapter

        return SmolVLAPolicyAdapter(env=env, **kwargs)
    raise ValueError(f"Unknown model {name!r}")
