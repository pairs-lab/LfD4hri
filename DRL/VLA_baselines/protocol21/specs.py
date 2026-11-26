"""Explicit model-specific interfaces; shared robot observations never imply equal tensors."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    horizon: int
    execute: int
    state_width: int
    history: int = 1
    image_size: int = 224
    depth: bool = False


SPECS = {
    'flower': Spec(10, 10, 21),
    'xvla': Spec(30, 30, 41),
    'pi0': Spec(5, 4, 32),
    'vla_adapter': Spec(8, 8, 29),
    'smolvla': Spec(5, 5, 21, image_size=512),
    'vla_jepa': Spec(7, 7, 21),
    'dita': Spec(10, 1, 28, history=2),
    'vlanext_s': Spec(8, 1, 28, history=8, image_size=256),
    'falcon_esm': Spec(10, 1, 21, depth=True),
    'seer': Spec(3, 1, 21, history=7),
}
