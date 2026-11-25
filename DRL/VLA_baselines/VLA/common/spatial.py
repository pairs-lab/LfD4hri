"""Language-grounded workspace regions used only by the VLA benchmark.

The legacy RL configuration contains a typo for ``right side`` in
``FIXED_DESTINATION``: it points to the bottom-right corner even though
``PLACE_TARGETS`` defines the linguistic region at the middle-right edge.
RL has already been trained and is left untouched.  VLA data collection and
evaluation use this validated mapping so text and geometry cannot disagree.
"""

from __future__ import annotations

from config import FIXED_DESTINATION as LEGACY_FIXED_DESTINATION
from config import PLACE_TARGETS


FIXED_DESTINATION = {
    name: tuple(float(value) for value in PLACE_TARGETS[name])
    for name in LEGACY_FIXED_DESTINATION
}


def validate_spatial_language_mapping() -> None:
    if set(FIXED_DESTINATION) != set(LEGACY_FIXED_DESTINATION):
        raise RuntimeError("VLA fixed-destination vocabulary diverges from the simulator")
    duplicates: dict[tuple[float, float, float], list[str]] = {}
    for name, position in FIXED_DESTINATION.items():
        duplicates.setdefault(position, []).append(name)
    collisions = {position: names for position, names in duplicates.items() if len(names) > 1}
    if collisions:
        raise RuntimeError(f"Language regions share coordinates: {collisions}")
    if FIXED_DESTINATION["right side"] != (0.25, -0.5, 0.0):
        raise RuntimeError("right side must be the middle-right workspace region")


validate_spatial_language_mapping()
