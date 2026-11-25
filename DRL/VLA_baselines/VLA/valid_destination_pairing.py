"""Geometry-only destination pairs; never change physical bodies or poses.

An uncommanded virtual destination in a sampled config need not be clear.
Retargeting must validate it BEFORE expert execution. If no configured
alternative is valid, replace an uncommanded virtual region with another
existing named region. This changes instruction/config metadata, not physics.
No teacher/neural outcomes choose the alternative, and no layouts are skipped.
"""
from dataclasses import replace

from VLA.audit_counterfactual_collection import alternate_destination, physical_scene_spec
from VLA.audit_finetune_data import _audit_rl_scene
from VLA.common.spatial import FIXED_DESTINATION


def scene_errors(task):
    failures = []
    _audit_rl_scene(task.to_dict(), "destination_pair", failures)
    return failures


def valid_destination_pair(task):
    if task.name not in ("move", "pick_place") or scene_errors(task):
        raise ValueError("Destination pair requires a valid original carry task")
    original_places = tuple(task.config_place_items)
    # Try each configured alternative separately: the legacy helper chooses
    # the most distant goal, not necessarily a clear one.
    for name in original_places:
        if name == task.place_name:
            continue
        restricted = replace(task, config_place_items=(task.place_name, name))
        candidate = replace(alternate_destination(restricted), config_place_items=original_places)
        if not scene_errors(candidate):
            return candidate, {"rule": "valid_existing_configured_destination", "replacement": None}
    # Replace only a non-selected virtual entry. Never add/remove/reclassify
    # a physical body, change cardinalities, move an obstacle, or invent XY.
    for old in original_places:
        if old == task.place_name or old not in FIXED_DESTINATION:
            continue
        for name in sorted(FIXED_DESTINATION):
            if name in original_places:
                continue
            places = tuple(name if item == old else item for item in original_places)
            temporary = replace(task, config_place_items=(task.place_name, name))
            candidate = replace(alternate_destination(temporary), config_place_items=places)
            if (not scene_errors(candidate)
                    and physical_scene_spec(candidate) == physical_scene_spec(task)):
                return candidate, {"rule": "clear_virtual_region_replacement",
                    "replacement": {"old": old, "new": name},
                    "original_config_place_items": list(original_places),
                    "new_config_place_items": list(places),
                    "physical_scene_changed": False, "selection_uses_outcomes": False}
    raise ValueError("No valid destination pair without changing the physical scene")
