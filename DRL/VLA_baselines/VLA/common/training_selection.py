"""Deterministic train-only camera/data probes with destination coverage."""
from .spatial import FIXED_DESTINATION


def select_probe_rows(candidates, count, include_fixed_destinations=False, destination_vocabulary=None):
    if count < 1 or len(candidates) < count:
        raise ValueError("Not enough training layouts for requested probe count")
    chosen = list(candidates[:count])
    if (include_fixed_destinations or destination_vocabulary is not None) and chosen[0]["task"]["name"] in ("move", "pick_place"):
        existing = {row["task"].get("place_name") for row in chosen}
        for destination in (destination_vocabulary if destination_vocabulary is not None else FIXED_DESTINATION):
            if destination in existing:
                continue
            match = next((row for row in candidates if row["task"].get("place_name") == destination), None)
            if match is None:
                raise ValueError(f"Source has no probe for {destination}")
            chosen.append(match)
            existing.add(destination)
    return chosen
