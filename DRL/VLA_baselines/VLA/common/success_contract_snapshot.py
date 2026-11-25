"""Offline source-label contracts; never change learned rollout thresholds."""
import math


def recorded_distance_thresholds(item):
    contract = item.get("success_contract", {})
    values = []
    for key in ("touch_distance_threshold_m", "object_goal_threshold_m"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Missing/invalid recorded success threshold: {key}")
        values.append(float(value))
    return tuple(values)


def resolve_audit_thresholds(item, required_steps, live_thresholds, allow_stricter_frozen=False):
    """Permit an explicit historical snapshot only if it is no looser than live.

    Recomputing labels with a changed live reward file is not a temporal data
    audit. Default callers nevertheless reject any mismatch, preserving the
    existing gate. Historical labels and arrays are never modified here.
    """
    recorded = recorded_distance_thresholds(item)
    if item["success_contract"].get("required_consecutive_steps") != required_steps:
        raise ValueError("Recorded consecutive-step contract differs from manifest")
    live = tuple(float(value) for value in live_thresholds)
    if recorded == live:
        return live
    if not allow_stricter_frozen:
        raise ValueError("Success contract does not match live reward.xml")
    if any(old > current for old, current in zip(recorded, live, strict=True)):
        raise ValueError("Historical success contract is looser than live reward.xml; explicit requalification required")
    return recorded
