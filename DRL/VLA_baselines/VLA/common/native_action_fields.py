"""Keep measured gripper STATE distinct from requested native ACTION labels."""
import numpy as np


def native_gripper_closure(episode):
    if "next_requested_gripper_closure" not in episode and any(
            key in episode for key in ("next_verified_grasp", "next_bilateral_pad_contact")):
        raise ValueError("V9 feedback cannot substitute for missing requested gripper action labels")
    key = "next_requested_gripper_closure" if "next_requested_gripper_closure" in episode else "next_gripper_closure"
    values = np.asarray(episode[key])
    if not np.isfinite(values).all() or np.any(values < -1e-6) or np.any(values > 1+1e-6):
        raise ValueError("Native requested gripper closure must be finite in [0,1]")
    return values


def copy_requested_gripper_field(archive, materialized):
    # Converters often load only a subset of archive members.  Validate on
    # the original archive before its V9 feedback sentinels are discarded;
    # otherwise a malformed V9 episode would silently fall back to measured
    # closure, although its requested CLOSE command may be very different.
    if any(key in archive for key in ("next_verified_grasp", "next_bilateral_pad_contact")):
        native_gripper_closure(archive)
    if "next_requested_gripper_closure" in archive:
        materialized["next_requested_gripper_closure"] = archive["next_requested_gripper_closure"]
