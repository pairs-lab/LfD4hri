"""Dependency-light mirror of the frozen physical extractor's field schema.

The loader checks this list against its actual replay receipt. Native model
servers need no PyBullet import just to format already measured observations.
"""
from VLA.simstate_feedback import FEATURE_NAMES as OLD_NAMES
SCHEMA = "simstate64_causal_scene_contact_feedback_v1"
FEATURE_NAMES = OLD_NAMES + [*[f"ee_rotation_6d_{i}" for i in range(6)],
    "target_aabb_extent_x", "target_aabb_extent_y", "target_aabb_extent_z", "jaw_gap_m",
    "target_right_pad_contact", "target_left_pad_contact", "attached_wrong_object", "any_object_attached",
    "target_velocity_x", "target_velocity_y", "target_velocity_z",
    "target_euler_roll", "target_euler_pitch", "target_euler_yaw",
    "pad_height_fraction_of_target_aabb", "target_pad_normal_force_n"] + [
    f"neighbor{n}_{field}" for n in range(6) for field in ("relative_x", "relative_y", "relative_z", "xy_bound_radius")]
