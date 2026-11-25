"""Audit-only causal decision frame; teacher references NEVER policy inputs."""
import numpy as np

SENSOR_FIELDS = ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure",
                 "gripper_open", "ee_position", "ee_rotation_6d")


def paired_decision_frame(first, second, horizon=5):
    """First text-dependent action from identical physical history.

    Do not choose a later, easier XY frame after differing yaw/gripper controls:
    that would no longer be a text-only intervention at identical sensors.
    Selection reads teacher labels, never model outputs or success/loss.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Positive integer diagnostic horizon required")
    a, b = (np.asarray(row["canonical_actions"]) for row in (first, second))
    if any(v.ndim != 2 or v.shape[1] != 7 or not np.isfinite(v).all() for v in (a, b)):
        raise ValueError("Finite canonical 7D actions required")
    length = min(len(a), len(b))
    differences = np.flatnonzero(np.any(a[:length] != b[:length], axis=1))
    if not len(differences):
        raise ValueError("No text-dependent action before either trajectory ends")
    frame = int(differences[0])
    for key in SENSOR_FIELDS:
        x, y = np.asarray(first[key]), np.asarray(second[key])
        if len(x) <= frame or len(y) <= frame or x.shape[1:] != y.shape[1:]:
            raise ValueError("Sensor timeline/shape differs: "+key)
        if not np.array_equal(x[:frame+1], y[:frame+1]):
            raise ValueError("Text-only intervention changed common-prefix sensor "+key)
    end = min(length, frame+horizon)
    separation = float(np.sqrt(np.mean(np.sum((a[frame:end, :2]-b[frame:end, :2])**2, axis=1))))
    return {"frame": frame, "identical_sensor_prefix_frames": frame+1,
        "identical_physical_control_prefix_steps": frame, "diagnostic_horizon": end-frame,
        "teacher_xy_separation_rms_m": separation,
        "informative_teacher_xy_separation": separation >= .002,
        "first_difference_dimensions": np.flatnonzero(a[frame] != b[frame]).tolist(),
        "selection_uses_neural_outcomes": False}
