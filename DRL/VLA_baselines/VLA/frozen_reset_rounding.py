"""Pixel/joint bit parity, permitting only machine-roundoff in FK aperture.

Normalized FK closure is float64 geometry computed from link transforms. Eight
float64 epsilon at normalized unit scale is NOT a physical sensor tolerance;
it is about 1.5e-16 metres at the 85mm gripper span. No image/joint tolerance.
"""
import numpy as np

FK_CLOSURE_ROUNDOFF = 8 * np.finfo(np.float64).eps


def compare_frozen_reset(new, original):
    exact = {key: bool(np.array_equal(new[key][0], original[key][0]))
             for key in ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure")}
    closure = float(new["gripper_closure"][0])
    previous = float(original["gripper_closure"][0])
    difference = abs(closure - previous)
    if (not all(exact[key] for key in ("front_rgb", "wrist_rgb", "joint_positions"))
            or not np.isfinite([closure, previous]).all() or difference > FK_CLOSURE_ROUNDOFF):
        raise ValueError("Frozen pixels/joints differ or FK closure exceeds machine-roundoff")
    return {"passed": True, "all_fields_bitwise_exact": all(exact.values()), "bitwise_exact": exact,
        "closure_abs_error": difference, "closure_machine_roundoff_limit": FK_CLOSURE_ROUNDOFF,
        "pixels_and_joints_tolerance": 0, "protocol": "exact_pixels_joints_float64_unit_scale_FK_roundoff_only"}
