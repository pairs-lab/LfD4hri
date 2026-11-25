"""Shared causal scene feedback for EVERY VLA family, not a controller.

All64 measured fields remain available. Numeric encoders use67 values,
replacing three Euler angles by sine/cosine pairs to avoid artificial jumps
at +/-pi. No desired pose/action, teacher phase or future value is computed.
"""
import re
import numpy as np
from VLA.scene64_schema import FEATURE_NAMES, SCHEMA

NUMERIC_SCHEMA = "simscene67_causal_angles_sincos_v1"
PROMPT_SCHEMA = "simscene64_native_language_complete_v1"
MARKER = "simulation observation before action:"
NUMERIC_NAMES = FEATURE_NAMES[:35] + [f"target_{axis}_{kind}" for axis in ("roll", "pitch", "yaw")
    for kind in ("sin", "cos")] + FEATURE_NAMES[38:]
SLICES = (("q_rad", 0, 6), ("encoder_closed", 6, 7), ("pad_world_xyz_m", 7, 10),
    ("previous_close", 10, 11), ("target_center_relative_to_pad_m", 11, 14),
    ("goal_relative_to_object_base_m", 14, 17), ("correct_bilateral_hold", 17, 18),
    ("tool_rotation_first_two_columns", 18, 24), ("target_size_xyz_m", 24, 27),
    ("jaw_gap_m", 27, 28), ("right_left_target_contact", 28, 30),
    ("wrong_attached_any_attached", 30, 32), ("target_velocity_xyz_m_per_s", 32, 35),
    ("target_roll_pitch_yaw_rad", 35, 38), ("pad_height_fraction_of_target", 38, 39),
    ("target_pad_force_N", 39, 40)) + tuple(
    (f"neighbor{n}_relative_xyz_and_xy_radius_m", 40+4*n, 44+4*n) for n in range(6))


def validate_scene(values):
    v = np.asarray(values, np.float32)
    if v.shape != (64,) or not np.isfinite(v).all(): raise ValueError("All64 finite pre-action fields required")
    if not 0 <= v[6] <= 1 or not 0 <= v[10] <= 1: raise ValueError("Invalid measured/requested closure")
    if any(v[i] not in (0, 1) for i in (17, 28, 29, 30, 31)): raise ValueError("Invalid contact/identity feedback")
    if v[30] > v[31] or v[17] != float(bool(v[31] and not v[30] and v[28] and v[29])):
        raise ValueError("Attachment identity and bilateral feedback disagree")
    if np.any(v[24:27] <= 0) or v[27] < 0 or v[39] < 0: raise ValueError("Invalid physical extent/aperture/force")
    columns = v[18:24].reshape(2, 3)
    if np.max(np.abs(columns @ columns.T - np.eye(2))) > 1e-4:
        raise ValueError("Actual tool6D rotation convention/orthonormality differs")
    neighbors = v[40:].reshape(6, 4); padding = neighbors[:, 3] == 0
    if np.any(neighbors[:, 3] < 0) or np.any(neighbors[padding] != 0): raise ValueError("Invalid neighbor padding")
    if np.any(np.diff(padding.astype(int)) < 0): raise ValueError("Neighbors must be packed in stable name order")
    return v


def encode_numeric(values):
    v = validate_scene(values)
    angle = np.stack((np.sin(v[35:38]), np.cos(v[35:38])), axis=-1).reshape(-1)
    result = np.r_[v[:35], angle, v[38:]].astype(np.float32)
    if result.shape != (67,): raise ValueError("Native67 dimension error")
    return result


def normalization(values):
    encoded = np.asarray([encode_numeric(v) for v in values], np.float64)
    if not len(encoded): raise ValueError("TRAIN before-action data required")
    empirical = {"mean": encoded.mean(0), "std": encoded.std(0), "min": encoded.min(0),
        "max": encoded.max(0), "count": np.array([len(encoded)], np.int64)}
    # Keep the inherited18 normalization EXACT. New near-constant numerical
    # orientation/velocity noise must not be amplified by division by1e-8.
    stats = {k: v.copy() for k, v in empirical.items()}
    stats["std"][18:] = np.maximum(stats["std"][18:], .001)
    for index in (28, 29, 30, 31): stats["mean"][index], stats["std"][index] = 0., 1.
    return ({k: v.tolist() for k, v in stats.items()}, {"scope": "regularized_TRAIN_conditioning_NOT_empirical_statistics",
        "empirical_stats": {k: v.tolist() for k, v in empirical.items()}, "first18_empirical_preserved": True,
        "new_continuous_std_floor": .001, "new_binary_identity_indices": [28, 29, 30, 31]})


def augment_prompt(instruction, values):
    instruction = str(instruction).strip()
    if not instruction or MARKER in instruction.lower(): raise ValueError("Original instruction required")
    v = validate_scene(values)
    fields = [name + "=" + ",".join(f"{0. if abs(float(x)) < .00005 else float(x):.4f}" for x in v[a:b])
        for name, a, b in SLICES]
    return instruction + "; " + MARKER + " " + "; ".join(fields)


def parse_prompt(prompt):
    if str(prompt).lower().count(MARKER) != 1: raise ValueError("Missing/duplicated scene feedback")
    parts = str(prompt).split(MARKER, 1)[1].strip().split("; "); result = []
    if len(parts) != len(SLICES): raise ValueError("Not all scene fields survived formatting")
    for text, (name, a, b) in zip(parts, SLICES, strict=True):
        if not text.startswith(name + "="): raise ValueError("Scene field name/order changed")
        words = text[len(name)+1:].split(",")
        if len(words) != b-a or any(not re.fullmatch(r"-?\d+\.\d{4}", word) for word in words):
            raise ValueError("Malformed scene numbers")
        result.extend(map(float, words))
    # Decimal transport intentionally rounds rotations, so validate original
    # orthonormality BEFORE transport; do not reject its <=5e-5 rounding here.
    v = np.asarray(result, np.float32)
    if v.shape != (64,) or not np.isfinite(v).all(): raise ValueError("Incomplete scene transport")
    return v


def model_input(env, observation):
    from VLA.simstate_scene64 import scene_state
    raw = scene_state(env, observation)
    return {"scene64": raw, "numeric67": encode_numeric(raw),
        "prompt": augment_prompt(observation.prompt, raw), "raw_schema": SCHEMA,
        "numeric_schema": NUMERIC_SCHEMA, "prompt_schema": PROMPT_SCHEMA}
