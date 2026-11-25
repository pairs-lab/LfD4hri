"""More complete CAUSAL privileged VLA observations, never an action planner.

Preserves state18 exactly. Adds orientation, measured target geometry/motion,
per-pad force/contact, attachment identity and every distractor in this
benchmark (maximum six). No physics step, desired action, teacher phase,
future value, reward or success flag is read into conditioning.
"""
import numpy as np
import pybullet as p

from VLA.simstate_feedback import FEATURE_NAMES as OLD_NAMES, feedback_state

SCHEMA = "simstate64_causal_scene_contact_feedback_v1"
MAX_NEIGHBORS = 6
FEATURE_NAMES = OLD_NAMES + [*[f"ee_rotation_6d_{i}" for i in range(6)],
    "target_aabb_extent_x", "target_aabb_extent_y", "target_aabb_extent_z", "jaw_gap_m",
    "target_right_pad_contact", "target_left_pad_contact", "attached_wrong_object", "any_object_attached",
    "target_velocity_x", "target_velocity_y", "target_velocity_z",
    "target_euler_roll", "target_euler_pitch", "target_euler_yaw",
    "pad_height_fraction_of_target_aabb", "target_pad_normal_force_n"] + [
    f"neighbor{n}_{field}" for n in range(MAX_NEIGHBORS) for field in ("relative_x", "relative_y", "relative_z", "xy_bound_radius")]


def scene_state(env, observation):
    first = feedback_state(env, observation)
    low, high = map(lambda x: np.asarray(x, np.float64), p.getAABB(env.pick_id))
    extent = high - low
    if np.any(extent <= 0): raise ValueError("Measured target bounds must be nondegenerate")
    pad = first[7:10].astype(np.float64)
    rotation = np.asarray(observation.ee_rotation_6d, np.float32)
    if rotation.shape != (6,): raise ValueError("Actual tool orientation required")
    points = p.getContactPoints(bodyA=env.gripper.body, bodyB=env.pick_id)
    forces = {link: sum(max(0., float(c[9])) for c in points if c[3] == link) for link in env.gripper.PAD_LINKS}
    # Match the EXISTING bilateral detector's per-contact force criterion,
    # rather than silently replacing it by a different sum-force threshold.
    contacts = [float(any(c[3] == link and float(c[9]) > .1 for c in points)) for link in env.gripper.PAD_LINKS]
    attached = env.gripper.grasped_object
    _, quaternion = p.getBasePositionAndOrientation(env.pick_id)
    velocity, _ = p.getBaseVelocity(env.pick_id)
    neighbors = sorted((name, body) for name, body in env.object_ids.items() if body != env.pick_id)
    if len(neighbors) > MAX_NEIGHBORS: raise ValueError("Do not silently omit surrounding objects")
    geometry = np.zeros((MAX_NEIGHBORS, 4), np.float32)
    for index, (_, body) in enumerate(neighbors):
        nlow, nhigh = map(lambda x: np.asarray(x, np.float64), p.getAABB(body))
        geometry[index, :3] = (nlow + nhigh) / 2 - pad
        geometry[index, 3] = .5 * np.linalg.norm((nhigh - nlow)[:2])
        if geometry[index, 3] <= 0: raise ValueError("Zero radius is reserved for missing-neighbor padding")
    values = np.r_[first, rotation, extent, env.gripper.width(), contacts,
        float(attached is not None and attached != env.pick_id), float(attached is not None),
        velocity, p.getEulerFromQuaternion(quaternion), (pad[2] - low[2]) / extent[2],
        sum(forces.values()), geometry.reshape(-1)].astype(np.float32)
    if values.shape != (64,) or not np.isfinite(values).all(): raise ValueError("Exactly64 finite causal features required")
    if not np.array_equal(values[:18], first): raise ValueError("Existing state18 contract changed")
    return values


def state_stats(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] != 64 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Complete finite before-action state64 required")
    return {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "min": values.min(0).tolist(), "max": values.max(0).tolist(), "count": [len(values)]}
