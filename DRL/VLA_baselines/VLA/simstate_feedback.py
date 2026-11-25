"""Explicit SIMULATION-PRIVILEGED VLA input ablation, not an action controller.

All values are measured BEFORE the policy action. No future action, teacher
phase, reward or success flag is included. The policy remains the sole source
of actions; the existing independent physics safety guard remains unchanged.
"""
import numpy as np

SCHEMA = "simstate18_causal_target_goal_feedback_v1"
FEATURE_NAMES = ["q0", "q1", "q2", "q3", "q4", "q5", "encoder_closure",
    "pad_center_x", "pad_center_y", "pad_center_z", "requested_close_intent",
    "requested_object_center_minus_pad_x", "requested_object_center_minus_pad_y",
    "requested_object_center_minus_pad_z", "goal_minus_object_base_x",
    "goal_minus_object_base_y", "goal_minus_object_base_z", "verified_correct_body_grasp"]


def feedback_state(env, observation):
    state = env.debug_state()
    pad = np.asarray(state["grasp_center"], dtype=np.float32)
    target = np.asarray(state["pick_center"], dtype=np.float32)
    base = np.asarray(state["pick_position"], dtype=np.float32)
    goal = np.asarray(state["goal_position"], dtype=np.float32)
    held = env.gripper.grasped_object == env.pick_id and bool(env.gripper._bilateral(env.pick_id))
    values = np.r_[observation.joint_positions.astype(np.float32),
        np.float32(env.normalized_gripper_closure()), pad,
        np.float32(env.commanded_gripper_closure()), target - pad, goal - base, np.float32(held)].astype(np.float32)
    if values.shape != (18,) or not np.isfinite(values).all():
        raise ValueError("Invalid causal privileged state")
    return values


def state_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 18 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Complete finite TRAIN state18 is required")
    return {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "min": values.min(0).tolist(), "max": values.max(0).tolist(), "count": [len(values)]}
