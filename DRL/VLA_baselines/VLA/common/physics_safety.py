"""Task-independent geometry/linkage safety for the versioned VLA gripper."""
import numpy as np
import pybullet as p


class PhysicsSafetyViolation(RuntimeError):
    pass


def physics_diagnostics(env):
    """Audit-only pre-rollback evidence; NEVER part of a policy observation."""
    states = p.getJointStates(env.gripper.body, env.gripper.JOINTS)
    q = np.asarray([state[0] for state in states])
    contacts = p.getContactPoints(bodyA=env.gripper.body)
    return {
        "joint_ids": list(env.gripper.JOINTS),
        "joint_positions_rad": q.tolist(),
        "joint_velocities_rad_s": [float(state[1]) for state in states],
        "follower_errors_rad": (q[1:] - env.gripper.SIGNS[1:] * q[0]).tolist(),
        "requested_gripper_command": float(env.gripper.command),
        "surface_clearance_m": float(surface_clearance(env)),
        "contacts": [{"other_body": int(c[2]), "gripper_link": int(c[3]),
                      "other_link": int(c[4]), "distance_m": float(c[8]),
                      "normal_force_n": float(c[9]),
                      "position_on_gripper": list(c[5]), "normal_on_other": list(c[7])}
                     for c in contacts],
    }


def surface_clearance(env, distance=.25):
    points = [c for body in env.ground_bodies for c in
              p.getClosestPoints(env.gripper.body, body, distance=distance)]
    return min((c[8] for c in points), default=distance)


def enforce_physics_safety(env):
    clearance = surface_clearance(env, .005)
    if clearance < -.0008:
        raise PhysicsSafetyViolation(f"Gripper penetrated a ground/table surface by {-clearance:.6f}m")
    states = p.getJointStates(env.gripper.body, env.gripper.JOINTS)
    q = np.array([s[0] for s in states])
    velocity = np.array([s[1] for s in states])
    error = float(np.max(np.abs(q[1:] - env.gripper.SIGNS[1:] * q[0])))
    if error > .15 or np.abs(velocity).max() > 15 or not -.015 <= q[0] <= .849:
        raise PhysicsSafetyViolation(f"Gripper linkage unsafe: mimic_error={error:.4f}rad, velocity={abs(velocity).max():.2f}rad/s")
    state = p.getLinkState(env.robot_id, env.robot_spec.mount_link, computeForwardKinematics=True)
    position, orientation = p.getBasePositionAndOrientation(env.gripper.body)
    inverse = p.invertTransform(state[4], state[5])
    relative = p.multiplyTransforms(inverse[0], inverse[1], position, orientation)
    drift = np.linalg.norm(np.asarray(relative[0]) - env.rest_mount_relative[0])
    if drift > .015:
        raise PhysicsSafetyViolation(f"Gripper mount separated by {drift:.5f}m")
    env.safety_min_clearance = min(env.safety_min_clearance, clearance)
    env.safety_max_mimic_error = max(env.safety_max_mimic_error, error)
