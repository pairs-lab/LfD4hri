"""Privileged teacher-only planar grasp planning; NEVER learned-policy inputs."""
import numpy as np
import pybullet as p

# Same collision-FK fully-open calibration used by BodyGraspEnvironment's
# encoder/aperture normalization, for its SHA-bound V9 Robotiq asset. Planning
# must use OPEN capacity, not the current (possibly fully closed) stroke.
V9_OPEN_PAD_GAP_M = .0885956060


def maximum_open_gap(env):
    if getattr(env, "observation_profile", {}).get("gripper_controller") == "body_contact_v9":
        return V9_OPEN_PAD_GAP_M
    return float(env.gripper.width())


def wrap_axis_angle(angle):
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def minimum_caliper_grasp(points):
    """Full XY support width, not a PCA/vertex-density approximation.

    The minimum support width of a convex polygon occurs normal to one of
    its hull edges. Using the full hull also accounts for the footprint that
    open fingers must clear while descending. This is teacher-only geometry.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Caliper planning requires finite Nx3 collision vertices")
    xy = sorted(set(map(tuple, points[:, :2])))
    if len(xy) < 3:
        raise ValueError("Degenerate XY collision footprint")

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    halves = []
    for ordered in (xy, xy[::-1]):
        half = []
        for point in ordered:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    hull = np.asarray(halves[0]+halves[1])
    if len(hull) < 3:
        raise ValueError("Degenerate XY collision footprint")
    options = []
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        along_axis = end-start
        along_axis /= np.linalg.norm(along_axis)
        jaw_axis = np.array([-along_axis[1], along_axis[0]])
        across, along = points[:, :2] @ jaw_axis, points[:, :2] @ along_axis
        centre = jaw_axis*(across.min()+across.max())/2 + along_axis*(along.min()+along.max())/2
        options.append((float(np.ptp(across)), jaw_axis, centre))
    width, jaw_axis, centre = min(options, key=lambda candidate: candidate[0])
    z = float((points[:, 2].min()+points[:, 2].max())/2)
    return {"point_world": np.r_[centre, z], "jaw_axis_world": jaw_axis,
            "cross_section_width_m": width,
            "touch_surface_world": np.r_[centre, points[:, 2].max()],
            "method": "collision_minimum_caliper_full_support"}


def plan_body_grasp(env, body):
    lo, hi = map(np.asarray, p.getAABB(body))
    fallback = (lo + hi) / 2
    try:
        _, vertices = p.getMeshData(body, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    except p.error:
        vertices = np.zeros((0, 3))
    if len(vertices) < 12:
        return {"point_world": fallback, "jaw_axis_world": None, "method": "central_aabb_for_primitives"}
    position, quaternion = p.getBasePositionAndOrientation(body)
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    points = vertices @ rotation.T + position
    # Reject a wrong mesh coordinate interpretation rather than silently
    # moving objects or training on an impossible grasp point.
    if np.any(points.min(axis=0) < lo - .015) or np.any(points.max(axis=0) > hi + .015):
        raise ValueError("Collision vertices do not match the world AABB")
    covariance = np.cov(points[:, :2].T)
    eigenvalues, axes = np.linalg.eigh(covariance)
    if eigenvalues[1] < 1.6 * max(eigenvalues[0], 1e-12):
        # Similar covariance eigenvalues do NOT prove that a footprint is
        # round or that it fits the open gripper at the current yaw. A flat
        # box previously sent both pads into its top while commanded OPEN.
        right, left = env.gripper.pad_positions()
        current_axis = np.asarray(left[:2]-right[:2], dtype=float)
        axis_length = float(np.linalg.norm(current_axis))
        gap = maximum_open_gap(env)
        if axis_length < 1e-6 or not np.isfinite(gap) or gap <= 0:
            raise ValueError("Caliper planning requires a valid measured open gripper aperture")
        current_axis /= axis_length
        clearance_margin = .004
        current_width = float(np.ptp(points[:, :2] @ current_axis))
        if current_width <= gap-clearance_margin:
            return {"point_world": fallback, "jaw_axis_world": None,
                    "method": "central_aabb_near_round",
                    "full_support_width_m": current_width, "open_gap_m": gap,
                    "descent_clearance_margin_m": clearance_margin}
        plan = minimum_caliper_grasp(points)
        if plan["cross_section_width_m"] > gap-clearance_margin:
            raise ValueError("No full near-round collision footprint fits the physical open gripper with descent clearance")
        if not (-.36 <= plan["point_world"][0] <= .36 and -.79 <= plan["point_world"][1] <= -.21):
            raise ValueError("Caliper grasp centre is outside the unchanged robot workspace")
        plan.update(full_support_width_m=plan["cross_section_width_m"],
                    current_yaw_support_width_m=current_width, open_gap_m=gap,
                    descent_clearance_margin_m=clearance_margin)
        return plan
    major, minor = axes[:, 1], axes[:, 0]
    along = points[:, :2] @ major
    across = points[:, :2] @ minor
    choices = []
    for quantile in (.5, .4, .6, .3, .7):
        value = float(np.quantile(along, quantile))
        band = np.abs(along - value) < .012
        if band.sum() < 8:
            continue
        low, high = np.quantile(across[band], [.02, .98])
        width = float(high - low)
        if .012 <= width <= .077:
            xy = major * value + minor * ((low + high) / 2)
            if not (-.36 <= xy[0] <= .36 and -.79 <= xy[1] <= -.21):
                continue
            z = float(np.quantile(points[band, 2], [.1, .9]).mean())
            top = float(np.quantile(points[band, 2], .95))
            choices.append((abs(quantile-.5), width, np.r_[xy, z], np.r_[xy, top]))
    if not choices:
        raise ValueError("No central collision cross-section fits the physical 85mm gripper")
    _, width, point, surface = min(choices, key=lambda x: (x[0], x[1]))
    return {"point_world": point, "jaw_axis_world": minor, "cross_section_width_m": width,
            "touch_surface_world": surface,
            "method": "collision_pca_central_cross_section"}
