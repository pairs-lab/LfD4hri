"""Expert-only yaw ablation for a preserved pad/bowl collision.

Same robot, scene and guards. Analytic pad-centre clearance is a proposal,
NOT a collision certificate; every resulting trajectory still needs replay.
"""
import numpy as np
import pybullet as p

from VLA.clearance_correction_teacher import ClearanceCorrectionTeacher
from VLA.collision_grasp_planner import maximum_open_gap


class NoFullSupportYaw(ValueError):
    pass


def axis_obstacle_clearance(axis, centre, obstacle_boxes, open_gap):
    pads = np.asarray(centre)[:2] + np.array([-1., 1.])[:, None] * np.asarray(axis) * (open_gap / 2 + .006)
    clearances = []
    for low, high in obstacle_boxes:
        low, high = np.asarray(low)[:2], np.asarray(high)[:2]
        outside = np.maximum(np.maximum(low - pads, pads - high), 0)
        distance = np.linalg.norm(outside, axis=1)
        inside = np.all((pads >= low) & (pads <= high), axis=1)
        depth = np.minimum(pads - low, high - pads).min(axis=1)
        distance[inside] = -depth[inside]
        clearances.extend((distance - .012).tolist())
    return min(clearances, default=np.inf)


def ray_sampled_pad_sweep_clear(axis, centre, open_gap, obstacle_ids, start_z, bottom_z):
    """Refine a bounding-box false positive, NOT a full swept-volume proof."""
    pads = np.asarray(centre)[:2] + np.array([-1., 1.])[:, None] * np.asarray(axis) * (open_gap / 2 + .006)
    rays_from, rays_to = [], []
    offsets = [np.zeros(2)] + [.012*np.array([np.cos(a), np.sin(a)])
                              for a in np.linspace(0, 2*np.pi, 16, endpoint=False)]
    for pad in pads:
        for offset in offsets:
            xy = pad + offset
            rays_from.append([*xy, start_z])
            rays_to.append([*xy, bottom_z])
    return not any(hit[0] in obstacle_ids for hit in p.rayTestBatch(rays_from, rays_to))


def select_clearance_axis(points, centre, obstacle_boxes, open_gap):
    points, centre = np.asarray(points, float), np.asarray(centre, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite world collision vertices")
    options = []
    for angle in np.linspace(0, np.pi, 72, endpoint=False):
        axis = np.array([np.cos(angle), np.sin(angle)])
        support = (points[:, :2] - centre[:2]) @ axis
        if max(abs(support.min()), abs(support.max())) * 2 > open_gap - .004:
            continue
        options.append((axis_obstacle_clearance(axis, centre, obstacle_boxes, open_gap), axis))
    if not options:
        raise NoFullSupportYaw("No full-support yaw clears the unchanged open gripper")
    score, axis = max(options, key=lambda option: option[0])
    if score <= 0:
        raise ValueError("No analytically clear pad-centre yaw; do not collect unsafe labels")
    return axis, float(score)


class NeighbourClearanceTeacher(ClearanceCorrectionTeacher):
    def reset(self, task):
        super().reset(task)
        if task.name == "touch":
            return
        position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        centre = rotation @ self.point_local + position
        obstacles = [p.getAABB(body) for body in self.env.object_ids.values()
                     if body != self.env.pick_id]
        if not obstacles:
            return
        _, vertices = p.getMeshData(self.env.pick_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        vertices = np.asarray(vertices, float).reshape(-1, 3)
        if len(vertices) < 12:
            low, high = map(np.asarray, p.getAABB(self.env.pick_id))
            points = np.array([[x, y, z] for x in (low[0], high[0])
                               for y in (low[1], high[1]) for z in (low[2], high[2])])
        else:
            points = vertices @ rotation.T + position
        low, high = map(np.asarray, p.getAABB(self.env.pick_id))
        if np.any(points.min(axis=0) < low-.015) or np.any(points.max(axis=0) > high+.015):
            raise ValueError("Collision vertices disagree with world AABB")
        gap = maximum_open_gap(self.env)
        try:
            axis, score = select_clearance_axis(points, centre, obstacles, gap)
            support_kind = "full_target_support"
        except NoFullSupportYaw:
            # A curved elongated object can fit at its central grasp section
            # without fitting its entire projection inside the open jaws.
            # Do not freely rotate that valid section onto an invalid axis.
            if self.plan.get("method") != "collision_pca_central_cross_section" or self.axis_local is None:
                raise
            axis = (rotation @ self.axis_local)[:2]
            axis /= np.linalg.norm(axis)
            score = axis_obstacle_clearance(axis, centre, obstacles, gap)
            if score <= 0:
                # A round bowl does not occupy its AABB's empty XY corners.
                # Refine that conservative proposal with actual collision
                # rays, retaining the ORIGINAL validated cross-section yaw.
                target, _ = self.target_and_rotation()
                others = {body for body in self.env.object_ids.values() if body != self.env.pick_id}
                top = max(p.getAABB(body)[1][2] for body in others) + .05
                if not ray_sampled_pad_sweep_clear(axis, centre, gap, others, top, target[2]-.022):
                    raise ValueError("Existing central-section pad sweep hits a neighbour; no safe proposal")
                self.plan["bbox_clearance_refined_by_sparse_collision_rays"] = True
            support_kind = "preserved_original_central_cross_section"
        self.axis_local = rotation.T @ np.r_[axis, 0.]
        self.plan.update(neighbour_clearance_axis_world=axis.tolist(),
                         neighbour_clearance_support_kind=support_kind,
                         analytic_pad_centre_clearance_m=score,
                         analytic_proposal_is_collision_certificate=False)
