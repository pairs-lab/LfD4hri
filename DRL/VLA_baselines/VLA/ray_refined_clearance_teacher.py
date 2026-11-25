"""Expert-only refinement of conservative neighbour AABBs.

Keeps width, target centre, robot and safety limits unchanged. Sparse actual
collision rays only nominate a yaw; strict trajectory/contact replay remains
mandatory. Never supplied as a fallback to a learned policy.
"""
import numpy as np
import pybullet as p
from VLA.adaptive_clearance_teacher import AdaptiveClearanceTeacher
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.collision_grasp_planner import maximum_open_gap
from VLA.neighbour_clearance_teacher import axis_obstacle_clearance, ray_sampled_pad_sweep_clear


def full_support_options(points, centre, gap):
    points, centre = np.asarray(points, float), np.asarray(centre, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid world collision vertices")
    result = []
    for angle in np.linspace(0, np.pi, 72, endpoint=False):
        axis = np.array([np.cos(angle), np.sin(angle)])
        projection = (points[:, :2]-centre[:2]) @ axis
        if 2*max(abs(projection.min()), abs(projection.max())) <= gap-.004:
            result.append(axis)
    return result


class RayRefinedClearanceTeacher(AdaptiveClearanceTeacher):
    def _alternate_clearance_plan(self, points, obstacles, boxes, gap, top, bottom):
        # Explicit offline extension point. The historical teacher retains
        # its original point and rejection unless a subclass nominates a
        # separately audited body-section proposal.
        return None

    def reset(self, task):
        CollisionPlannedTeacher.reset(self, task)
        if task.name == "touch":
            return
        position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        centre = rotation @ self.point_local + position
        others = {body for body in self.env.object_ids.values() if body != self.env.pick_id}
        if not others:
            return
        boxes = [p.getAABB(body) for body in sorted(others)]
        _, vertices = p.getMeshData(self.env.pick_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        vertices = np.asarray(vertices, float).reshape(-1, 3)
        low, high = map(np.asarray, p.getAABB(self.env.pick_id))
        points = (vertices @ rotation.T + position) if len(vertices) >= 12 else np.array(
            [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        if np.any(points.min(axis=0) < low-.015) or np.any(points.max(axis=0) > high+.015):
            raise ValueError("Collision vertices disagree with world AABB")
        gap = maximum_open_gap(self.env)
        options = full_support_options(points, centre, gap)
        kind = "full_target_support"
        if not options:
            if self.plan.get("method") != "collision_pca_central_cross_section" or self.axis_local is None:
                raise ValueError("No unchanged full-support or validated central-section yaw")
            axis = (rotation @ self.axis_local)[:2]
            options = [axis / np.linalg.norm(axis)]
            kind = "preserved_original_central_cross_section"
        ranked = sorted(((axis_obstacle_clearance(axis, centre, boxes, gap), axis)
                         for axis in options), key=lambda item: item[0], reverse=True)
        target, _ = self.target_and_rotation()
        top = max(box[1][2] for box in boxes)+.05
        for score, axis in ranked:
            if score > 0 or ray_sampled_pad_sweep_clear(axis, centre, gap, others, top, target[2]-.022):
                self.axis_local = rotation.T @ np.r_[axis, 0.]
                self.plan.update(neighbour_clearance_axis_world=axis.tolist(),
                    neighbour_clearance_support_kind=kind, analytic_pad_centre_clearance_m=float(score),
                    bbox_clearance_refined_by_sparse_collision_rays=bool(score <= 0),
                    analytic_proposal_is_collision_certificate=False)
                return
        alternate = self._alternate_clearance_plan(points, others, boxes, gap, top, target[2]-.022)
        if alternate is not None:
            self.point_local = rotation.T @ (np.asarray(alternate["point_world"])-position)
            self.axis_local = rotation.T @ np.r_[alternate["jaw_axis_world"], 0.]
            self.touch_surface_local = rotation.T @ (np.asarray(alternate["touch_surface_world"])-position)
            self.plan.update(alternate)
            return
        raise ValueError("All width-valid pad-sweep proposals hit neighbours; preserve scene")
