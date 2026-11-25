"""OFFLINE ONLY: preserve a blocked elongated scene, try other BODY sections.

The old planner validates five central sections but throws four away before
checking neighbours. This extension is invoked ONLY after its exact original
clearance rejection. No object movement, larger aperture, success relaxation,
runtime privileged input or learned-policy fallback. Proposals require real
canonical/native replay before dataset qualification.
"""
import numpy as np

from VLA.neighbour_clearance_teacher import axis_obstacle_clearance, ray_sampled_pad_sweep_clear
from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher


def central_section_options(points, gap):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12 or not np.isfinite(points).all():
        raise ValueError("Finite collision mesh vertices required")
    if not np.isfinite(gap) or gap <= .004:
        raise ValueError("Invalid physical open aperture")
    eigenvalues, axes = np.linalg.eigh(np.cov(points[:, :2].T))
    if eigenvalues[1] < 1.6*max(eigenvalues[0], 1e-12):
        return []
    major, minor = axes[:, 1], axes[:, 0]
    along, across = points[:, :2]@major, points[:, :2]@minor
    options = []
    for quantile in (.5, .4, .6, .3, .7):
        value = float(np.quantile(along, quantile))
        band = np.abs(along-value) < .012
        if band.sum() < 8:
            continue
        lower, upper = np.quantile(across[band], [.02, .98])
        width = float(upper-lower)
        xy = major*value+minor*((lower+upper)/2)
        # Unlike a width-only quantile check, also retain ALL observed local
        # support within the original open capacity with the same4mm margin.
        full_extent = float(2*np.max(np.abs(across[band]-(lower+upper)/2)))
        if (not .012 <= width <= .077 or full_extent > gap-.004
                or not (-.36 <= xy[0] <= .36 and -.79 <= xy[1] <= -.21)):
            continue
        z = float(np.quantile(points[band, 2], [.1, .9]).mean())
        options.append({"point_world": np.r_[xy, z].tolist(),
            "jaw_axis_world": minor.tolist(), "cross_section_width_m": width,
            "full_local_section_support_m": full_extent, "central_section_quantile": quantile,
            "touch_surface_world": np.r_[xy, np.quantile(points[band, 2], .95)].tolist()})
    return options


class CentralSectionClearanceTeacher(YawClearTouchTeacher):
    def _alternate_clearance_plan(self, points, obstacles, boxes, gap, top, bottom):
        if self.plan.get("method") != "collision_pca_central_cross_section":
            return None
        proposals = []
        for option in central_section_options(points, gap):
            score = axis_obstacle_clearance(option["jaw_axis_world"], option["point_world"], boxes, gap)
            ray_clear = None if score > .004 else ray_sampled_pad_sweep_clear(
                option["jaw_axis_world"], option["point_world"], gap, obstacles, top, bottom)
            option.update(analytic_pad_centre_clearance_m=float(score),
                bbox_clearance_refined_by_sparse_collision_rays=bool(ray_clear),
                proposed_clear=bool(score > .004 or ray_clear))
            proposals.append(option)
        safe = [row for row in proposals if row["proposed_clear"]]
        self.central_section_proposals = proposals
        if not safe:
            return None
        selected = max(safe, key=lambda row: (row["analytic_pad_centre_clearance_m"],
                                             -abs(row["central_section_quantile"]-.5)))
        return {**selected, "neighbour_clearance_axis_world": selected["jaw_axis_world"],
            "neighbour_clearance_support_kind": "alternate_validated_central_body_section",
            "central_section_proposals": proposals,
            "original_point_rejected_by_unchanged_clearance": True,
            "analytic_proposal_is_collision_certificate": False}
