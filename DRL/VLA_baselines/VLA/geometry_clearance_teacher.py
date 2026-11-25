"""Expert-only low-body proposals bounded by ACTUAL whole-gripper clearance.

No static guessed minimum, no change to controller/floor guard/body band.
Top-down vertical distance proposals still require strict physical replay.
"""
import numpy as np
import pybullet as p
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.ray_refined_clearance_teacher import RayRefinedClearanceTeacher


def geometry_midbody_height(requested, table_top, low, high, physical_floor):
    if not np.isfinite([requested, table_top, low, high, physical_floor]).all() or high <= low:
        raise ValueError("Invalid body/floor geometry")
    maximum = low+.8*(high-low)
    preferred = min(table_top+.0305, maximum-.0003)
    height = max(float(requested), preferred, float(physical_floor))
    if height > maximum:
        raise ValueError("Actual whole-gripper floor conflicts with the unchanged body band")
    return height


class GeometryClearanceTeacher(RayRefinedClearanceTeacher):
    def target_and_rotation(self):
        target, yaw = CollisionPlannedTeacher.target_and_rotation(self)
        # Nearest actual collision surfaces, not their inflated AABBs. Only a
        # vertical top-down proposal is justified by this translation bound.
        contacts = [c for body in self.env.ground_bodies for c in
                    p.getClosestPoints(self.env.gripper.body, body, distance=2.)]
        if not contacts:
            raise ValueError("No actual ground clearance geometry")
        closest = min(contacts, key=lambda c: c[8])
        if closest[7][2] < .999:
            raise ValueError("Nonvertical ground normal; no height-only clearance proposal")
        tcp = self.env.gripper.grasp_center()
        floor = float(tcp[2]-closest[8]+.0035)
        low, high = map(np.asarray, p.getAABB(self.env.pick_id))
        target[2] = geometry_midbody_height(target[2], p.getAABB(self.env.table_id)[1][2], low[2], high[2], floor)
        self.plan["actual_ground_clearance_proposal_margin_m"] = .0035
        self.plan["height_proposal_is_collision_certificate"] = False
        return target, yaw
