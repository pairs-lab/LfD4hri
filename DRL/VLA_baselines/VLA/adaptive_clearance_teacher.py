"""Expert-only bounded clearance for a preserved low-profile gelatin scene.

Keeps the body-contact band and physics safeguards unchanged. Feasible height
is a proposal; actual replay/contact audit is still mandatory.
"""
import numpy as np
import pybullet as p
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.neighbour_clearance_teacher import NeighbourClearanceTeacher


def adaptive_midbody_height(requested, table_top, object_low, object_high):
    values = np.asarray([requested, table_top, object_low, object_high], float)
    if not np.isfinite(values).all() or object_high <= object_low:
        raise ValueError("Invalid body/table geometry")
    maximum = object_low + .80*(object_high-object_low)
    nominal = table_top + .0305
    # Do not lower the simulator floor guard. This expert proposal only uses
    # up to3mm less EXTRA margin, and stays strictly inside the same80% band.
    minimum = min(nominal, maximum-.0003)
    if minimum < table_top+.0275:
        raise ValueError("No safe clearance proposal inside the unchanged body band")
    height = max(float(requested), minimum)
    if height > maximum:
        raise ValueError("Requested grasp point exceeds the unchanged body band")
    return height


class AdaptiveClearanceTeacher(NeighbourClearanceTeacher):
    def target_and_rotation(self):
        target, yaw = CollisionPlannedTeacher.target_and_rotation(self)
        low, high = map(np.asarray, p.getAABB(self.env.pick_id))
        target[2] = adaptive_midbody_height(target[2], p.getAABB(self.env.table_id)[1][2], low[2], high[2])
        return target, yaw
