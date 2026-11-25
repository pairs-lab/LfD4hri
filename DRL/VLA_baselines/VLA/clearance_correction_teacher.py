"""Expert-only correction teacher; unchanged V9 robot/controller/success guards.

An observed empty-close correction contacted the TABLE with a pad (>200N),
which blocked closure without tripping penetration limits. Keep low-object
expert waypoints higher while retaining the required midbody contact band.
This is not a learned-policy fallback or a modification of an active collector.
"""
import numpy as np
import pybullet as p
from VLA.collision_planned_teacher import CollisionPlannedTeacher


def safe_midbody_height(requested, table_top, object_low, object_high):
    minimum = table_top + .0305
    maximum = object_low + .80*(object_high-object_low)
    height = max(float(requested), minimum)
    if height > maximum:
        raise ValueError("No declared table-clearance waypoint inside the unchanged 80pct body band")
    return height


class ClearanceCorrectionTeacher(CollisionPlannedTeacher):
    def target_and_rotation(self):
        target, yaw = super().target_and_rotation()
        low, high = map(np.asarray, p.getAABB(self.env.pick_id))
        target[2] = safe_midbody_height(target[2], p.getAABB(self.env.table_id)[1][2], low[2], high[2])
        return target, yaw
