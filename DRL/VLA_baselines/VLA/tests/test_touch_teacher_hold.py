from types import SimpleNamespace
import unittest
import numpy as np

from VLA.collision_planned_teacher import CollisionPlannedTeacher


class TouchTeacherHoldTest(unittest.TestCase):
    def test_touch_streak_never_commands_more_downward_motion_after_success(self):
        teacher = CollisionPlannedTeacher(SimpleNamespace(_last_success=True))
        teacher.task = SimpleNamespace(name="touch")
        # No object/pose lookup is required after success, and no motor CLOSE.
        action = teacher.predict(SimpleNamespace(step=10))
        np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, -1])
