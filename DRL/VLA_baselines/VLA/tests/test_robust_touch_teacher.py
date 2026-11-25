import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from VLA.robust_touch_teacher import limit_near_contact_motion, GentleForceTouchTeacher


class RobustTouchTeacherTest(unittest.TestCase):
    def test_no_blanket_slowdown_or_input_mutation(self):
        action = np.array([.004, -.004, -.004, 0, 0, 0, -1], np.float32)
        for distance in (None, .011):
            np.testing.assert_array_equal(limit_near_contact_motion(action, distance, "descend"), action)
        np.testing.assert_array_equal(limit_near_contact_motion(action, .001, "approach"), action)
        slowed = limit_near_contact_motion(action, .005, "descend")
        np.testing.assert_allclose(slowed[:3], [.002, -.002, -.002])
        np.testing.assert_array_equal(slowed[3:], action[3:])
        self.assertEqual(float(action[0]), float(np.float32(.004)))

    @patch("VLA.robust_touch_teacher.RobustTouchTeacher.predict")
    @patch("VLA.robust_touch_teacher.p.getClosestPoints")
    def test_gentle_final_millimetre_cap_does_not_change_gripper(self, closest, predict):
        teacher = GentleForceTouchTeacher.__new__(GentleForceTouchTeacher)
        teacher.env = SimpleNamespace(gripper=SimpleNamespace(body=1), pick_id=2)
        teacher.task = SimpleNamespace(name="touch")
        teacher.phase = "descend"
        action = np.array([.002, -.002, -.002, 0, 0, 0, -1], np.float32)
        for contacts, expected in (([], .002), ([(0,)], .00004)):
            closest.return_value = contacts
            predict.return_value = action.copy()
            actual = teacher.predict(None)
            np.testing.assert_allclose(actual[:3], [expected, -expected, -expected])
            np.testing.assert_array_equal(actual[3:], action[3:])
