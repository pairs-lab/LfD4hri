import unittest
import numpy as np
from VLA.surface_limited_touch_teacher import surface_limited_motion


def contact(gap,normal=(0,0,1)):
    return tuple([0]*7+[normal,gap])


class SurfaceLimitedTouchTeacherTest(unittest.TestCase):
    def test_bounds_coarse_approach_and_fine_contact_without_changing_gripper(self):
        action=np.array([0,0,-.002,0,0,0,-1],np.float32)
        for gap,expected in ((.0012,.0006),(.0001,.00002),(-.0001,.00002)):
            result=surface_limited_motion(action,[contact(gap)])
            self.assertAlmostEqual(float(result[2]),-expected,places=9)
            np.testing.assert_array_equal(result[3:],action[3:])
        self.assertEqual(float(action[2]),float(np.float32(-.002)))

    def test_does_not_slow_tangential_or_separating_motion(self):
        for delta in ((.002,0,0),(0,0,.002)):
            action=np.r_[delta,[0,0,0,-1]].astype(np.float32)
            np.testing.assert_array_equal(surface_limited_motion(action,[contact(.0001)]),action)

    def test_all_surface_normals_constrain_the_same_translation(self):
        action=np.array([-.002,0,-.002,0,0,0,-1],np.float32)
        actual=surface_limited_motion(action,[contact(.0004),contact(.0001,(1,0,0))])
        np.testing.assert_allclose(actual[:3],[-.00002,0,-.00002])
