import unittest
import numpy as np
import pybullet as p
from utils.camera import Camera, CINEMATIC_VIEWS


class CinematicCameraTests(unittest.TestCase):
    def test_requested_poses_keep_original_optical_convention(self):
        for view, pos, quat in zip(CINEMATIC_VIEWS[3:],
                (Camera.front_position, Camera.side_position, Camera.right_side_position),
                (Camera.front_orientation, Camera.side_orientation, Camera.right_side_orientation)):
            r = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            np.testing.assert_allclose(view['eye'], pos)
            np.testing.assert_allclose(np.array(view['target'])-pos, r @ [0, 0, 1])
            np.testing.assert_allclose(view['up'], r @ [0, -1, 0])

    def test_two_sides_mirror_the_same_workspace(self):
        left, right = CINEMATIC_VIEWS[-2:]
        mirror = np.array([-1, 1, 1])
        for key in ('eye', 'target', 'up'):
            np.testing.assert_allclose(np.array(left[key])*mirror, right[key], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
