import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from env_eval.oracle_trajectory import PoseCurve, minimum_jerk, carry_curve
from env_eval.oracle_demo import parse_config, scenes


class TrajectoryTests(unittest.TestCase):
    def test_rest_endpoints_and_monotonic_timing(self):
        t=np.linspace(0,1,1001)
        s=minimum_jerk(t)
        self.assertTrue(np.all(np.diff(s)>=-1e-12))
        np.testing.assert_allclose(s[[0,-1]],[0,1])
        h=1e-4
        for endpoint in [0.,1.]:
            speed=(minimum_jerk(endpoint+h)-minimum_jerk(endpoint-h))/(2*h)
            acceleration=(minimum_jerk(endpoint+h)-2*minimum_jerk(endpoint)+minimum_jerk(endpoint-h))/h**2
            self.assertLess(abs(speed),1e-6)
            self.assertLess(abs(acceleration),.002)

    def test_shortest_quaternion_path_and_unit_norm(self):
        q0=Rotation.from_euler('z',170,degrees=True).as_quat()
        q1=Rotation.from_euler('z',-170,degrees=True).as_quat()
        curve=PoseCurve([[0,0,0],[1,2,3]],q0,q1,2)
        point,q=curve.sample(1)
        np.testing.assert_allclose(point,[.5,1,1.5])
        self.assertAlmostEqual(np.linalg.norm(q),1)
        self.assertAlmostEqual(abs(Rotation.from_quat(q).as_euler('xyz',degrees=True)[2]),180)

    def test_carry_is_single_curved_path_without_internal_rest(self):
        curve=carry_curve([0,0,.03],[.3,0,.05],[0,1,0,0],.20)
        pts=np.array([curve.sample(t)[0] for t in np.linspace(0,curve.duration,501)])
        self.assertGreater(pts[250,2],.20)
        self.assertTrue(np.all(np.linalg.norm(np.diff(pts,axis=0)[20:-20],axis=1)>1e-7))
        np.testing.assert_allclose(pts[[0,-1]],[[0,0,.03],[.3,0,.05]])

    def test_all_requested_scenes_and_front_destination(self):
        selected=scenes()
        self.assertEqual(len(selected),16)
        for s in selected:
            self.assertIn(s.pick,s.objects)
            self.assertEqual(len(s.task().scene_objects)+1+(s.task().place_name is not None),len(s.objects))
        front=selected[9].task()
        self.assertIsNone(front.place_name)
        self.assertLess(front.place_position[1],front.pick_position[1])

    def test_text_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scene.txt'
            path.write_text('pick object: blue block\nplace object: yellow bowl\nconfig: blue block, yellow bowl, egg\ncommand: put the blue block into the yellow bowl\n')
            scene=parse_config(path)
            self.assertEqual(scene.action,'put')
            self.assertEqual(scene.task().pick_name,'blue block')
        with self.assertRaises(ValueError):
            PoseCurve([[0,0,0],[1,1,1]],[0,0,0,1],[0,0,0,1],0)

if __name__=='__main__':unittest.main()
