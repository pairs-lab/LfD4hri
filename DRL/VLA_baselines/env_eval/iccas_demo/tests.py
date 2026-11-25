"""Small CPU tests: scene coverage, real cavity, nonzero prop inertia, layouts."""
import unittest
from pathlib import Path
import numpy as np
import pybullet as p
from env_eval.camera_editions import rectangles
from .assets import cup,brush
from .scenes import SCENES
from .render import SETTINGS,VIEWS


class Contracts(unittest.TestCase):
    def test_exact_scene_robot_camera_coverage(self):
        self.assertEqual(len(SCENES),10);self.assertEqual(10*2*len(VIEWS),80)
        self.assertEqual(VIEWS,(2,3,4,6))
        self.assertEqual([len(s.objects) for s in SCENES],[5,6,6,6,6,6,7,7,10,10])
        for s in SCENES:
            self.assertEqual(len(dict(s.objects)),len(s.objects))
            for i,step in enumerate(s.steps):
                t=s.task(i)
                self.assertEqual(len(t.scene_objects)+(2 if step.place else 1),len(s.objects))
    def test_duplicate_blue_blocks_are_two_physical_entities(self):
        names=dict(SCENES[4].objects)
        self.assertIn('blue block',names);self.assertIn('blue block 2',names)
        self.assertNotEqual(names['blue block'],names['blue block 2'])
    def test_multitask_order(self):
        self.assertEqual([s.pick for s in SCENES[7].steps],['orange orange','yellow tennis_ball','red apple'])
        self.assertEqual([s.place for s in SCENES[8].steps],['red bowl','blue bowl','yellow bowl','orange bowl'])
    def test_native_standard_inset(self):
        self.assertEqual(rectangles('front',.31,'left',SETTINGS),[('front',35,37,594,334)])
    def test_physical_hollow_cup_and_nonzero_brush_inertia(self):
        client=p.connect(p.DIRECT)
        try:
            body=cup((0,0,0))
            center=p.rayTest((0,0,.2),(0,0,-.01))[0]
            rim=p.rayTest((.052,0,.2),(.052,0,-.01))[0]
            self.assertEqual(center[0],body);self.assertLess(center[3][2],.01)
            self.assertGreater(rim[3][2],.08)
            b=brush((.3,0,.03));self.assertTrue(all(v>0 for v in p.getDynamicsInfo(b,-1)[2]))
            for obj in (body,b):
                path=p.getVisualShapeData(obj)[0][4].decode()
                self.assertTrue(Path(path).is_file());self.assertTrue(path.endswith('.obj'))
        finally:p.disconnect(client)


if __name__=='__main__':unittest.main()
