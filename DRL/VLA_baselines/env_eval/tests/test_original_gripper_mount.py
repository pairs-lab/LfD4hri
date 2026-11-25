import unittest
import tempfile
from pathlib import Path
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from utils.Robotiq2F85 import Robotiq2F85
from utils.Robotiq2F85UF850 import Robotiq2F85UF850
from env_eval.legacy_gripper_environment import LegacyGripperEnvironment
from env_eval.oracle_demo import scenes


class OriginalMountTests(unittest.TestCase):
    def test_original_class_stays_flush_during_rotation_and_stop(self):
        for robot in ('uf850', 'ur5e'):
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as folder:
                with LegacyGripperEnvironment(robot, Path(folder)) as env:
                    env.reset(scenes()[0].task())
                    expected = Robotiq2F85UF850 if robot == 'uf850' else Robotiq2F85
                    self.assertIs(type(env.gripper), expected)
                    self.assertNotEqual(env.gripper.body, env.robot_id)
                    info = p.getConstraintInfo(env.gripper.mount_constraint)
                    self.assertEqual(info[3], -1 if robot == 'uf850' else 0)
                    self.assertEqual(info[4], p.JOINT_FIXED)
                    original = env.q.copy()
                    for step in range(600):
                        target = original.copy()
                        if step < 360:
                            target[-1] += .45*np.sin(2*np.pi*step/360)
                        env._apply_arm_targets(target)
                        env.gripper.apply_motor_control()
                        p.stepSimulation()
                        env.gripper.post_step()
                    self.assertLess(env.gripper.max_mount_translation_error, .0001)
                    self.assertLess(env.gripper.max_mount_rotation_error, .05)
                    flange = p.getLinkState(env.robot_id, env.robot_spec.mount_link, computeForwardKinematics=True)
                    root = p.getBasePositionAndOrientation(env.gripper.body)
                    relative = Rotation.from_quat(flange[5]).inv().apply(np.asarray(root[0]) - flange[4])
                    self.assertLess(np.linalg.norm(relative - env.gripper.mount_offset), .0001)
                    if robot == 'uf850':
                        seat = np.asarray(root[0])
                        self.assertLess(np.linalg.norm(seat - flange[4]), .0001)

    def test_broad_mesh_mounting_face_is_seated_not_the_locating_lip(self):
        import trimesh
        import utils.Robotiq2F85 as module
        mesh=trimesh.load(Path(module.ROOT_PATH)/'asset/robotiq_2f_85/robotiq-2f-base.obj',force='mesh',process=False)
        selected=mesh.face_normals[:,2]<-.999
        z=np.round(mesh.triangles_center[selected,2]*.1,5)
        areas=mesh.area_faces[selected]*.01
        planes={value:areas[z==value].sum() for value in set(z) if value<.01}
        main_plane=max(planes,key=planes.get)
        gap=Robotiq2F85UF850.MOUNT_OFFSET[2]+main_plane
        self.assertLess(abs(gap),.0002)


if __name__ == '__main__':
    unittest.main()
