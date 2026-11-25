import unittest
import tempfile
from pathlib import Path
import numpy as np
import pybullet as p
from env_eval.legacy_gripper_environment import LegacyGripperEnvironment
from env_eval.oracle_demo import scenes


class BowlPhysicsTests(unittest.TestCase):
    def test_static_bowl_is_grounded_and_has_a_real_interior(self):
        with tempfile.TemporaryDirectory() as folder:
            with LegacyGripperEnvironment('uf850', Path(folder)) as env:
                env.reset(scenes()[14].task())
                lo, hi = map(np.array, p.getAABB(env.place_id))
                self.assertAlmostEqual(lo[2], p.getAABB(env.table_id)[1][2], places=5)
                centre = (lo+hi)/2
                p.resetBasePositionAndOrientation(env.pick_id,
                    [centre[0], centre[1], hi[2]+.08], [0, 0, 0, 1])
                for _ in range(480):
                    env.gripper.apply_motor_control()
                    p.stepSimulation()
                bottom = p.getAABB(env.pick_id)[0][2]
                self.assertLess(bottom, hi[2]-.01)
                self.assertTrue(p.getContactPoints(env.pick_id, env.place_id))
                self.assertLess(np.linalg.norm(p.getBaseVelocity(env.pick_id)[0]), .005)


if __name__ == '__main__':
    unittest.main()
