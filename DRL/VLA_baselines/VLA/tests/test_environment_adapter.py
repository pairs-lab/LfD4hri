from __future__ import annotations

import unittest

import numpy as np
import pybullet as p

from VLA.common.environment import ManipulationEnv
from VLA.common.tasks import TASKS


class ExactJointTargetAdapterTest(unittest.TestCase):
    def test_fk_conversion_restores_measured_joint_state(self) -> None:
        with ManipulationEnv(image_size=16, seed=0) as env:
            observation = env.reset(TASKS["touch"])
            assert env.robot_id is not None
            # A reset starts nearly motionless. Give the arm real nonzero
            # velocities so accidentally clearing them cannot pass this test.
            for joint, position, velocity in zip(env.robot_spec.arm_joint_ids, observation.joint_positions,
                                                  (.3, -.4, .2, -.15, .1, -.2), strict=True):
                p.resetJointState(env.robot_id, joint, float(position), targetVelocity=velocity)
            before = np.asarray(
                [
                    p.getJointState(env.robot_id, joint)[:2]
                    for joint in env.robot_spec.arm_joint_ids
                ],
                dtype=np.float64,
            )
            targets = observation.joint_positions + np.asarray(
                (0.03, -0.02, 0.01, 0.015, 0.0, -0.01), dtype=np.float32
            )

            action = env.joint_targets_to_canonical(targets, 0.75)

            after = np.asarray(
                [
                    p.getJointState(env.robot_id, joint)[:2]
                    for joint in env.robot_spec.arm_joint_ids
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(after, before, atol=1e-12, rtol=0.0)
            self.assertEqual(action.shape, (7,))
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertAlmostEqual(float(action[6]), 0.5)
            self.assertTrue(np.all(np.abs(action[:3]) <= 0.05))
            self.assertTrue(np.all(np.abs(action[3:6]) <= 0.25))


if __name__ == "__main__":
    unittest.main()
