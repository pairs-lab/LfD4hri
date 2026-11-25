"""python -m unittest env_eval.case_demon.tests"""
import unittest

import numpy as np

from env_eval.case_demon.motion import JointPlan, MAX_ACCELERATION, MAX_VELOCITY
from env_eval.case_demon.robots import ROBOTS
from env_eval.case_demon.scenes import TASKS, build


class SceneTests(unittest.TestCase):
    def test_fifteen_distinct_tasks_and_layouts(self):
        scenes = [build(i) for i in range(1, 16)]
        self.assertEqual(len({(s.target, s.goal) for s in scenes}), 15)
        self.assertEqual(len({s.objects for s in scenes}), 15)
        self.assertEqual(len(TASKS), 15)

    def test_three_environments_per_robot(self):
        for robot in ROBOTS:
            self.assertEqual(sorted(build(i).env for i in range(1, 16) if build(i).robot == robot), [1, 2, 3])

    def test_deterministic_and_candidate_only_repacks(self):
        self.assertEqual(build(4), build(4))
        other = build(4, 1)
        self.assertEqual((other.target, other.goal), (build(4).target, build(4).goal))
        self.assertNotEqual(other.objects, build(4).objects)

    def test_no_overlap(self):
        for i in range(1, 16):
            objects = build(i).objects
            for a, (name, xy) in enumerate(objects):
                for other, position in objects[a+1:]:
                    self.assertGreater(np.linalg.norm(np.subtract(xy, position)), .05, (i, name, other))


class BaseFrameTests(unittest.TestCase):
    def test_yaw_maps_repo_workspace_to_drl_workspace(self):
        for robot in ('Panda', 'KUKA', 'Kinova'):
            yaw = ROBOTS[robot].base_yaw
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            np.testing.assert_allclose(rotation@[.5, 0], [0, -.5], atol=1e-9)


class JointPlanTests(unittest.TestCase):
    def test_boundary_velocities_and_knots(self):
        v0 = np.array([.2, -.1])
        plan = JointPlan([[0, 0], [.4, .2], [.5, .3]], [0, 1, 2], v0=v0)
        np.testing.assert_allclose(plan.sample(0)[1], v0, atol=1e-9)
        np.testing.assert_allclose(plan.sample(plan.duration)[1], 0, atol=1e-9)
        np.testing.assert_allclose(plan.sample(plan.duration)[0], [.5, .3], atol=1e-9)

    def test_time_stretches_to_respect_limits(self):
        plan = JointPlan([[0.], [3.], [6.]], [0, .2, .4])
        self.assertGreater(plan.duration, .4)
        self.assertLessEqual(plan.peak_velocity, MAX_VELOCITY*1.01)
        self.assertLessEqual(plan.peak_acceleration, MAX_ACCELERATION*1.01)


class PolicyStyleTests(unittest.TestCase):
    def test_deterministic_and_silent_when_envelope_is_zero(self):
        from env_eval.case_demon.gripper import PolicyStyleExecution
        plan = JointPlan([[0., 0.], [.5, .2], [.6, .3]], [0, 1, 2], degree=5)
        a = PolicyStyleExecution(np.random.default_rng(3), 2).targets(plan, lambda s: 1.)
        b = PolicyStyleExecution(np.random.default_rng(3), 2).targets(plan, lambda s: 1.)
        np.testing.assert_array_equal(a[1], b[1])
        times, quiet = PolicyStyleExecution(np.random.default_rng(3), 2).targets(plan, lambda s: 0.)
        np.testing.assert_allclose(quiet, [plan.sample(t)[0] for t in times])
        self.assertGreater(np.abs(a[1]-quiet).max(), 1e-4)


if __name__ == '__main__':
    unittest.main()
