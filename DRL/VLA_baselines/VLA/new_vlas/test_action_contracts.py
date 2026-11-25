import unittest

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from VLA.new_vlas.action_contracts import (absolute20_from_canonical,
    canonical_from_absolute20, native8, normalize_bounds, bounds_stats, pack_native20)


class ContractTests(unittest.TestCase):
    def test_absolute20_roundtrip_and_native_pybullet_rotation(self):
        rng = np.random.default_rng(202643)
        orientations = Rotation.random(100, random_state=rng)
        six = orientations.as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        xyz = np.tile([0., -.5, .4], (100, 1))
        actions = rng.uniform(-.02, .02, (100, 7))
        actions[:, 3:6] = rng.uniform(-.20, .20, (100, 3))
        actions[:, 6] = rng.choice([-1., 1.], 100)
        target = absolute20_from_canonical(xyz, six, actions)
        np.testing.assert_allclose(canonical_from_absolute20(target, xyz, six), actions, atol=1e-7, rtol=0)
        for q, action, result in zip(orientations.as_quat(), actions, target, strict=True):
            native = p.multiplyTransforms((0, 0, 0), q, (0, 0, 0),
                                          p.getQuaternionFromEuler(action[3:6]))[1]
            matrix = np.asarray(p.getMatrixFromQuaternion(native)).reshape(3, 3)
            np.testing.assert_allclose(result[3:9], matrix[:, :2].reshape(6), atol=1e-6, rtol=0)
        np.testing.assert_equal(target[:, 9], (actions[:, 6]+1)/2)

    def test_adapter_gripper_not_inverted_or_normalized_twice(self):
        values = np.zeros((10, 7), np.float32)
        values[:, 0] = np.linspace(-.03, .03, 10)
        values[::2, 6] = 1
        stats = bounds_stats(values, mask=[True]*6+[False])
        normalized = normalize_bounds(values, stats)
        np.testing.assert_array_equal(normalized[:, 6], values[:, 6])
        # The official 1e-8 denominator epsilon creates this tiny offset for
        # near-constant channels; native unnormalization cancels it exactly.
        np.testing.assert_allclose(normalized[:, 1:6], 0, atol=6e-5, rtol=0)
        low, high = (np.asarray(stats[k]) for k in ('q01', 'q99'))
        decoded = .5*(normalized+1)*(high-low+1e-8)+low
        np.testing.assert_allclose(decoded[:, 1:6], 0, atol=1e-10, rtol=0)

    def test_native8_is_current_measured_pose_and_aperture(self):
        result = native8(np.array([0., -.5, .4]), np.array([1, 0, 0, 0, 1, 0]), .08)
        np.testing.assert_allclose(result, [0, -.5, .4, 0, 0, 0, .04, -.04], atol=1e-7)

    def test_selected_UR_domain_matches_ACTUAL_official_converter(self):
        import ast
        from pathlib import Path
        source = Path(__file__).resolve().parents[1]/'third_party/xvla/datasets/utils.py'
        parsed = ast.parse(source.read_text())
        # Execute actual pure official functions, without importing unrelated
        # dataset packaging dependencies. No rewritten stand-in converter.
        selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('euler_to_rotate6d', 'rotate6d_to_quat')]
        namespace = {'np': np, 'R': Rotation}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        eulers = np.random.default_rng(202644).uniform(-2, 2, (100, 3))
        matrices = Rotation.from_euler('xyz', eulers).as_matrix()
        six = matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        packed = pack_native20(np.zeros((100, 3)), six, np.zeros(100))
        expected = namespace['euler_to_rotate6d'](eulers)
        np.testing.assert_allclose(packed[:, 3:9], expected, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(packed[:, 10:], 0)
        decoded = Rotation.from_quat(namespace['rotate6d_to_quat'](packed[:, 3:9])).as_matrix()
        np.testing.assert_allclose(decoded, matrices, atol=2e-7, rtol=0)

    def test_native8_home_pi_branch_is_continuous_and_rotation_equivalent(self):
        angles = np.linspace(np.pi-.02, np.pi+.02, 101)
        matrices = Rotation.from_euler('xyz', np.c_[np.zeros(101), angles, np.zeros(101)]).as_matrix()
        six = matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        state = native8(np.zeros((101, 3)), six, np.full(101, .08))
        self.assertLess(float(np.linalg.norm(np.diff(state[:, 3:6], axis=0), axis=1).max()), .001)
        np.testing.assert_allclose(Rotation.from_rotvec(state[:, 3:6]).as_matrix(), matrices, atol=2e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
