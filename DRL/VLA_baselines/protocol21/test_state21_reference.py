import unittest
import numpy as np
from state21_reference import NAMES, pack, fit_train_stats, normalize, model_state, pi0_state_text


class ContractTests(unittest.TestCase):
    def fixture(self, **changes):
        args = dict(q=[0, .1, .2, .3, .4, .5], closure01=.25,
                    pad_midpoint_world=[.1, -.5, .2], previous_signed_close=-1.,
                    tool_to_world_rotation=np.eye(3), surface_jaw_gap_m=.04,
                    left_external_normal_force_N=.1, right_external_normal_force_N=.2)
        args.update(changes)
        return pack(**args)

    def test_order(self):
        x = self.fixture()
        self.assertEqual(len(NAMES), 21)
        np.testing.assert_allclose(x[11:17], [1, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(x[18:], [0, 1, .3])

    def test_strict_contact_threshold(self):
        x = self.fixture(left_external_normal_force_N=.10001, right_external_normal_force_N=0.)
        np.testing.assert_array_equal(x[18:20], [1, 0])

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            self.fixture(tool_to_world_rotation=np.zeros((3, 3)))

    def test_no_task_fields_accepted(self):
        with self.assertRaises(TypeError):
            self.fixture(target_position=[1, 2, 3])

    def test_normalizer(self):
        x = np.stack([self.fixture(), self.fixture(closure01=.75)])
        stats = fit_train_stats(x)
        z = normalize(x, stats)
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_allclose(z[:, 6], [-1, 1])
        np.testing.assert_array_equal(z[:, 18:20], x[:, 18:20])

    def test_method_shapes(self):
        raw = self.fixture()
        stats = fit_train_stats([raw])
        for family in ('flower', 'smolvla', 'vla_jepa', 'falcon_esm'):
            self.assertEqual(model_state(raw, stats, family).shape, (21,))
        for family, width in (('xvla', 20), ('vla_adapter', 8), ('dita', 7), ('vlanext_s', 7)):
            self.assertEqual(model_state(raw, stats, family, native_prefix=np.zeros(width)).shape, (width+21,))
        seer = model_state(raw, stats, 'seer', native_prefix=np.zeros(7))
        self.assertEqual(seer['native7'].shape, (7,))
        self.assertEqual(seer['robot21'].shape, (21,))
        pi = model_state(raw, stats, 'pi0')
        self.assertEqual(pi.shape, (32,))
        np.testing.assert_array_equal(pi[21:], np.zeros(11))

    def test_prompt(self):
        prompt = pi0_state_text(self.fixture())
        self.assertEqual(len(prompt.split('; ')), 21)
        self.assertNotIn('goal', prompt)
        self.assertNotIn('target', prompt)

    def test_reject_old_statistics(self):
        with self.assertRaises(ValueError):
            normalize(self.fixture(), {'mean': [0]*67, 'std': [1]*67})


if __name__ == '__main__':
    unittest.main()
