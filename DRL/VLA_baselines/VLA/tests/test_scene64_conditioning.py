import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from VLA.scene64_conditioning import NUMERIC_NAMES, encode_numeric, augment_prompt, parse_prompt, validate_scene
from VLA.scene64_dataset import Scene64NativeDataset


def valid():
    v = np.zeros(64, np.float32); v[18:24] = [1, 0, 0, 0, 1, 0]
    v[24:27] = [.04, .05, .06]; v[27] = .08
    return v


class Tests(unittest.TestCase):
    def test_numeric_keeps_all_first35_and_smoothly_wraps_target_orientation(self):
        left, right = valid(), valid(); left[37] = np.pi-1e-5; right[37] = -np.pi+1e-5
        a, b = encode_numeric(left), encode_numeric(right)
        self.assertEqual(len(NUMERIC_NAMES), 67); self.assertEqual(a.shape, (67,))
        np.testing.assert_array_equal(a[:35], left[:35]); np.testing.assert_array_equal(a[41:], left[38:])
        self.assertLess(float(np.max(np.abs(a[35:41]-b[35:41]))), 3e-5)

    def test_all64_prompt_fields_roundtrip_no_lost_neighbors(self):
        v = valid(); v[40:44] = [.123456, -.2, .03, .04]
        text = augment_prompt("Pick the grey block", v)
        np.testing.assert_allclose(parse_prompt(text), v, atol=.000051, rtol=0)
        self.assertIn("neighbor5_relative_xyz_and_xy_radius_m", text)
        with self.assertRaises(ValueError): augment_prompt(text, v)
        with self.assertRaises(ValueError): parse_prompt(text.split("; neighbor5")[0])

    def test_feedback_fail_closed(self):
        for index, value in ((17, 1), (30, 1), (24, 0), (39, -1), (63, -1)):
            v = valid(); v[index] = value
            with self.assertRaises(ValueError): validate_scene(v)
        v = valid(); v[44:48] = [.1, .1, .1, .02]
        with self.assertRaises(ValueError): validate_scene(v)

    def test_batched_fetch_never_bypasses_native_overlay(self):
        fake = object.__new__(Scene64NativeDataset)
        with patch.object(Scene64NativeDataset, "__getitem__", autospec=True, side_effect=lambda obj, i: i+100):
            self.assertEqual(Scene64NativeDataset.__getitems__(fake, [0, 2]), [100, 102])


if __name__ == "__main__": unittest.main()
