import unittest
import numpy as np

from VLA.smolvla.robust_encoder_input import nearstatic_clip


class RobustEncoderInputTests(unittest.TestCase):
    def stats(self):
        return {"mean":[1,1,1,1,4.7123,1],"std":[.5,.5,.5,.5,.0003,.5],
                "min":[0,0,0,0,4.711,0],"max":[2,2,2,2,4.713,2]}

    def test_in_range_training_values_byte_unchanged(self):
        data = np.array([[1,1,1,1,4.7127,1],[1,1,1,1,4.711,1]],dtype=np.float32)
        actual, changed = nearstatic_clip(data,self.stats())
        self.assertEqual(actual.tobytes(),data.tobytes())
        self.assertFalse(changed.any())

    def test_only_small_nearstatic_excursion_bounded(self):
        data = np.array([[3,1,1,1,4.715,1]],dtype=np.float32)
        actual, changed = nearstatic_clip(data,self.stats())
        self.assertEqual(actual[0,0],3)
        self.assertAlmostEqual(float(actual[0,4]),4.713,places=6)
        self.assertEqual(np.flatnonzero(changed).tolist(),[4])

    def test_meaningful_rotation_not_hidden(self):
        data = np.array([[1,1,1,1,4.8,1]],dtype=np.float32)
        actual, changed = nearstatic_clip(data,self.stats())
        self.assertEqual(actual.tobytes(),data.tobytes())
        self.assertFalse(changed.any())
