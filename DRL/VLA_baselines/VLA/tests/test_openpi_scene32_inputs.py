import unittest
import numpy as np
from VLA.pi0.scene32_inputs import state32,statistics,RAW_SLOTS,FEATURES


def valid():
    v=np.zeros(64,np.float32);v[18:24]=[1,0,0,0,1,0];v[24:27]=[.04,.04,.06];v[27]=.08;return v


class Tests(unittest.TestCase):
    def test_complete32_causal_fields_and_wrapped_native_joints(self):
        raw=valid();raw[:6]=7;raw[11:14]=[.2,-.3,.1];raw[14:17]=[.12,.02,.03]
        x=state32(raw)
        self.assertEqual(len(RAW_SLOTS),32);self.assertEqual(len(FEATURES),32)
        np.testing.assert_array_equal(x[6:],raw[list(RAW_SLOTS)][6:]);np.testing.assert_allclose(x[:6],7-2*np.pi,atol=1e-6)

    def test_rare_contact_feedback_never_divides_by_zero(self):
        raw=np.stack([valid(),valid()]);stats,receipt=statistics(raw)
        self.assertTrue(receipt['first7_native_empirical_unchanged'])
        for i in receipt['binary_indices']:
            self.assertEqual(stats['mean'][i],.5);self.assertEqual(stats['std'][i],.5)
            self.assertEqual(stats['q01'][i],0);self.assertEqual(stats['q99'][i],1)
        self.assertTrue(np.all(np.asarray(stats['std'])[7:]>=.001-1e-9))

    def test_invalid_attachment_and_future_values_are_not_accepted(self):
        raw=valid();raw[17]=1
        with self.assertRaises(ValueError):state32(raw)
        raw=valid();raw[39]=np.nan
        with self.assertRaises(ValueError):state32(raw)


if __name__=='__main__':unittest.main()
