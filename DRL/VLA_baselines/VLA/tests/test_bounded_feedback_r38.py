import unittest
import numpy as np
from VLA.smolvla.bounded_feedback_r38 import bound


class BoundTests(unittest.TestCase):
    def test_in_support_measurements_bit_identical(self):
        x=np.linspace(-2,2,134,dtype=np.float32).reshape(2,67)
        result,changed=bound(x,np.zeros(67),np.ones(67))
        np.testing.assert_array_equal(result,x);self.assertFalse(changed.any())

    def test_extremes_bounded_and_sign_preserved(self):
        x=np.zeros(67,np.float32);x[32]=1.;x[33]=-1.
        mean=np.zeros(67,np.float32);std=np.full(67,.001,np.float32)
        y,changed=bound(x,mean,std)
        self.assertEqual(np.flatnonzero(changed).tolist(),[32,33])
        self.assertLessEqual(float(np.max(np.abs((y-mean)/(std+1e-8)))),10.00001)
        self.assertGreater(y[32],0);self.assertLess(y[33],0)
        self.assertEqual(x[32],1.)

    def test_epsilon_matches_native_zero_std(self):
        y,_=bound(np.ones(67,np.float32),np.zeros(67),np.zeros(67))
        np.testing.assert_allclose(y,np.full(67,1e-7),rtol=1e-7)

    def test_nonfinite_or_wrong_shape_rejected(self):
        for x in (np.zeros(66),np.full(67,np.nan)):
            with self.assertRaises(ValueError):bound(x,np.zeros(67),np.ones(67))
        with self.assertRaises(ValueError):bound(np.zeros(67),np.zeros(67),np.ones(67),limit=-1)


if __name__=='__main__':unittest.main()
