import unittest
import numpy as np

from VLA.rdt.audit_signal_precision import roundtrip_bf16


class SignalPrecisionTest(unittest.TestCase):
    def test_absolute_joint_fine_motion_can_disappear_despite_float32_masters(self):
        values=np.array([[2.,-1.5],[2.002,-1.498]],np.float32)
        self.assertFalse(np.array_equal(values[0],values[1]))
        rounded=roundtrip_bf16(values)
        self.assertEqual(rounded.dtype,np.float32)
        np.testing.assert_array_equal(rounded[0],rounded[1])

    def test_exact_binary_representable_values_are_preserved(self):
        values=np.array([0.,.5,-1.,2.],np.float32)
        np.testing.assert_array_equal(roundtrip_bf16(values),values)


if __name__=='__main__':unittest.main()
