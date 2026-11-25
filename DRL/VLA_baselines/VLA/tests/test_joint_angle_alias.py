import unittest
import numpy as np

from VLA.smolvla.joint_angle_alias import aliases,range_proof


class JointAngleAliasTests(unittest.TestCase):
    def test_in_range_training_values_are_byte_identical(self):
        mean=np.array([1.,-1.,1.,-2.,4.71,-.2])
        value=mean.astype(np.float32);value[2]+=.1
        out,turns=aliases(value,mean)
        self.assertEqual(out.tobytes(),value.tobytes());self.assertFalse(turns.any())

    def test_extra_winding_is_the_same_physical_rotation(self):
        mean=np.array([1.,-1.,1.,-2.,4.71,-.2]);value=mean.astype(np.float32)
        value[5]-=np.float32(2*np.pi)
        out,turns=aliases(value,mean)
        self.assertEqual(turns[5],-1)
        np.testing.assert_allclose(np.sin(out),np.sin(value),atol=1e-6,rtol=0)
        np.testing.assert_allclose(np.cos(out),np.cos(value),atol=1e-6,rtol=0)

    def test_cannot_silently_wrap_a_source_with_multiple_windings(self):
        stats={"min":[-4.]*6,"max":[4.]*6}
        with self.assertRaises(ValueError):range_proof(stats,np.zeros(6))

    def test_batch_and_invalid_measurement_shapes(self):
        mean=np.zeros(6);value=np.zeros((2,6),dtype=np.float32);value[1,0]=np.float32(2*np.pi)
        out,turns=aliases(value,mean)
        self.assertEqual(turns[1,0],1);self.assertLess(abs(out[1,0]),1e-6)
        with self.assertRaises(ValueError):aliases(np.zeros(5),mean)


if __name__=="__main__":unittest.main()
