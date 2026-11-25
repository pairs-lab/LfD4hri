import unittest
import numpy as np
from VLA.maintenance.storage_baseline_r91 import reason,ROOT
from .native import normalize,denormalize
from .data import geometry


class Contracts(unittest.TestCase):
    def test_gripper_single_inversion(self):
        stats={'actions':{'q01':[-.1]*7,'q99':[.1]*7}}
        values=np.array([[.01,-.02,.03,.04,-.05,.06,-1.],[-.01,.02,-.03,-.04,.05,-.06,1.]],np.float32)
        native=normalize(values,stats)
        np.testing.assert_array_equal(native[:,6],[1.,0.])
        np.testing.assert_allclose(denormalize(native,stats),values,atol=2e-8)

    def test_camera_cv_axes_and_pixel_centers(self):
        V=np.eye(4);P=np.eye(4);P[0,0]=P[1,1]=2.
        K,cv=geometry(V.T.ravel(),P.T.ravel())
        np.testing.assert_array_equal(K,[[384,0,191.5],[0,384,191.5],[0,0,1]])
        np.testing.assert_array_equal(cv,np.diag([1,-1,-1,1]))
        point=np.array([0.,0.,-1.,1.]);self.assertEqual((cv@point)[2],1.)

    def test_cleanup_cannot_touch_system_or_selected(self):
        from pathlib import Path
        for p in ('/etc/passwd','/home/uet/.cache/uv/archive-v0/x',
            str(ROOT/'eval/ICCAS_demo/x.mp4'),str(ROOT/'VLA/checkpoints/xvla-base/model.safetensors'),
            str(ROOT/'VLA/smolvla_scene67/x/checkpoints/010000/pretrained_model/model.safetensors'),
            str(ROOT/'VLA/smolvla/old/checkpoints/x/trajectory.npz')):
            self.assertIsNone(reason(Path(p)))
        self.assertIsNotNone(reason(ROOT/'VLA/smolvla/old/checkpoints/x/training_state/optimizer_state.safetensors'))


if __name__=='__main__':unittest.main()
