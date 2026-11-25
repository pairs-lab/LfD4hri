"""Regression for the real native current-frame singleton time axis."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from VLA.smolvla.scene67_dataset import Scene67Dataset
from VLA.scene64_dataset import nearest_joint_targets


class Tests(unittest.TestCase):
    def test_correction_preserves_native_time_pixels_padding_and_actions(self):
        d=object.__new__(Scene67Dataset);d.original_length=100;d.correction_ends=[3]
        d.parent_rows=[{}];d.states=np.zeros((103,67),np.float32)
        cameras=['observation.images.front','observation.images.wrist'];d.meta=SimpleNamespace(camera_keys=cameras)
        template={'action':torch.zeros(5,7),'action_is_pad':torch.zeros(5,dtype=torch.bool),
            'observation.state':torch.zeros(1,7),'observation.state_is_pad':torch.ones(1,dtype=torch.bool),
            'index':torch.tensor(0),'frame_index':torch.tensor(0),'episode_index':torch.tensor(0),
            'timestamp':torch.tensor(0.)}
        for k in cameras:template[k]=torch.zeros(1,3,4,4);template[k+'_is_pad']=torch.ones(1,dtype=torch.bool)
        raw={'front_rgb':np.full((3,4,4,3),23,np.uint8),'wrist_rgb':np.full((3,4,4,3),171,np.uint8),
            'canonical_actions':np.arange(21,dtype=np.float32).reshape(3,7)}
        with patch.object(Scene67Dataset,'_correction_payload',return_value=(raw,template)):
            out=d._correction(102)
        self.assertEqual(out[cameras[0]].shape,(1,3,4,4))
        torch.testing.assert_close(out[cameras[0]],torch.full((1,3,4,4),23/255),rtol=0,atol=0)
        torch.testing.assert_close(out['action'],torch.from_numpy(np.repeat(raw['canonical_actions'][2:3],5,axis=0)),rtol=0,atol=0)
        self.assertEqual(out['action_is_pad'].tolist(),[False,True,True,True,True])
        self.assertFalse(out['observation.state_is_pad'].any())
        for k in cameras:self.assertFalse(out[k+'_is_pad'].any())

    def test_historical_images_never_broadcast_current_correction(self):
        d=object.__new__(Scene67Dataset);d.original_length=0;d.correction_ends=[1];d.parent_rows=[]
        d.states=np.zeros((1,67),np.float32);d.meta=SimpleNamespace(camera_keys=['observation.images.front'])
        template={'action':torch.zeros(5,7),'observation.state':torch.zeros(1,7),
            'observation.images.front':torch.zeros(2,3,4,4)}
        raw={'canonical_actions':np.zeros((1,7),np.float32),'front_rgb':np.zeros((1,4,4,3),np.uint8)}
        with patch.object(Scene67Dataset,'_correction_payload',return_value=(raw,template)):
            with self.assertRaises(ValueError):d._correction(0)

    def test_periodic_joint_target_has_short_native_delta(self):
        current=np.full(6,np.pi-.01);future=np.full((5,6),-np.pi+.01)
        target=nearest_joint_targets(future,current)
        np.testing.assert_allclose(target-current,.02,atol=2e-7,rtol=0)
        np.testing.assert_allclose(np.sin(target),np.sin(future),atol=2e-7,rtol=0)


if __name__=='__main__':unittest.main()
