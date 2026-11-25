from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from VLA.smolvla.target_crop import CAMERA,NATIVE_CAMERA


class FactoryTests(unittest.TestCase):
    def test_real_factory_resolves_three_views_before_heavy_model_loading(self):
        from lerobot.policies.factory import make_policy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        cfg=SmolVLAConfig(device='cpu',max_state_dim=67,pretrained_path=Path('/inspection_only_no_weights'))
        cfg.input_features={};cfg.output_features={};observed=[]
        features={'observation.state':{'dtype':'float32','shape':[67]},
            'action':{'dtype':'float32','shape':[7]}}
        for camera in ('observation.images.front','observation.images.wrist',CAMERA):
            features[camera]={'dtype':'image','shape':(384,384,3),'names':['height','width','channels']}
        meta=SimpleNamespace(features=features,stats={})
        class Reached(Exception):pass
        class Stub:
            @classmethod
            def from_pretrained(cls,**kwargs):
                actual=kwargs['config'];observed.append(actual)
                self.assertEqual(tuple(actual.input_features['observation.state'].shape),(67,))
                self.assertIn(NATIVE_CAMERA,actual.input_features)
                raise Reached()
        rename={'observation.images.front':'observation.images.camera1','observation.images.wrist':'observation.images.camera2',CAMERA:NATIVE_CAMERA}
        with patch('lerobot.policies.factory.get_policy_class',return_value=Stub),self.assertRaises(Reached):
            make_policy(cfg=cfg,ds_meta=meta,rename_map=rename)
        self.assertEqual(len(observed),1)


if __name__=='__main__':unittest.main()
