from types import SimpleNamespace
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from VLA.scene64_r33_client import Scene64R33Policy


class Tests(unittest.TestCase):
    def test_measured_open_and_explicit_previous_frames(self):
        p=Scene64R33Policy.__new__(Scene64R33Policy);p.env=object();p.index=0;p.seed=1;p.url='local';p._previous=None
        sent=[]
        def post(url,json,timeout):
            sent.append(json)
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'action_source':'native_neural_model',
                'prompt_sha256':hashlib.sha256(json['prompt'].encode()).hexdigest(),'actions':[[0.]*7],
                'action_kind':'canonical_delta_ee_7d'})
        p.session=SimpleNamespace(post=post)
        obs=SimpleNamespace(prompt='Touch the yellow block',joint_positions=np.arange(6,dtype=np.float32),
            gripper_open=np.float32(.6095944),front_rgb='first_front',wrist_rgb='first_wrist')
        with patch('VLA.scene64_r33_client.model_input',return_value={'scene64':np.zeros(64),'prompt':'augmented'}),patch('VLA.scene64_r33_client.png',side_effect=lambda value:value):
            p.predict(obs);obs.front_rgb='second_front';obs.wrist_rgb='second_wrist';p.predict(obs)
        self.assertEqual(sent[0]['native_proprio'][6],float(obs.gripper_open))
        self.assertEqual(sent[1]['previous_front_rgb_png'],'first_front')
        self.assertEqual(sent[1]['previous_wrist_rgb_png'],'first_wrist')
        self.assertEqual(sent[1]['front_rgb_png'],'second_front')
        self.assertEqual(len(sent[0]['native_proprio']),16)


if __name__=='__main__':unittest.main()
