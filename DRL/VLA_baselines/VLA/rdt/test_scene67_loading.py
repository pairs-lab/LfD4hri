"""Tiny REAL upstream RDT save/reload and zero-gain compatibility; CPU only."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from VLA.rdt import server
from VLA.rdt.scene67_native import load,install_gains
from models.rdt_runner import RDTRunner


class TestLoading(unittest.TestCase):
    def test_exact_base_and_new_gain_checkpoint(self):
        config={'action_dim':128,'pred_horizon':5,'rdt':{'hidden_size':64,'depth':1,'num_heads':4},
            'lang_adaptor':'linear','img_adaptor':'linear','state_adaptor':'linear','lang_token_dim':16,
            'img_token_dim':16,'state_token_dim':128,'max_lang_cond_len':4,'img_cond_len':6,
            'noise_scheduler':{'num_train_timesteps':20,'num_inference_timesteps':2,
                'beta_schedule':'squaredcos_cap_v2','prediction_type':'sample','clip_sample':False}}
        native=RDTRunner(action_dim=128,pred_horizon=5,config=config,lang_token_dim=16,img_token_dim=16,
            state_token_dim=128,max_lang_cond_len=4,img_cond_len=6,dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text(json.dumps(config))
            torch.save(native.state_dict(),root/'pytorch_model.bin');first,_=load(root,5)
            for key,value in native.state_dict().items():torch.testing.assert_close(first.state_dict()[key],value,rtol=0,atol=0)
            self.assertTrue(torch.all(first.scene67_gain==0))
            with torch.no_grad():first.scene67_gain.copy_(torch.linspace(-.02,.03,49))
            torch.save(first.state_dict(),root/'pytorch_model.bin');second,_=load(root,5)
            for key,value in first.state_dict().items():torch.testing.assert_close(second.state_dict()[key],value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
