import unittest
import numpy as np
import torch

from VLA.rdt.simstate_inputs import append_causal_conditioning,EXTRA_SLOTS
from VLA.rdt.training_data import wrap_joints,VALID_INDICES


class RDTSimStateInputTests(unittest.TestCase):
    def fixture(self):
        state=np.arange(36,dtype=np.float32).reshape(2,18)/40
        native=torch.zeros(2,1,128)
        native[:,0,:6]=torch.from_numpy(wrap_joints(state[:,:6]))
        native[:,0,10]=torch.from_numpy(1-state[:,6])
        return native,state,{"mean":[0.]*18,"std":[1.]*18}

    def test_only_conditioning_slots_change_not_robot_state_or_output_layout(self):
        native,state,stats=self.fixture()
        out=append_causal_conditioning(native,state,stats)
        torch.testing.assert_close(out[:,0,list(VALID_INDICES)],native[:,0,list(VALID_INDICES)],rtol=0,atol=0)
        torch.testing.assert_close(out[:,0,list(EXTRA_SLOTS)],torch.from_numpy(state[:,7:]),rtol=0,atol=0)
        self.assertFalse(native[:,:,list(EXTRA_SLOTS)].any())

    def test_existing_fields_cannot_be_overwritten(self):
        native,state,stats=self.fixture();native[0,0,112]=1
        with self.assertRaises(ValueError):append_causal_conditioning(native,state,stats)

    def test_wrong_temporal_sensor_state_is_rejected(self):
        native,state,stats=self.fixture();state[0,0]+=.01
        with self.assertRaises(ValueError):append_causal_conditioning(native,state,stats)

    def test_open_close_sign_cannot_be_inverted(self):
        native,state,stats=self.fixture();native[0,0,10]=float(state[0,6])
        with self.assertRaises(ValueError):append_causal_conditioning(native,state,stats)

    def test_nonfinite_or_non18_features_are_rejected(self):
        native,state,stats=self.fixture()
        with self.assertRaises(ValueError):append_causal_conditioning(native,state[:,:17],stats)
        state[0,11]=np.nan
        with self.assertRaises(ValueError):append_causal_conditioning(native,state,stats)


if __name__=="__main__":unittest.main()
