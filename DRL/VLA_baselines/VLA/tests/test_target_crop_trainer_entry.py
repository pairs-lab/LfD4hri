import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from VLA.smolvla import train_target_crop as recipe
from VLA.smolvla.target_crop import SCHEMA


class TrainerEntryTests(unittest.TestCase):
    def test_real_recipe_sampling_hook_is_valid_before_dataset_construction(self):
        from VLA.smolvla import train_scene67,train_fp32,balanced_sampling,scene67_dataset
        previous=(balanced_sampling.frame_weights,balanced_sampling.sampling_contract,scene67_dataset.Scene67Dataset)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);gate=root/'gate.json';gate.write_text(json.dumps({'passed':True,'schema':SCHEMA,'samples':128,'bound_files':{}}))
            rows=[]
            for task in ('touch','pick','move','pick_place'):
                row={'file':task+'.npz','steps':2,'task':{'name':task,'pick_name':'object'}};rows.append(row)
                np.savez(root/row['file'],canonical_actions=np.array([[.02,0,0,0,0,0,-1],[0,0,-.01,0,0,0,1]],np.float32))
            class FakeDataset:
                def __init__(self,*_a,**_k):
                    self.parent_rows=rows;self.raw64=np.zeros((16,64),np.float32);self.original_length=8
                    self.ends=[2,4,6,8];self.audit={'source':str(root)};self.corrections=rows
            observed=[]
            def fake_native_main():
                # This is the ACTUAL order inside train_fp32.main: describe
                # the recipe before the trainer creates its dataset.
                observed.append(balanced_sampling.sampling_contract(None,.1))
                data=scene67_dataset.Scene67Dataset(None,None,None)
                weights=balanced_sampling.frame_weights(rows,.1,target_balanced=True)
                self.assertEqual(len(weights),8);self.assertAlmostEqual(float(weights.sum()),1)
                self.assertTrue((root/'actual_target_crop_sampling.json').exists())
            def fake_scene67_main():train_fp32.main()
            env={'VLA_TARGET_CROP_GATE':str(gate),'VLA_SCENE67_UPDATES':'2','VLA_SMOLVLA_MASTER_AUDIT':str(root/'master.json')}
            try:
                with patch.dict(os.environ,env),patch.object(recipe,'TargetCropDataset',FakeDataset),\
                     patch.object(train_scene67,'main',fake_scene67_main),patch.object(train_fp32,'main',fake_native_main):
                    recipe.main()
                self.assertEqual(observed[0]['profile'],SCHEMA)
                self.assertTrue(observed[0]['third_view_current_GT_only'])
            finally:
                balanced_sampling.frame_weights,balanced_sampling.sampling_contract,scene67_dataset.Scene67Dataset=previous


if __name__=='__main__':unittest.main()
