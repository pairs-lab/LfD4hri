import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from VLA.smolvla.target_crop import target_crop,target_view


class CropTests(unittest.TestCase):
    def test_current_mask_selects_context_not_future_or_hidden_texture(self):
        rgb=np.zeros((384,384,3),np.uint8);rgb[50:70,100:120]=[250,50,20]
        mask=np.zeros((384,384),bool);mask[50:70,100:120]=True
        crop,receipt=target_crop(rgb,mask)
        self.assertEqual(crop.shape,rgb.shape);self.assertEqual(receipt['box_xyxy'],[90,40,130,80])
        self.assertEqual(receipt['visible_pixels'],400);self.assertGreater(np.count_nonzero(crop[:,:,0]),400)
        self.assertEqual(int(crop[192,192,0]),250)
    def test_empty_mask_keeps_full_current_view_and_does_not_invent_object(self):
        rgb=np.full((384,384,3),80,np.uint8);result,receipt=target_crop(rgb,np.zeros((384,384),bool))
        np.testing.assert_array_equal(result,rgb);self.assertTrue(receipt['fallback_current_full_front'])
    def test_edge_crop_is_bounded_and_shape_preserving(self):
        mask=np.zeros((384,384),bool);mask[380:,380:]=True
        result,receipt=target_crop(np.zeros((384,384,3),np.uint8),mask)
        self.assertEqual(receipt['box_xyxy'],[352,352,384,384]);self.assertEqual(result.shape,(384,384,3))
    def test_bad_mask_and_rgb_fail_closed(self):
        with self.assertRaises(ValueError):target_crop(np.zeros((384,384,3),np.uint8),np.zeros((384,384),np.uint8))

    def test_real_wrist_view_can_supply_visible_target_without_hidden_rerender(self):
        front=np.zeros((384,384,3),np.uint8);wrist=np.full_like(front,180)
        empty=np.zeros((384,384),bool);visible=empty.copy();visible[100:120,100:120]=True
        crop,receipt=target_view(front,wrist,empty,visible)
        self.assertEqual(receipt['source_camera'],'wrist');self.assertTrue(np.all(crop==180))
        crop,receipt=target_view(front,wrist,empty,empty)
        self.assertEqual(receipt['source_camera'],'front');np.testing.assert_array_equal(crop,front)

    def test_two_view_adapter_cannot_silently_omit_new_privileged_view(self):
        from VLA.smolvla.scene67_adapter import Scene67SmolVLA
        from VLA.smolvla.target_crop import SCHEMA
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'scene67_contract.json').write_text(json.dumps({'schema':SCHEMA}))
            (root/'config.json').write_text(json.dumps({'max_state_dim':67,'input_features':{'observation.state':{'shape':[67]}}}))
            with self.assertRaisesRegex(ValueError,'separately trained67'):Scene67SmolVLA(None,str(root))


if __name__=='__main__':unittest.main()
