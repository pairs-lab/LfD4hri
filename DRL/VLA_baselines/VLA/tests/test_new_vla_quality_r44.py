import copy
import unittest

import numpy as np
from PIL import Image

from VLA.audit_new_vla_quality_r44 import physical_signature


class NativeQualityAuditTest(unittest.TestCase):
    def task(self):
        return {'pick_name': 'red block', 'pick_position': [0., -.5, .03],
            'place_name': None, 'place_position': None,
            'scene_objects': [{'name': 'blue block', 'position': [.2, -.5, .03],
                               'fixed': False, 'source_role': 'pick'}]}

    def test_physical_signature_not_language_or_requested_identity(self):
        task = self.task()
        alternate = copy.deepcopy(task)
        alternate['pick_name'], alternate['pick_position'] = 'blue block', [.2, -.5, .03]
        alternate['scene_objects'] = [{'name': 'red block', 'position': [0., -.5, .03],
                                       'fixed': False, 'source_role': 'place'}]
        alternate['prompt'] = 'Different command for the same physical scene'
        self.assertEqual(physical_signature(task), physical_signature(alternate))

    def test_real_geometry_change_is_not_hidden_by_metadata(self):
        task = self.task()
        alternate = copy.deepcopy(task)
        alternate['pick_position'][0] += .001
        self.assertNotEqual(physical_signature(task), physical_signature(alternate))

    def test_named_empty_goal_does_not_invent_a_physical_object(self):
        task = self.task()
        alternate = {**task, 'place_name': 'middle', 'place_position': [0., -.5, 0.]}
        self.assertEqual(physical_signature(task), physical_signature(alternate))

    def test_actual_PIL_mask_lookup_matches_per_frame_resizer(self):
        rng = np.random.default_rng(44)
        mask = rng.integers(0, 2, (3, 384, 384)).astype(bool)
        indices = np.asarray(Image.fromarray(np.arange(384, dtype=np.int32)[None])
            .resize((224, 1), Image.Resampling.NEAREST))[0]
        reduced = mask[:, indices][:, :, indices]
        for frame in range(len(mask)):
            np.testing.assert_array_equal(reduced[frame], np.asarray(Image.fromarray(mask[frame])
                .resize((224, 224), Image.Resampling.NEAREST)))


if __name__ == '__main__':
    unittest.main()
