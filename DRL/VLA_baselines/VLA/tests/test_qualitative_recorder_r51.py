import unittest
from unittest.mock import patch

import numpy as np

from VLA.qualitative_recorder_r51 import (
    apply_render_state, array_hash, camera_stats, canonical_hash, select_layouts,
)
from VLA.qualitative_paired_pilot_r51 import oracle_benchmark_evidence


class ReadOnlyRecorderTests(unittest.TestCase):
    def test_retraction_is_not_retroactive_touch_failure(self):
        score, _ = oracle_benchmark_evidence({'benchmark_region_success_at_terminal': False}, 'touch',
            [{'physics_tick': 792, 'target_contact_force_N': 74.9}])
        self.assertTrue(score)

    def test_unlogged_first_hit_is_unknown_not_failure(self):
        score, _ = oracle_benchmark_evidence({'benchmark_region_success_at_terminal': False}, 'touch',
            [{'physics_tick': 784, 'target_contact_force_N': 2.7}])
        self.assertIsNone(score)

    def test_real_logged_false_is_preserved(self):
        score, _ = oracle_benchmark_evidence({'benchmark_region_success_at_any_matched_10Hz_sample': False}, 'pick', [])
        self.assertFalse(score)

    def test_first_hit_survives_later_retraction(self):
        score, _ = oracle_benchmark_evidence({'benchmark_region_success_at_any_matched_10Hz_sample': True,
            'benchmark_region_success_at_terminal': False}, 'touch', [])
        self.assertTrue(score)

    def test_hash_encodes_dtype_and_shape(self):
        self.assertNotEqual(array_hash(np.zeros(4, np.uint8)), array_hash(np.zeros(4, np.float32)))
        self.assertNotEqual(array_hash(np.zeros(4, np.uint8)), array_hash(np.zeros((2, 2), np.uint8)))

    def test_json_tuple_list_equivalence_is_not_a_scene_difference(self):
        self.assertEqual(canonical_hash({'q': (1., 2.)}), canonical_hash({'q': [1., 2.]}))

    def test_stats_are_true_target_and_link_masks(self):
        env = type('Env', (), {'pick_id': 2, 'gripper': type('Gripper', (), {'body': 3, 'PAD_LINKS': [4, 9]})()})()
        mask = np.full((20, 20), -1, np.int64)
        mask[5:10, 6:12] = 2
        mask[10:12, 4:7] = 3 + ((4+1) << 24)
        mask[10:12, 13:16] = 3 + ((9+1) << 24)
        stats = camera_stats(mask, env)
        self.assertEqual(stats['object_pixels'], 30)
        self.assertEqual(stats['pad_pixels'], [6, 6])
        self.assertEqual(stats['object_margin_px'], 5)

    def test_empty_target_is_reported_not_fabricated(self):
        env = type('Env', (), {'pick_id': 2, 'gripper': type('Gripper', (), {'body': 3, 'PAD_LINKS': [4, 9]})()})()
        stats = camera_stats(np.full((10, 10), -1, np.int64), env)
        self.assertEqual(stats['object_pixels'], 0)
        self.assertEqual(stats['object_margin_px'], -1)

    def test_layout_preserves_reference_size(self):
        frames = [{'wide': [{'projected_pick_box': [900, 600, 950, 650], 'projected_place_box': None}]*6}]
        layouts = select_layouts(frames)
        self.assertEqual(layouts['1']['rects'][0], ['handeye', 35, 37, 594, 334])

    def test_layout_checks_entire_trajectory_and_destination(self):
        clear = {'projected_pick_box': [900, 600, 950, 650], 'projected_place_box': None}
        left_goal = {'projected_pick_box': [900, 600, 950, 650], 'projected_place_box': [100, 100, 120, 120]}
        layouts = select_layouts([{'wide': [clear]*6}, {'wide': [left_goal]*6}])
        self.assertEqual(layouts['3']['side'], 'right')

    def test_cannot_hide_both_sides(self):
        a = {'projected_pick_box': [100, 100, 120, 120], 'projected_place_box': [1400, 100, 1420, 120]}
        with self.assertRaises(ValueError):
            select_layouts([{'wide': [a]*6}])

    def test_render_tolerance_is_separate_from_exact_physics_gate(self):
        state = [{'body': 1, 'pose': [[0., 0., 0.], [0., 0., 0., 1.]], 'joints': [.5]}]
        perturbed = [{'body': 1, 'pose': [[0., 0., 0.], [0., 0., 1e-16, 1.]], 'joints': [.5]}]
        with patch('VLA.qualitative_recorder_r51.p.resetBasePositionAndOrientation'), \
             patch('VLA.qualitative_recorder_r51.p.resetJointState'), \
             patch('VLA.qualitative_recorder_r51.render_state', return_value=perturbed):
            self.assertEqual(apply_render_state(state)['quaternion_component'], 1e-16)

    def test_render_rejects_shifted_object(self):
        state = [{'body': 1, 'pose': [[0., 0., 0.], [0., 0., 0., 1.]], 'joints': []}]
        moved = [{'body': 1, 'pose': [[.001, 0., 0.], [0., 0., 0., 1.]], 'joints': []}]
        with patch('VLA.qualitative_recorder_r51.p.resetBasePositionAndOrientation'), \
             patch('VLA.qualitative_recorder_r51.render_state', return_value=moved):
            with self.assertRaises(ValueError):
                apply_render_state(state)


if __name__ == '__main__':
    unittest.main()
