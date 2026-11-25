import unittest
import numpy as np
from VLA.obstacle_aware_touch_teacher import box_xy_clearance, translated_vertical_sweep_clearance, touch_open_wait_action
from VLA.audit_preserved_touch_failure import preserved_reset_metadata


class ObstacleAwareTouchTest(unittest.TestCase):
    def test_recovery_export_retains_canonical_reset_and_physical_startup(self):
        item = {"variant": 0, "canonical_source": "/tmp/immutable-source", "paired_scene_source": "normal.npz",
                "paired_reset_strategy": "canonical_spawn_order_then_teacher_retarget",
                "recovery_start_actions_not_training_labels": [[0, 0, 0, 0, 0, 0, 1]],
                "success": False, "safety_violation": "old failure"}
        result = preserved_reset_metadata(item)
        self.assertNotIn("success", result)
        self.assertNotIn("safety_violation", result)
        self.assertEqual(result["paired_scene_source"], "normal.npz")
        self.assertEqual(result["variant"], 0)
        self.assertEqual(result["recovery_start_actions_not_training_labels"], item["recovery_start_actions_not_training_labels"])
        result["recovery_start_actions_not_training_labels"][0][6] = -1
        self.assertEqual(item["recovery_start_actions_not_training_labels"][0][6], 1)

    def test_closed_gripper_cannot_cache_open_pad_geometry(self):
        for closure in (.97348, .76648, .18048):
            np.testing.assert_array_equal(touch_open_wait_action(closure), [0, 0, 0, 0, 0, 0, -1])
        self.assertIsNone(touch_open_wait_action(.0179))
        with self.assertRaises(ValueError): touch_open_wait_action(float("nan"))

    def test_unused_finger_is_part_of_clearance(self):
        fingers = [([-.06, -.01, 0], [-.04, .01, .04]), ([.04, -.01, 0], [.06, .01, .04])]
        obstacle = [([-.06, -.02, 0], [-.02, .02, .06])]
        bad = translated_vertical_sweep_clearance(fingers, obstacle, [0, 0, 0], .2)
        good = translated_vertical_sweep_clearance(fingers, obstacle, [.1, 0, 0], .2)
        self.assertLess(bad, 0)
        self.assertGreater(good, .004)

    def test_sweep_accounts_for_upper_neighbour_but_ignores_high_palm(self):
        obstacle = [([0, 0, .1], [.1, .1, .15])]
        self.assertLess(translated_vertical_sweep_clearance([([0, 0, 0], [.1, .1, .03])],
                                                          obstacle, [0, 0, 0], .2), 0)
        self.assertEqual(translated_vertical_sweep_clearance([([0, 0, .2], [.1, .1, .3])],
                                                            obstacle, [0, 0, 0], .2), np.inf)

    def test_invalid_geometry_fails_closed(self):
        with self.assertRaises(ValueError):
            box_xy_clearance([0, 0, 0], [-1, 1, 1], [0, 0, 0], [1, 1, 1])
        with self.assertRaises(ValueError):
            translated_vertical_sweep_clearance([], [], [np.nan, 0, 0], .2)


if __name__ == "__main__":
    unittest.main()
