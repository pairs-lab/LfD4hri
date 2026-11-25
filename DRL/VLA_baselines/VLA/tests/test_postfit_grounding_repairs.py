import unittest
from types import SimpleNamespace
import numpy as np

from VLA.audit_saved_vla_videos import frame_plan
from VLA.audit_smolvla_closedloop_divergence import trajectory_distance
from VLA.staged_acquisition_teacher import horizontal_first_action
from VLA.audit_staged_teacher_pilot import valid_initial_staged_action
from VLA.audit_staged_localization_frames import localization_frames


class PostfitGroundingRepairsTests(unittest.TestCase):
    def test_post_hover_frame_selection_ignores_initial_vertical_motion(self):
        actions = np.zeros((8, 7))
        actions[:3, 2] = -.02
        actions[3:5, 0] = .03
        actions[5:, 2] = -.015
        self.assertEqual(localization_frames(actions),
                         {"reset": 0, "first_xy_after_hover": 3, "first_down_after_xy": 5})
        actions[:, :2] = 0
        with self.assertRaises(ValueError): localization_frames(actions)

    def test_initial_lowering_is_vertical_bounded_and_open_not_a_target_guess(self):
        self.assertTrue(valid_initial_staged_action([0, 0, -.02, 0, 0, 0, -1]))
        self.assertFalse(valid_initial_staged_action([.02, 0, -.02, 0, 0, 0, -1]))
        self.assertFalse(valid_initial_staged_action([0, 0, -.05, 0, 0, 0, -1]))
        self.assertFalse(valid_initial_staged_action([0, 0, -.02, 0, 0, 0, 1]))

    def test_horizontal_travel_never_descends_before_alignment(self):
        action, ready = horizontal_first_action([0, 0, .5], [.2, .1, .2], 0, .5, False)
        self.assertFalse(ready)
        self.assertEqual(action[2], 0)
        self.assertLessEqual(abs(action[0]), .03000001)
        self.assertEqual(action[-1], -1)

    def test_aligned_descent_is_bounded_and_new_xy_error_pauses_descent(self):
        action, ready = horizontal_first_action([0, 0, .5], [.005, .002, .2], 0, .5, False)
        self.assertTrue(ready)
        self.assertAlmostEqual(action[2], -.015)
        action, ready = horizontal_first_action([0, 0, .5], [.02, .002, .2], 0, .5, True)
        self.assertEqual(action[2], 0)

    def test_yaw_misalignment_also_blocks_descent(self):
        action, ready = horizontal_first_action([0, 0, .5], [0, 0, .2], .10, .5, False)
        self.assertFalse(ready)
        self.assertEqual(action[2], 0)
        self.assertAlmostEqual(action[5], .08)

    def test_actual_video_offsets_include_terminal_frame(self):
        trial = lambda n: {"file": str(n), "prompt": "text", "task": "pick", "success": False,
                           "steps": n, "trace": [None]*n}
        count, rows = frame_plan([trial(3), trial(8)])
        self.assertEqual(count, 13)
        self.assertEqual(rows[1]["video_frame_offset"], 4)
        self.assertEqual(rows[1]["absolute_frames"][-1], 12)
        bad = trial(3); bad["trace"].pop()
        with self.assertRaises(ValueError): frame_plan([bad])

    def test_pose_distance_uses_same_episode_and_wrapped_joints(self):
        observation = SimpleNamespace(ee_position=np.array([0, 0, .3]), joint_positions=np.zeros(6))
        arrays = {"ee_position": np.array([[0, 0, .5], [.01, 0, .3]]),
                  "joint_positions": np.full((2, 6), 2*np.pi), "canonical_actions": np.zeros((2, 7))}
        result = trajectory_distance(observation, arrays)
        self.assertEqual(result["nearest_expert_frame"], 1)
        self.assertAlmostEqual(result["nearest_expert_tcp_distance_m"], .01)
        self.assertLess(result["nearest_expert_joint_wrapped_rms_rad"], 1e-10)


if __name__ == "__main__":
    unittest.main()
