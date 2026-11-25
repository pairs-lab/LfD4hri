import copy
import unittest
import numpy as np
from VLA.collect_paired_grounding_candidate import pair_errors, pair_decision_frame_errors, select_grounding_rows, select_target_coverage_rows


class PairedGroundingCandidateTest(unittest.TestCase):
    def test_post_hover_probe_requires_same_sensors_common_prefix_and_informative_xy(self):
        first = {key: np.zeros((5, 3)) for key in ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure", "ee_position", "ee_rotation_6d")}
        first["canonical_actions"] = np.zeros((5, 7))
        second = copy.deepcopy(first)
        second["canonical_actions"][2:, 0] = .02
        self.assertTrue(pair_decision_frame_errors(first, second, 2)["all_actual_sensors_identical"])
        second["front_rgb"][2, 0] = 1
        with self.assertRaisesRegex(ValueError, "changed actual sensors"):
            pair_decision_frame_errors(first, second, 2)
        second["front_rgb"][2, 0] = 0
        second["canonical_actions"][0, 0] = .01
        with self.assertRaisesRegex(ValueError, "common physical action prefix"):
            pair_decision_frame_errors(first, second, 2)
        second["canonical_actions"][:] = 0
        with self.assertRaisesRegex(ValueError, "do not discriminate"):
            pair_decision_frame_errors(first, second, 2)
        second["canonical_actions"] = np.zeros((1, 7))
        with self.assertRaisesRegex(ValueError, "outside a paired trajectory"):
            pair_decision_frame_errors(first, second, 2)

    def rows(self):
        return [{"file": f"{task}_{scene}", "task": {"name": task, "pick_name": "red apple",
                 "pick_position": [scene, number, 0], "place_name": None, "place_position": None,
                 "scene_objects": []}} for number, task in enumerate(("touch", "pick", "move", "pick_place"))
                for scene in range(3)]

    def test_selection_uses_only_ordinary_distinct_layout_metadata(self):
        rows = self.rows()
        selected = select_grounding_rows(rows)
        altered = copy.deepcopy(rows)
        for row in altered:
            row.update(success=False, neural_success=False)
        self.assertEqual(selected, [{k:v for k,v in row.items() if k not in {"success", "neural_success"}}
                                    for row in select_grounding_rows(altered)])
        rows[0]["paired_scene_source"] = "old"
        self.assertNotIn(rows[0], select_grounding_rows(rows))

    def test_count_bounds_and_insufficient_unique_scenes_fail_closed(self):
        for count in (0, 5, True):
            with self.assertRaises(ValueError):
                select_grounding_rows(self.rows(), count)
        rows = self.rows()
        for row in rows:
            row["task"]["pick_position"][0] = 0
        with self.assertRaises(ValueError):
            select_grounding_rows(rows)

    def test_destination_intervention_selects_only_carry_primitives(self):
        selected = select_grounding_rows(self.rows(), 2, ("move", "pick_place"))
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["task"]["name"] for row in selected}, {"move", "pick_place"})
        with self.assertRaises(ValueError):
            select_grounding_rows(self.rows(), 2, ("unknown",))

    def test_object_balanced_selection_is_metadata_only_and_retains_each_name(self):
        rows = self.rows()
        for row in rows:
            if row["file"].endswith("_2"):
                row["task"]["pick_name"] = "green pear"
        selected = select_target_coverage_rows(rows)
        self.assertEqual(len(selected), 8)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["neural_success"] = False
        self.assertEqual([r["file"] for r in selected], [r["file"] for r in select_target_coverage_rows(changed)])
        for task in ("touch", "pick", "move", "pick_place"):
            self.assertEqual({r["task"]["pick_name"] for r in selected if r["task"]["name"] == task}, {"red apple", "green pear"})

    def test_balanced_selection_does_not_silently_skip_unavailable_target(self):
        rows = self.rows()
        rows[0]["task"]["pick_name"] = "green pear"
        rows[0]["recovery_start_actions_not_training_labels"] = [[0] * 7]
        with self.assertRaisesRegex(ValueError, "Missing ordinary"):
            select_target_coverage_rows(rows)

    def arrays(self):
        return {"prompt": np.asarray("Touch the red apple"), "joint_positions": np.zeros((2, 6)),
                "gripper_closure": np.zeros(2), "front_rgb": np.zeros((2, 4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((2, 4, 4, 3), dtype=np.uint8), "canonical_actions": np.zeros((2, 7))}

    def test_matching_pixels_different_text_and_controls(self):
        first, second = self.arrays(), self.arrays()
        second["prompt"] = np.asarray("Touch the green pear")
        second["canonical_actions"][0, 0] = .01
        self.assertEqual(pair_errors(first, second)["front_initial_max_pixel_error"], 0)
        for key in ("prompt", "canonical_actions", "front_rgb", "joint_positions", "gripper_closure"):
            altered = copy.deepcopy(second)
            if key in {"prompt", "canonical_actions"}:
                altered[key] = first[key].copy()
            else:
                altered[key].flat[0] = 1
            with self.assertRaises(ValueError):
                pair_errors(first, altered)
