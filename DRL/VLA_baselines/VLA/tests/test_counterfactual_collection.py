import unittest

import numpy as np

from dataclasses import replace
from VLA.audit_counterfactual_collection import alternate_target, alternate_destination, physical_scene_spec, trajectory_action_difference, recovery_visibility_names
from VLA.common.tasks import SceneObjectSpec, TaskSpec


class CounterfactualTrajectoryTest(unittest.TestCase):
    def test_recovery_checks_unselected_physical_goals_too(self):
        from types import SimpleNamespace
        task = SimpleNamespace(config_pick_items=("red block", "blue block"),
            config_place_items=("green bowl", "middle", "blue block"))
        self.assertEqual(recovery_visibility_names(task, {"red block": 1, "blue block": 2, "green bowl": 3}),
                         ("red block", "blue block", "green bowl"))
        with self.assertRaises(ValueError):
            recovery_visibility_names(task, {"red block": 1, "blue block": 2})

    def test_same_first_clipped_move_does_not_mean_same_trajectory(self):
        first = np.zeros((3, 7), dtype=np.float32)
        second = first.copy()
        first[0, :3] = second[0, :3] = .045
        second[1, 0] = -.045
        result = trajectory_action_difference(first, second)
        self.assertGreater(result["common_prefix_max_action_difference"], .001)

    def test_identical_action_sequences_are_identified(self):
        action = np.zeros((3, 7), dtype=np.float32)
        result = trajectory_action_difference(action, action)
        self.assertEqual(result["common_prefix_max_action_difference"], 0)
        self.assertFalse(result["different_trajectory_lengths"])

    def test_retarget_preserves_physical_scene_and_fixed_flags(self):
        task = TaskSpec(task_id=0, name="touch", prompt="Touch the red block", pick_name="red block",
            pick_position=(.1, -.5, .03), scene_objects=(
                SceneObjectSpec("blue block", (-.1, -.5, .03), False, "pick"),
                SceneObjectSpec("green bowl", (0, -.3, 0), True, "place")))
        other = alternate_target(task)
        self.assertEqual(physical_scene_spec(task), physical_scene_spec(other))
        self.assertEqual(other.prompt, "Touch the blue block")
        self.assertNotEqual(physical_scene_spec(task), physical_scene_spec(replace(other, pick_position=(.2, -.5, .03))))

    def test_destination_intervention_preserves_existing_bodies(self):
        task = TaskSpec(task_id=3, name="pick_place", prompt="Put the red block onto the blue bowl",
            pick_name="red block", pick_position=(.1, -.5, .03), place_name="blue bowl", place_position=(.2, -.3, 0),
            config_place_items=("blue bowl", "gray bowl"), config_selected_place="blue bowl",
            scene_objects=(SceneObjectSpec("gray bowl", (-.2, -.7, 0), True, "place"),))
        other = alternate_destination(task)
        self.assertEqual(other.place_name, "gray bowl")
        self.assertEqual(other.prompt, "Put the red block onto the gray bowl")
        self.assertEqual(physical_scene_spec(task), physical_scene_spec(other))


if __name__ == "__main__":
    unittest.main()
