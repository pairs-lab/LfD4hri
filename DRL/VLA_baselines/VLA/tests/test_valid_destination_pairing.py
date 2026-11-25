import unittest

from VLA.audit_counterfactual_collection import alternate_destination, physical_scene_spec
from VLA.common.tasks import SceneObjectSpec, TaskSpec
from VLA.valid_destination_pairing import scene_errors, valid_destination_pair


class ValidDestinationPairingTest(unittest.TestCase):
    def task(self):
        return TaskSpec(task_id=3, name="pick_place", prompt="Put the yellow banana onto the red gelatin_box",
            pick_name="yellow banana", pick_position=(.10982297, -.74222159, 0),
            place_name="red gelatin_box", place_position=(-.18755696, -.70892846, 0),
            scene_profile="rl_random", config_pick_items=("yellow banana", "white egg", "red apple", "purple block"),
            config_place_items=("left side", "red gelatin_box"), config_selected_place="red gelatin_box",
            scene_objects=(SceneObjectSpec("white egg", (-.13977019, -.54573471, .03), False, "pick"),
                           SceneObjectSpec("red apple", (.11720116, -.46246234, 0), False, "pick"),
                           SceneObjectSpec("purple block", (-.07084099, -.26626559, .03), False, "pick")))

    def test_obstructed_legacy_goal_repaired_without_physical_changes(self):
        task = self.task()
        self.assertEqual(scene_errors(task), [])
        self.assertTrue(scene_errors(alternate_destination(task)))
        candidate, rule = valid_destination_pair(task)
        self.assertEqual(scene_errors(candidate), [])
        self.assertEqual(physical_scene_spec(candidate), physical_scene_spec(task))
        self.assertEqual(candidate.pick_name, task.pick_name)
        self.assertEqual(candidate.config_pick_items, task.config_pick_items)
        self.assertEqual(len(candidate.config_place_items), len(task.config_place_items))
        self.assertEqual(rule["rule"], "clear_virtual_region_replacement")
        self.assertNotEqual(candidate.prompt, task.prompt)
        self.assertNotEqual(candidate.place_name, "left side")

    def test_valid_configured_goal_retained(self):
        from dataclasses import replace
        task = replace(self.task(), config_place_items=("right side", "red gelatin_box"))
        candidate, rule = valid_destination_pair(task)
        self.assertEqual(candidate.place_name, "right side")
        self.assertEqual(candidate.config_place_items, task.config_place_items)
        self.assertEqual(rule["rule"], "valid_existing_configured_destination")

    def test_invalid_original_not_silently_repaired(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            valid_destination_pair(replace(self.task(), pick_position=(-.13977019, -.54573471, .03)))
