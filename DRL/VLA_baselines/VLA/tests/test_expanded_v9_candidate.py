import unittest
from VLA.collect_expanded_v9_candidate import FrozenRewardReset, frozen_thresholds, stratified_names
from config import PICK_TARGETS, PLACE_TARGETS


class ExpandedV9CandidateTest(unittest.TestCase):
    def rows(self):
        return [{"task": {"name": task}, "success_contract": {"touch_distance_threshold_m": .015,
                 "object_goal_threshold_m": .02}} for task in ("touch", "pick", "move", "pick_place")]

    def test_frozen_thresholds_require_every_task_and_consistent_numeric_values(self):
        self.assertEqual(frozen_thresholds(self.rows())["pick"], (.015, .02))
        with self.assertRaises(ValueError):
            frozen_thresholds(self.rows()[:-1])
        rows = self.rows()
        rows.append({"task": {"name": "touch"}, "success_contract": {
            "touch_distance_threshold_m": .025, "object_goal_threshold_m": .02}})
        with self.assertRaises(ValueError):
            frozen_thresholds(rows)

    def test_stratification_uses_existing_vocabularies_and_avoids_overlap(self):
        for task in ("touch", "pick", "move", "pick_place"):
            picks, places = set(), set()
            for number in range(600):
                pick, place = stratified_names(task, number)
                picks.add(pick); places.add(place)
                self.assertNotEqual(pick, place)
            self.assertEqual(picks, set(PICK_TARGETS))
            self.assertEqual(places, set(PLACE_TARGETS) if task in {"move", "pick_place"} else {None})

    def test_reset_pins_only_recorded_distances_after_normal_reset(self):
        class Env:
            def reset(self, task):
                self.dist_threshold, self.obj_threshold = .025, .03
                return "real observation"
        class Task:
            name = "touch"
        env = Env()
        wrapper = FrozenRewardReset(env, {"touch": (.015, .02)})
        self.assertEqual(wrapper.reset(Task()), "real observation")
        self.assertEqual((env.dist_threshold, env.obj_threshold), (.015, .02))
