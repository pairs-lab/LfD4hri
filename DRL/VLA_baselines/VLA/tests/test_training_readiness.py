import unittest

from VLA.audit_training_readiness import freeze_split, physical_scene_key


class TrainingReadinessTest(unittest.TestCase):
    def test_all_variants_recovery_and_tasks_stay_together(self):
        rows = [{"scene_group": str(scene), "task": task, "variant": variant, "recovery": recovery}
                for scene in range(10) for task in ("touch", "pick")
                for variant in (0, 1) for recovery in (0, 2)]
        split = freeze_split(rows)
        train = {r["scene_group"] for r in split["train"]}
        validation = {r["scene_group"] for r in split["validation"]}
        self.assertFalse(train & validation)
        self.assertEqual(len(validation), 2)
        self.assertEqual(split, freeze_split(list(rows)))

    def test_target_selection_does_not_change_physical_scene_key(self):
        task = {"pick_name": "red apple", "pick_position": [0, 0, 0],
                "place_name": "gray bowl", "place_position": [1, 0, 0],
                "scene_objects": [{"name": "blue block", "position": [0, 1, 0], "fixed": False}]}
        other = {**task, "pick_name": "blue block", "pick_position": [0, 1, 0],
                 "scene_objects": [{"name": "red apple", "position": [0, 0, 0], "fixed": False}]}
        self.assertEqual(physical_scene_key(task), physical_scene_key(other))
        other["place_position"] = [2, 0, 0]
        self.assertNotEqual(physical_scene_key(task), physical_scene_key(other))

    def test_single_scene_cannot_be_split(self):
        with self.assertRaises(ValueError):
            freeze_split([{"scene_group": "one", "task": "pick"}])

    def test_coverage_constraint_keeps_rare_labels_in_training(self):
        rows = [{"scene_group": str(i), "task": "move", "pick_name": "apple",
                 "place_name": "rare bowl" if i == 0 else "common bowl"} for i in range(10)]
        split = freeze_split(rows, preserve_training_vocabulary=True)
        self.assertEqual(len(split["validation"]), 2)
        self.assertIn("rare bowl", {r["place_name"] for r in split["train"]})
        self.assertEqual(split, freeze_split(rows, preserve_training_vocabulary=True))

    def test_impossible_coverage_constraint_fails_closed(self):
        rows = [{"scene_group": str(i), "task": "pick", "pick_name": str(i), "place_name": None}
                for i in range(4)]
        with self.assertRaises(ValueError):
            freeze_split(rows, preserve_training_vocabulary=True)

    def test_predeclared_minimum_repetitions_are_independent_groups(self):
        rows = [{"scene_group": str(i), "task": "move", "pick_name": "apple",
                 "place_name": "rare" if i < 3 else "common"} for i in range(10)]
        split = freeze_split(rows, preserve_training_vocabulary=True, minimum_train_repetitions=3)
        self.assertEqual(sum(r["place_name"] == "rare" for r in split["train"]), 3)
        copies = [r for r in rows for _ in range(4)]
        grouped = freeze_split(copies, preserve_training_vocabulary=True, minimum_train_repetitions=3)
        self.assertEqual(len({r["scene_group"] for r in grouped["train"] if r["place_name"] == "rare"}), 3)
        with self.assertRaises(ValueError):
            freeze_split(rows, minimum_train_repetitions=3)


if __name__ == "__main__":
    unittest.main()
