import unittest
from VLA.common.training_selection import select_probe_rows
from VLA.common.spatial import FIXED_DESTINATION


class TrainingSelectionTest(unittest.TestCase):
    def test_named_regions_not_omitted_by_first_n_layouts(self):
        rows = [{"file": "first", "task": {"name": "move", "place_name": "blue bowl"}}]
        rows += [{"file": name, "task": {"name": "move", "place_name": name}} for name in FIXED_DESTINATION]
        selected = select_probe_rows(rows, 1, True)
        self.assertEqual(len(selected), 10)
        self.assertEqual({r["task"]["place_name"] for r in selected} - {"blue bowl"}, set(FIXED_DESTINATION))

    def test_missing_region_fails_closed(self):
        rows = [{"task": {"name": "pick_place", "place_name": "blue bowl"}}]
        with self.assertRaises(ValueError):
            select_probe_rows(rows, 1, True)

    def test_unmodified_selection_when_disabled(self):
        rows = [{"task": {"name": "touch"}}, {"task": {"name": "touch"}}]
        self.assertEqual(select_probe_rows(rows, 1), rows[:1])

    def test_all_destination_selection_includes_missing_physical_bowls(self):
        names = ("blue bowl", "brown bowl", "gray bowl", "middle")
        rows = [{"task": {"name": "move", "place_name": name}} for name in names]
        selected = select_probe_rows(rows, 1, destination_vocabulary=names + ("gray bowl",))
        self.assertEqual(len(selected), 4)
        self.assertEqual({r["task"]["place_name"] for r in selected}, set(names))


if __name__ == "__main__":
    unittest.main()
