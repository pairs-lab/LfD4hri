import unittest
from config import PICK_TARGETS
from VLA.preflight_expanded_teacher_grid import declared_grid


class ExpandedTeacherGridTest(unittest.TestCase):
    def plan(self):
        return {"expert_teacher": "staged_open_obstacle_touch", "final_test_used": False,
                "seed": 2026101003, "episodes_per_task": 380, "closed_hover_recovery_every": 2}

    def test_all_original_targets_all_four_tasks_and_only_scheduled_recoveries(self):
        groups = declared_grid(self.plan())
        self.assertEqual(set(groups), {"touch", "pick", "move", "pick_place"})
        self.assertEqual(sum(map(len, groups.values())), 112)
        for jobs in groups.values():
            ordinary = [r for r in jobs if not r["closed_hover"]]
            self.assertEqual({r["task"]["pick_name"] for r in ordinary}, set(PICK_TARGETS))
            self.assertEqual(len(ordinary), 19)
            for row in jobs:
                if row["closed_hover"]:
                    filename = row["file"].replace("_closedhover", "")
                    self.assertEqual(row["task"], next(r for r in ordinary if r["file"] == filename)["task"])
        self.assertEqual(declared_grid(self.plan()), groups)

    def test_wrong_teacher_or_test_declaration_rejected(self):
        for change in ({"expert_teacher": "staged"}, {"final_test_used": True}, {"episodes_per_task": 3}):
            with self.assertRaises(ValueError): declared_grid({**self.plan(), **change})


if __name__ == "__main__":
    unittest.main()
