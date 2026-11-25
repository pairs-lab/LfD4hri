import unittest
from VLA.create_tiny_fit_view import select_pairs


class TinyFitSelectionTest(unittest.TestCase):
    def rows(self):
        return [{"task": {"name": t}, "paired_scene_source": t + str(scene), "variant": v,
                 "recovery_start_actions_not_training_labels": [1] if recovery else []}
                for t in ("touch", "pick", "move", "pick_place")
                for recovery in (True, False) for scene in range(2) for v in (0, 1)]

    def test_regular_counterfactual_pairs_only(self):
        rows = select_pairs(self.rows())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(not r["recovery_start_actions_not_training_labels"] for r in rows))
        self.assertTrue(all(r["paired_scene_source"].endswith("0") for r in rows))

    def test_missing_branch_rejected(self):
        with self.assertRaises(ValueError):
            select_pairs([r for r in self.rows() if r["variant"] == 0])


if __name__ == "__main__":
    unittest.main()
