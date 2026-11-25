import unittest
import numpy as np
from VLA.smolvla.audit_target_grounding import grounding_metrics
from VLA.smolvla.balanced_sampling import frame_weights, grounding_initial_weights


class TargetGroundingTests(unittest.TestCase):
    def test_text_sensitivity_can_be_in_wrong_direction(self):
        a, b = np.zeros((5, 7)), np.zeros((5, 7))
        a[:, 0], b[:, 0] = .02, -.02
        wrong = grounding_metrics([b, a], [a, b])
        self.assertLess(wrong["difference_projection_onto_teacher"], 0)
        self.assertEqual(wrong["requested_teacher_preferred"], [False, False])
        correct = grounding_metrics([a, b], [a, b])
        self.assertAlmostEqual(correct["difference_projection_onto_teacher"], 1)
        self.assertEqual(correct["requested_teacher_preferred"], [True, True])

    def test_text_independent_average_is_not_correct_grounding(self):
        a, b = np.zeros((3, 7)), np.zeros((3, 7))
        a[:, 0], b[:, 0] = .03, -.03
        result = grounding_metrics([(a+b)/2]*2, [a, b])
        self.assertEqual(result["requested_teacher_preferred"], [False, False])
        self.assertEqual(result["difference_projection_onto_teacher"], 0)

    def test_shared_vertical_first_moves_marked_uninformative(self):
        a = np.zeros((5, 7)); a[:, 2] = -.03
        result = grounding_metrics([a, a], [a, a])
        self.assertFalse(result["informative_teacher_xy_separation"])
        self.assertIsNone(result["requested_teacher_preferred"])

    def rows(self):
        return [{"steps": 4, "task": {"name": task, "pick_name": str(v), "prompt": task+str(v)},
                 "variant": v, "grounding_text_intervention": "target",
                 "paired_collection_source": "collection", "canonical_source": "parent",
                 "paired_scene_source": task, "recovery_start_actions_not_training_labels": []}
                for task in ("touch", "pick", "move", "pick_place") for v in (0, 1)]

    def test_grounding_mixture_has_exact_declared_mass(self):
        rows = self.rows(); weights = frame_weights(rows, .3, True, .2)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertAlmostEqual(weights[::4].sum(), .44)
        for n in range(4):
            self.assertAlmostEqual(weights[n*8:(n+1)*8].sum(), .25)
        reference = frame_weights(rows, .3, True)
        np.testing.assert_array_equal(reference, frame_weights(rows, .3, True, 0))

    def test_incomplete_or_recovery_pairs_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError): grounding_initial_weights(rows[:-1])
        rows[0]["recovery_start_actions_not_training_labels"] = [[1]]
        with self.assertRaises(ValueError): grounding_initial_weights(rows)

    def test_repeated_alternate_target_does_not_bias_object_names(self):
        rows = []
        for task in ("touch", "pick", "move", "pick_place"):
            for scene, targets in enumerate((("rare", "common"), ("common", "extra"))):
                for variant, target in enumerate(targets):
                    rows.append({"steps": 4, "task": {"name": task, "pick_name": target, "prompt": task+target},
                        "variant": variant, "grounding_text_intervention": "target",
                        "paired_collection_source": "collection", "canonical_source": "parent",
                        "paired_scene_source": task+str(scene)})
        weights = frame_weights(rows, .3, True, .2)
        for task in ("touch", "pick", "move", "pick_place"):
            for target in ("rare", "common", "extra"):
                mass = sum(weights[number*4:number*4+4].sum() for number, row in enumerate(rows)
                           if row["task"]["name"]==task and row["task"]["pick_name"]==target)
                self.assertAlmostEqual(mass, 1/12)


if __name__ == "__main__":
    unittest.main()
