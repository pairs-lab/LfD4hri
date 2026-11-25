import unittest
import numpy as np
from VLA.audit_sampling_phase_mass import phase_masks, weighted_metrics


def arrays():
    return {"canonical_actions": np.zeros((3, 7)), "gripper_closure": np.zeros(3),
        "next_grasped_object": np.zeros(3, bool), "next_verified_grasp": np.zeros(3, bool)}


class SamplingPhaseMassTest(unittest.TestCase):
    def test_acquisition_not_counted_as_empty_close_lift(self):
        data = arrays()
        data["canonical_actions"][0, [2, 6]] = [.01, 1]
        data["next_grasped_object"][0] = True
        masks = phase_masks(data)
        self.assertTrue(masks["close_acquires_correct_object"][0])
        self.assertFalse(masks["target_unheld_close_lift_over_5mm"][0])

    def test_reopen_uses_actual_feedback(self):
        data = arrays()
        data["gripper_closure"][0] = .99
        data["canonical_actions"][0, 6] = -1
        metrics = weighted_metrics(phase_masks(data), [.1, .3, .6])
        self.assertAlmostEqual(metrics["open_while_closed_target_unheld"]["probability_mass"], .1)
        self.assertEqual(metrics["open_while_closed_target_unheld"]["frames"], 1)

    def test_measured_hold_precedes_drop_and_retreat(self):
        data = arrays()
        data["next_grasped_object"][0] = True
        data["canonical_actions"][1, 2] = -.01
        masks = phase_masks(data)
        self.assertTrue(masks["correct_held_to_unheld"][1])
        self.assertFalse(masks["target_unheld_vertical_descent_over_3mm"][1])

    def test_nonfinite_or_unaligned_weights_rejected(self):
        data = arrays()
        data["gripper_closure"][0] = np.nan
        with self.assertRaises(ValueError):
            phase_masks(data)
        with self.assertRaises(ValueError):
            weighted_metrics({"reset": np.ones(3, bool)}, [.5, .5])


if __name__ == "__main__":
    unittest.main()
