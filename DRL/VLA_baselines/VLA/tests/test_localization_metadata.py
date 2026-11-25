import unittest

from VLA.audit_localization_metadata import spatial_cell, summarize


class LocalizationMetadataTests(unittest.TestCase):
    def row(self, pick="red apple", position=(-.1, -.5, 0.), others=None):
        return {"task": {"name": "pick", "pick_name": pick,
                         "pick_position": position, "place_name": None,
                         "scene_objects": others or []}}

    def test_boundary_cells(self):
        self.assertEqual(spatial_cell((-.25, -.75, 0.)), (0, 0))
        self.assertEqual(spatial_cell((.25, -.25, 0.)), (4, 4))

    def test_outside_or_nonfinite_rejected(self):
        for position in ((.3, -.5), (0., float("nan"))):
            with self.assertRaises(ValueError):
                spatial_cell(position)

    def test_recovery_copy_is_not_new_layout(self):
        row = self.row()
        recovery = dict(row, recovery_start_actions_not_training_labels=[[0] * 7], teacher_policy="recovery")
        report = summarize([row, recovery])
        self.assertEqual(report["physical_layouts"], 1)
        self.assertEqual(report["tasks"]["pick"]["episodes"], 2)
        self.assertEqual(report["tasks"]["pick"]["target_spawn_coverage"]["red apple"]["independent_layouts"], 1)
        self.assertFalse(report["data_sufficiency_established"])

    def test_same_scene_retargeting_and_confusion_diagnostics(self):
        apple = {"name": "red apple", "position": (-.1, -.5, 0.), "fixed": False}
        berry = {"name": "red strawberry", "position": (-.08, -.5, 0.), "fixed": False}
        rows = [self.row(others=[berry]), self.row("red strawberry", berry["position"], [apple])]
        report = summarize(rows)
        self.assertEqual(report["physical_layouts"], 1)
        self.assertEqual(report["layouts_with_multiple_requested_objects"], 1)
        self.assertEqual(report["tasks"]["pick"]["labelled_layouts_with_same_color_distractor"], 2)
        self.assertEqual(report["tasks"]["pick"]["labelled_layouts_with_neighbor_center_under_6cm"], 2)

    def test_stale_scene_identity_rejected(self):
        with self.assertRaises(ValueError):
            summarize([dict(self.row(), scene_group="changed")])


if __name__ == "__main__":
    unittest.main()
