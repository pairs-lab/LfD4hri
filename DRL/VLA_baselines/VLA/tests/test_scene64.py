from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from VLA.simstate_scene64 import FEATURE_NAMES, scene_state, state_stats
from VLA.collect_scene64_recovery40 import select_train_cases, OfflineRecoveryTeacher
from VLA.audit_orientation_support import tilt


class SceneTests(unittest.TestCase):
    def make(self, attached=None, reverse=False, neighbors=2):
        first = np.zeros(18, np.float32); first[7:10] = [0, 0, .1]
        objects = {"target": 1, **{f"neighbor{i}": i+2 for i in range(neighbors)}}
        if reverse: objects = dict(reversed(list(objects.items())))
        grip = SimpleNamespace(body=30, PAD_LINKS=(4, 9), grasped_object=attached, width=lambda: .08)
        env = SimpleNamespace(pick_id=1, gripper=grip, object_ids=objects)
        obs = SimpleNamespace(ee_rotation_6d=np.array([1, 0, 0, 1, 0, 0], np.float32))
        def bounds(body): return ([body * .1, -.02, 0], [body * .1+.04, .02, .06])
        def contact(link, force):
            c = [0.] * 10; c[3], c[9] = link, force; return tuple(c)
        with patch("VLA.simstate_scene64.feedback_state", return_value=first), \
             patch("VLA.simstate_scene64.p.getAABB", side_effect=bounds), \
             patch("VLA.simstate_scene64.p.getContactPoints", return_value=[contact(4, .11), contact(9, .09), contact(9, .09)]), \
             patch("VLA.simstate_scene64.p.getBasePositionAndOrientation", return_value=([0, 0, 0], [0, 0, 0, 1])), \
             patch("VLA.simstate_scene64.p.getBaseVelocity", return_value=([.01, 0, 0], [0, 0, 0])), \
             patch("VLA.simstate_scene64.p.stepSimulation") as step:
            result = scene_state(env, obs); step.assert_not_called()
        np.testing.assert_array_equal(result[:18], first)
        return result

    def test_shape_names_contacts_and_padding(self):
        values = self.make()
        self.assertEqual(len(FEATURE_NAMES), 64); self.assertEqual(values.shape, (64,))
        np.testing.assert_array_equal(values[28:30], [1, 0])
        self.assertAlmostEqual(values[39], .29, places=6)
        self.assertTrue(np.all(values[48:] == 0))

    def test_wrong_attached_not_confused_with_no_attachment(self):
        np.testing.assert_array_equal(self.make()[30:32], [0, 0])
        np.testing.assert_array_equal(self.make(attached=1)[30:32], [0, 1])
        np.testing.assert_array_equal(self.make(attached=2)[30:32], [1, 1])

    def test_neighbor_order_is_stable_and_complete(self):
        np.testing.assert_array_equal(self.make(), self.make(reverse=True))
        with self.assertRaises(ValueError): self.make(neighbors=7)

    def test_stats_fail_closed(self):
        self.assertEqual(state_stats(np.stack([self.make(), self.make()]))["count"], [2])
        with self.assertRaises(ValueError): state_stats(np.zeros((3, 18)))
        with self.assertRaises(ValueError): state_stats(np.full((3, 64), np.nan))

    def test_actual_transposed_column_orientation_contract(self):
        identity = np.eye(3)
        angle = .4
        tilted = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        self.assertAlmostEqual(tilt(identity[:, :2].T.reshape(6))[0], 0.)
        self.assertAlmostEqual(tilt(tilted[:, :2].T.reshape(6))[0], np.degrees(angle))

    def test_predeclared_train_selection_does_not_read_neural_outcomes(self):
        rows = []
        for task in ("touch", "pick", "move", "pick_place"):
            for i in range(12):
                rows.append({"success": True, "file": f"{task}_{i}.npz",
                    "task": {"name": task, "pick_name": f"object{i}", "scene_key": f"{task}_{i}"}})
        with patch("VLA.collect_scene64_recovery40.physical_scene_key", side_effect=lambda t: t["scene_key"]):
            cases = select_train_cases(rows, {"touch_0"})
        self.assertEqual(len(cases), 40)
        self.assertEqual(len({c["row"]["task"]["scene_key"] for c in cases}), 40)
        self.assertNotIn("touch_0", {c["row"]["task"]["scene_key"] for c in cases})
        self.assertEqual({c["prefix_controls"] for c in cases}, {20, 40, 60})

    def test_wrong_held_release_precedes_offline_replanning(self):
        env = SimpleNamespace(pick_id=1, gripper=SimpleNamespace(grasped_object=2))
        teacher = OfflineRecoveryTeacher(env); task = SimpleNamespace(name="move")
        with patch("VLA.staged_acquisition_teacher.StagedAcquisitionTeacher.reset") as reset, \
             patch("VLA.staged_acquisition_teacher.StagedAcquisitionTeacher.predict", return_value=np.zeros(7)) as predict:
            teacher.reset(task); reset.assert_not_called()
            np.testing.assert_array_equal(teacher.predict(None), [0, 0, 0, 0, 0, 0, -1])
            reset.assert_not_called(); predict.assert_not_called()
            env.gripper.grasped_object = None
            teacher.predict(None); reset.assert_called_once_with(task); predict.assert_called_once()

    def test_correct_held_start_does_not_open_requested_object(self):
        env = SimpleNamespace(pick_id=1, gripper=SimpleNamespace(grasped_object=1, _bilateral=lambda _: True))
        teacher = OfflineRecoveryTeacher(env)
        with patch("VLA.staged_acquisition_teacher.StagedAcquisitionTeacher.reset"):
            teacher.reset(SimpleNamespace(name="move"))
        self.assertEqual(teacher.phase, "lift"); self.assertFalse(teacher.release_wrong)


if __name__ == "__main__": unittest.main()
