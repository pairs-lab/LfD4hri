from types import SimpleNamespace
import unittest
import numpy as np

from VLA.recovery_teacher import RecoveryTeacher


class RecoveryTeacherTest(unittest.TestCase):
    def state(self, grasped):
        z = np.zeros(3)
        return {"grasped": grasped, "pick_position": z, "pick_center": z,
                "initial_pick_position": z, "grasp_center": z,
                "place_position": z, "goal_position": z}

    def policy(self, grasped):
        p = RecoveryTeacher(SimpleNamespace(debug_state=lambda: self.state(grasped)))
        p.reset(SimpleNamespace(name="pick"))
        return p

    def test_timeout_without_grasp_reopens_not_lifts(self):
        p = self.policy(False)
        p.phase, p.phase_steps = "close", 15
        action = p.predict(SimpleNamespace(step=7))
        self.assertEqual(p.phase, "approach")
        self.assertLess(action[6], 0)
        self.assertEqual(p.recovery_events[0]["reason"], "close_without_grasp")

    def test_lost_grasp_reopens_at_each_carry_phase(self):
        for phase in ("lift", "transport", "lower", "hold"):
            p = self.policy(False)
            p.phase = phase
            self.assertLess(p.predict(SimpleNamespace(step=12))[6], 0)
            self.assertEqual(p.phase, "approach")

    def test_real_grasp_can_advance_from_close(self):
        p = self.policy(True)
        p.phase, p.phase_steps = "close", 15
        self.assertGreater(p.predict(SimpleNamespace(step=7))[6], 0)
        self.assertEqual(p.phase, "lift")
        self.assertFalse(p.recovery_events)
