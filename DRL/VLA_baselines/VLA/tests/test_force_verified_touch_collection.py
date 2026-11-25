from types import SimpleNamespace
import unittest
from unittest.mock import patch
from VLA.collect_touch_replay_correction import ForceVerifiedTouchCollection


class ForceVerifiedTouchCollectionTest(unittest.TestCase):
    def wrapper(self, *, safety=None, step=4, force=0.):
        env = SimpleNamespace(gripper=SimpleNamespace(body=1), pick_id=2,
            step_count=step, task=SimpleNamespace(max_steps=80), _last_success=True)
        info = {"success":True, "raw_success":True, "strict_raw_success":True, "safety_violation":safety}
        env.step = lambda action: (None, 1., True, info)
        return ForceVerifiedTouchCollection(env, None, None), info

    @patch("VLA.collect_touch_replay_correction.p.getContactPoints", return_value=[tuple([0]*10)])
    def test_cached_contact_keeps_collecting_without_changing_reward_or_env(self, contacts):
        wrapper, original_info = self.wrapper()
        _, reward, done, info = wrapper.step(None)
        self.assertFalse(done)
        self.assertFalse(info["success"])
        self.assertTrue(info["raw_success"])
        self.assertTrue(original_info["success"])
        self.assertTrue(wrapper.env._last_success)
        self.assertFalse(wrapper._last_success)
        self.assertEqual(reward, 1.)

    @patch("VLA.collect_touch_replay_correction.p.getContactPoints", return_value=[])
    def test_never_overrides_safety_or_timeout(self, contacts):
        for kwargs in ({"safety":"table_penetration"}, {"step":80}):
            wrapper, _ = self.wrapper(**kwargs)
            self.assertTrue(wrapper.step(None)[2])
            self.assertFalse(wrapper.step(None)[3]["success"])

    @patch("VLA.collect_touch_replay_correction.p.getContactPoints", return_value=[tuple([0]*9+[.05])])
    def test_positive_force_preserves_original_success(self, contacts):
        wrapper, _ = self.wrapper()
        self.assertFalse(wrapper.step(None)[2])
        _, _, done, info = wrapper.step(None)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertTrue(wrapper._last_success)

    @patch("VLA.collect_touch_replay_correction.p.getContactPoints", return_value=[tuple([0]*9+[109.])])
    def test_high_force_is_not_an_accepted_demo(self, contacts):
        wrapper, _ = self.wrapper()
        _, _, done, info = wrapper.step(None)
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertEqual(info["expert_quality_failure"], "contact_force_above_predeclared_5N")
        self.assertTrue(info["raw_success"])
