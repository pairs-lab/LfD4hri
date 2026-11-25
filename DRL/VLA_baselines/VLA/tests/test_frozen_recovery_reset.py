from unittest.mock import patch
from types import SimpleNamespace
import unittest

from VLA.recollect_near_miss_frozen_contract import FrozenRecoveryReset


class FrozenRecoveryResetTests(unittest.TestCase):
    def test_pin_after_every_reset_not_before_overwritten_startup(self):
        env = SimpleNamespace(dist_threshold=.025, obj_threshold=.03, normalized_gripper_closure=lambda: .97)
        observation = SimpleNamespace(front_rgb=[[1]], wrist_rgb=[[2]], joint_positions=[3])
        reference = {"front_rgb": [[1]], "wrist_rgb": [[2]], "joint_positions": [3], "gripper_closure": .97}
        task = SimpleNamespace(name="pick", to_dict=lambda: {"name": "pick", "position": (0.0, -.5, 0.0)})
        proxy = FrozenRecoveryReset(env, "source", {"task": {"name": "pick", "position": [0.0, -.5, 0.0]}},
                                    {"pick": (.015, .02)}, reference)
        def resetting(*args):
            env.dist_threshold, env.obj_threshold = .025, .03
            return observation
        with patch("VLA.recollect_near_miss_frozen_contract.reset_source_episode", side_effect=resetting):
            for _ in range(2):
                self.assertIs(proxy.reset(task), observation)
                self.assertEqual((env.dist_threshold, env.obj_threshold), (.015, .02))
                self.assertFalse(any(proxy.reset_errors.values()))

    def test_changed_task_rejected_before_retargeting(self):
        proxy = FrozenRecoveryReset(None, "source", {"task": {"name": "pick"}}, {}, {})
        with self.assertRaisesRegex(ValueError, "changed original"):
            proxy.reset(SimpleNamespace(to_dict=lambda: {"name": "move"}))


if __name__ == "__main__":
    unittest.main()
