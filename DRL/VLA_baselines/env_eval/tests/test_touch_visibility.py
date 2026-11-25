import unittest
import tempfile
from pathlib import Path
from env_eval.legacy_gripper_environment import LegacyGripperEnvironment
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.oracle_demo import scenes


class TouchVisibilityTests(unittest.TestCase):
    def test_gentle_contact_is_visible_for_multiple_video_frames(self):
        for robot in ('uf850','ur5e'):
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                with LegacyGripperEnvironment(robot,root/'assets') as env:
                    scene=scenes()[2];env.reset(scene.task())
                    rollout=CinematicRollout(env,scene,root)
                    result=rollout.run()
                    self.assertTrue(result['overall_pass'])
                    visible=[s for s in rollout.samples if s['phase']=='touch' and s['contact_force_n']>.02]
                    self.assertGreaterEqual(len(visible),5)
                    self.assertLess(result['max_touch_force_n'],5.)


if __name__=='__main__':unittest.main()
