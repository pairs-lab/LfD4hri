from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from protocol21.sensors import capture


class SensorTests(unittest.TestCase):
    def test_all_external_contacts_and_sides(self):
        from VLA.common.body_gripper import BodyContactRobotiq
        g=BodyContactRobotiq.__new__(BodyContactRobotiq)
        g.body=5;g.command=-1;g.width=lambda:.04
        g.pad_positions=lambda:(np.array([0,0,.2]),np.array([.04,0,.2]))
        env=SimpleNamespace(gripper=g,robot_id=1,robot_spec=SimpleNamespace(mount_link=9),client_id=0,step_count=2,control_hz=10)
        # No pick_id, goal, object_ids or attachment state is supplied.
        obs=SimpleNamespace(joint_positions=np.zeros(6),front_rgb=np.zeros((4,4,3),np.uint8),
                            wrist_rgb=np.zeros((4,4,3),np.uint8),prompt='touch red block')
        def c(body,link,force):return (0,5,body,link,0,None,None,None,0,force)
        contacts=[c(22,9,.06),c(23,9,.06),c(24,4,.05),c(1,4,100),c(5,9,100)]
        link=(None,None,None,None,[0,0,.3],[0,0,0,1])
        with patch('pybullet.getLinkState',return_value=link),patch('pybullet.getContactPoints',return_value=contacts) as query:
            f=capture(env,obs)
        np.testing.assert_allclose(f.raw_state21[18:],[1,0,.17])
        self.assertNotIn('bodyB',query.call_args.kwargs)
        np.testing.assert_allclose(f.raw_state21[7:10],[.02,0,.2])

    def test_legacy_centroid_gripper_rejected(self):
        with self.assertRaises(ValueError):capture(SimpleNamespace(gripper=object()),None)
