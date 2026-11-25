import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from env_eval.oracle_style import styled_approach,styled_carry,styled_lift

class StyledTrajectoryTests(unittest.TestCase):
    def test_lift_has_no_height_overshoot(self):
        curve=styled_lift([0,0,.025],[0,0,.175],[0,1,0,0])
        pts=np.array([curve.sample(t)[0] for t in np.linspace(0,curve.duration,501)])
        self.assertTrue(np.all(np.diff(pts[:,2])>=-1e-10))
        self.assertLessEqual(pts[:,2].max(),.175+1e-10)
        self.assertGreater(np.ptp(pts[:,0]),.003)

    def test_arc_length_timing_has_bell_shaped_speed(self):
        curve=styled_carry([-.12,-.5,.025],[.15,-.5,.045],[0,1,0,0],clearance=.13,duration=3.7)
        time=np.linspace(0,curve.duration,1001);pts=np.array([curve.sample(t)[0] for t in time])
        speed=np.linalg.norm(np.diff(pts,axis=0),axis=1)/np.diff(time)
        peak=np.argmax(speed)
        self.assertTrue(np.all(np.diff(speed[:peak-1])>=-1e-5))
        self.assertTrue(np.all(np.diff(speed[peak+1:])<=1e-5))
        self.assertLess(speed[0],.0001);self.assertLess(speed[-1],.0001)

    def test_wrist_is_continuous_and_changes_during_transport(self):
        curve=styled_carry([-.12,-.5,.025],[.15,-.5,.045],[0,1,0,0],clearance=.13,duration=3.7)
        quats=np.array([curve.sample(t)[1] for t in np.linspace(0,curve.duration,1001)])
        rotations=Rotation.from_quat(quats)
        steps=(rotations[:-1].inv()*rotations[1:]).magnitude()
        self.assertLess(steps.max(),.001)
        self.assertGreater((rotations[0].inv()*rotations[500]).magnitude(),.01)
        self.assertLess((rotations[0].inv()*rotations[-1]).magnitude(),1e-8)

    def test_approach_orientation_has_zero_endpoint_speed(self):
        q0=Rotation.from_euler('xyz',[.2,np.pi,.2]).as_quat();q1=[0,1,0,0]
        curve=styled_approach([0,-.3,.35],[-.12,-.5,.025],q0,q1,duration=3.5)
        epsilon=1e-4
        for t in [0.,curve.duration-epsilon]:
            a,b=curve.sample(t)[1],curve.sample(t+epsilon)[1]
            speed=(Rotation.from_quat(a).inv()*Rotation.from_quat(b)).magnitude()/epsilon
            self.assertLess(speed,1e-5)

    def test_large_turn_finishes_before_contact_without_terminal_overshoot(self):
        q0=Rotation.from_euler('xyz',[.8,.2,-1.4]).as_quat()
        q1=Rotation.from_euler('xyz',[0,np.pi,np.pi/2]).as_quat()
        curve=styled_approach([0,-.3,.35],[-.12,-.5,.025],q0,q1,duration=3.5)
        for clock in np.linspace(.85,1.,31):
            error=(Rotation.from_quat(q1).inv()*curve.orientation(clock)).magnitude()
            self.assertLess(error,1e-9)

if __name__=='__main__':unittest.main()
