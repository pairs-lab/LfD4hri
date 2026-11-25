"""UF850 mount fix based on the user's working Robotiq2F85 settings.

Same native gripper meshes, articulation, public API and finger controller;
only the flange mount/dynamics differ from the original UR5e class.
"""
import pybullet
from utils.Robotiq2F85 import Robotiq2F85


class Robotiq2F85UF850(Robotiq2F85):
    MOUNT_OFFSET = MOUNT_PARENT_POSITION = (0., 0., 0.)
    MOUNT_CHILD_LINK = -1
    MOUNT_CHILD_POSITION = (0., 0., 0.)

    def _configure_mount_dynamics(self):
        # One root constraint. The broad mounting face is at local z=0;
        # the -4 mm mesh extremity is a locating lip, not the seating plane.
        pybullet.changeConstraint(self.mount_constraint, maxForce=50000., erp=.9)
        pybullet.changeDynamics(self.robot, self.tool, mass=.1,
                                localInertiaDiagonal=[.1, .1, .1])
        pybullet.changeDynamics(self.body, -1, mass=.2,
                                localInertiaDiagonal=[.0002, .0002, .00018],
                                linearDamping=0., angularDamping=0.)
