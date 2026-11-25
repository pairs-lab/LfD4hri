"""Versioned Robotiq body-contact controller; legacy assets/controller unchanged.

Same joints and collision meshes. Correctly rendered pads and surface-based TCP.
Assist is allowed only after sustained bilateral pad contact, never from an
object-specific target ID or a no-contact geometric envelope.
"""
import xml.etree.ElementTree as ET
import numpy as np
import pybullet as p
import trimesh

from VLA.common.robotiq import Robotiq2F85


class BodyContactRobotiq(Robotiq2F85):
    JOINTS = (1, 3, 5, 6, 8, 10)
    SIGNS = np.array([1, -1, 1, 1, -1, 1])

    def __init__(self, *args, **kwargs):
        use_gears = kwargs.pop("mimic_gear", False)
        self._latched = None
        self._contact_streak = {}
        self._lost_contact_ticks = 0
        super().__init__(*args, **kwargs)
        self.mimic_constraints = []
        if use_gears:
            for joint, sign in zip(self.JOINTS[1:], self.SIGNS[1:]):
                gear = p.createConstraint(self.body, 1, self.body, joint,
                    jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
                    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
                p.changeConstraint(gear, gearRatio=float(-sign), maxForce=120., erp=.8)
                self.mimic_constraints.append(gear)
        root = kwargs.get("asset_root") or args[2]
        urdf = ET.parse(root / "robotiq_2f_85/robotiq_2f_85.urdf").getroot()
        self.vertices = {}
        for side, link in (("right", 4), ("left", 9)):
            descriptor = urdf.find(f"link[@name='robotiq_2f_85_{side}_pad']/collision/geometry/mesh")
            mesh = trimesh.load(root / "robotiq_2f_85" / descriptor.get("filename"), force="mesh", process=False)
            self.vertices[link] = np.asarray(mesh.vertices) * np.array(list(map(float, descriptor.get("scale").split())))

    def world_pad_vertices(self):
        result = []
        for link in self.PAD_LINKS:
            state = p.getLinkState(self.body, link, computeForwardKinematics=True)
            rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
            result.append(self.vertices[link] @ rotation.T + state[4])
        return result

    def pad_positions(self):
        return tuple(v.mean(axis=0).astype(np.float32) for v in self.world_pad_vertices())

    def width(self):
        right, left = self.world_pad_vertices()
        axis = left.mean(axis=0) - right.mean(axis=0)
        axis /= max(np.linalg.norm(axis), 1e-12)
        return float(max(0., (left @ axis).min() - (right @ axis).max()))

    def _bilateral(self, body):
        contacts = p.getContactPoints(bodyA=self.body, bodyB=body)
        return set(self.PAD_LINKS) <= {c[3] for c in contacts if c[9] > .1}

    def _candidate_between_fingers(self):
        candidates = []
        for body in self.object_ids:
            good = self._bilateral(body)
            self._contact_streak[body] = self._contact_streak.get(body, 0) + 1 if good else 0
            if self._contact_streak[body] < 3:
                continue
            low, high = map(np.asarray, p.getAABB(body))
            centre = self.grasp_center()
            fraction = (centre[2] - low[2]) / max(high[2] - low[2], 1e-6)
            if .25 <= fraction <= .80:
                candidates.append((float(np.linalg.norm(centre - (low+high)/2)), body))
        return min(candidates)[1] if candidates else None

    def _create_assisted_grasp(self, body):
        self._latched = np.array([p.getJointState(self.body, i)[0] for i in self.JOINTS])
        # Small pressure increment, not continued drive through a held object.
        self._latched += .005 * self.SIGNS
        self._lost_contact_ticks = 0
        super()._create_assisted_grasp(body)

    def apply_motor_control(self):
        target = self.target_angle if self.grasp_constraint is None or self._latched is None else self._latched[0]
        p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL,
            targetPosition=float(target), force=30., positionGain=.3, velocityGain=1., maxVelocity=1.5)
        actual = p.getJointState(self.body, 1)[0]
        # Mimic the ACTUAL main actuator, not the requested full-close angle.
        # Otherwise blocked fingers and independently driven followers diverge.
        for joint, sign in zip(self.JOINTS[1:], self.SIGNS[1:]):
            p.setJointMotorControl2(self.body, joint, p.POSITION_CONTROL,
                targetPosition=float(sign * actual), force=60., positionGain=.8,
                velocityGain=1., maxVelocity=1.91986177778)

    def post_step(self):
        if getattr(self, "physics_guard", None) is not None:
            self.physics_guard()
        if self.grasped_object is not None:
            self._lost_contact_ticks = 0 if self._bilateral(self.grasped_object) else self._lost_contact_ticks + 1
            if self._lost_contact_ticks >= 12:
                self._release_assisted_grasp()
        super().post_step()

    def _release_assisted_grasp(self):
        self._latched = None
        self._contact_streak = {}
        self._lost_contact_ticks = 0
        super()._release_assisted_grasp()
