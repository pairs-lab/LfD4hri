"""Deterministic PyBullet controller for the repository's Robotiq 2F-85 asset.

This uses the actual articulated gripper URDF under ``asset/robotiq_2f_85``.
An optional contact-based fixed constraint stabilizes a grasp after both fingers
have closed around an object.  It is deliberately identical for every policy and
is not a suction replacement: the constraint can only be created while the
object is inside the closing finger envelope and is released when the fingers
open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pybullet as p


class Robotiq2F85:
    OPEN_ANGLE = 0.0
    CLOSED_ANGLE = 0.78
    PAD_LINKS = (4, 9)

    def __init__(
        self,
        robot_id: int,
        mount_link: int,
        asset_root: Path,
        object_ids: Iterable[int],
        assist_grasp: bool = True,
    ) -> None:
        self.robot_id = robot_id
        self.mount_link = mount_link
        self.object_ids = list(object_ids)
        self.assist_grasp = assist_grasp
        self.command = -1.0
        self.grasp_constraint: int | None = None
        self.grasped_object: int | None = None

        mount_state = p.getLinkState(robot_id, mount_link, computeForwardKinematics=True)
        self.body = p.loadURDF(
            str(asset_root / "robotiq_2f_85" / "robotiq_2f_85.urdf"),
            mount_state[4],
            mount_state[5],
            useFixedBase=False,
        )

        # This is the same transform used by the repository's original
        # utils/Robotiq2F85.py, but attached to the gripper base rather than run
        # from a background thread.
        self.mount_constraint = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=mount_link,
            childBodyUniqueId=self.body,
            childLinkIndex=0,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.07),
            childFrameOrientation=p.getQuaternionFromEuler((0, 0, np.pi / 2)),
        )
        p.changeConstraint(self.mount_constraint, maxForce=1e7)

        for robot_link in range(-1, p.getNumJoints(robot_id)):
            for gripper_link in range(-1, p.getNumJoints(self.body)):
                p.setCollisionFilterPair(robot_id, self.body, robot_link, gripper_link, 0)

        for link in range(-1, p.getNumJoints(self.body)):
            p.changeDynamics(
                self.body,
                link,
                lateralFriction=2.0,
                spinningFriction=0.05,
                rollingFriction=0.001,
                restitution=0.0,
            )

        self.open()

    def open(self) -> None:
        self.command = -1.0
        self._release_assisted_grasp()

    def close(self) -> None:
        self.command = 1.0

    def set_command(self, command: float) -> None:
        if command > 0.2:
            self.close()
        elif command < -0.2:
            self.open()

    @property
    def target_angle(self) -> float:
        alpha = (self.command + 1.0) * 0.5
        return float(self.OPEN_ANGLE + alpha * (self.CLOSED_ANGLE - self.OPEN_ANGLE))

    def apply_motor_control(self) -> None:
        q = self.target_angle
        joint_ids = (1, 3, 5, 6, 8, 10)
        targets = (q, -q, q, q, -q, q)
        p.setJointMotorControlArray(
            self.body,
            joint_ids,
            p.POSITION_CONTROL,
            targetPositions=targets,
            forces=(120.0,) * len(joint_ids),
            positionGains=(0.8,) * len(joint_ids),
        )

    def post_step(self) -> None:
        if self.command > 0.2 and self.grasp_constraint is None and self.assist_grasp:
            candidate = self._candidate_between_fingers()
            if candidate is not None:
                self._create_assisted_grasp(candidate)

    def pad_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            np.asarray(p.getLinkState(self.body, link, computeForwardKinematics=True)[0], dtype=np.float32)
            for link in self.PAD_LINKS
        )  # type: ignore[return-value]

    def grasp_center(self) -> np.ndarray:
        left, right = self.pad_positions()
        return (left + right) * 0.5

    def width(self) -> float:
        left, right = self.pad_positions()
        return float(np.linalg.norm(left - right))

    def is_touching(self, object_id: int) -> bool:
        return bool(p.getContactPoints(bodyA=self.body, bodyB=object_id))

    def _candidate_between_fingers(self) -> int | None:
        center = self.grasp_center()
        left, right = self.pad_positions()
        finger_axis = right - left
        finger_distance = float(np.linalg.norm(finger_axis))
        if finger_distance < 1e-6:
            return None
        finger_axis /= finger_distance

        best: tuple[float, int] | None = None
        for object_id in self.object_ids:
            aabb_min, aabb_max = p.getAABB(object_id)
            position = (
                np.asarray(aabb_min, dtype=np.float32)
                + np.asarray(aabb_max, dtype=np.float32)
            ) * 0.5
            offset = position - center
            along = abs(float(np.dot(offset, finger_axis)))
            radial = float(np.linalg.norm(offset - np.dot(offset, finger_axis) * finger_axis))
            has_contact = bool(p.getContactPoints(bodyA=self.body, bodyB=object_id))
            # Require either physical contact or a very tight geometric envelope.
            eligible = has_contact or (along < finger_distance * 0.55 and radial < 0.047)
            distance = float(np.linalg.norm(offset))
            if eligible and distance < 0.075 and (best is None or distance < best[0]):
                best = (distance, object_id)
        return None if best is None else best[1]

    def _create_assisted_grasp(self, object_id: int) -> None:
        parent_pose = p.getLinkState(self.body, 0, computeForwardKinematics=True)[:2]
        child_pose = p.getBasePositionAndOrientation(object_id)
        world_to_parent = p.invertTransform(parent_pose[0], parent_pose[1])
        child_in_parent = p.multiplyTransforms(
            world_to_parent[0], world_to_parent[1], child_pose[0], child_pose[1]
        )
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.body,
            parentLinkIndex=0,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=child_in_parent[0],
            parentFrameOrientation=child_in_parent[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0, 1),
        )
        p.changeConstraint(self.grasp_constraint, maxForce=250.0)
        self.grasped_object = object_id

    def _release_assisted_grasp(self) -> None:
        if self.grasp_constraint is not None:
            try:
                p.removeConstraint(self.grasp_constraint)
            except p.error:
                pass
        self.grasp_constraint = None
        self.grasped_object = None
