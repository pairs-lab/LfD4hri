"""Shared PyBullet benchmark environment for RL/VLA comparisons.

The scene reuses the robot, object, workspace, and color assets from the DRL
project, while replacing suction with the repository's articulated Robotiq
2F-85.  Policies act through a single end-effector delta-pose contract so model
specific tokenizers never leak into simulator logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pybullet_data

from config import BOUNDS, COLORS

from .action import CanonicalAction
from .observation import VLAObservation
from .robotiq import Robotiq2F85
from .spatial import FIXED_DESTINATION
from .tasks import TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "asset"

CAMERA_CONFIG = {
    "front": {
        # A pulled-back, left-offset overview.  The earlier (-0.42, -1.38)
        # mount made the blocks sharp but cropped most of the arm after a
        # 224 px VLA resize.  This pose keeps the complete UR5e, Robotiq and
        # full random_env workspace in one square frame.
        # Camera-v4 was exhaustively checked over all 19 PICK_TARGETS at a
        # 3x3 grid 1 mm inside the full random_env workspace.  Moving farther
        # back reduces near/far perspective distortion; the narrower FOV
        # preserves at least an 8x8 target footprint at the actual 224 input
        # while keeping both tall objects and the complete robot in frame.
        "eye_world": (-0.65, -1.80, 1.05),
        "target_world": (-0.02, -0.46, 0.27),
        "up_world": (0.0, 0.0, 1.0),
        # Keep the camera at the farther/left position requested for a clear
        # overview, but use a narrower lens so a 4 cm block remains large
        # enough after VLA preprocessors resize the image to 224--256 px.
        "fov_degrees": 38.0,
    },
    "wrist": {
        # Hand-eye audit (36 trajectory frames plus four extrinsic variants)
        # showed that the centred mount let the Robotiq palm occlude the block
        # during approach.  A 5 cm lateral offset moves the gripper into the
        # lower edge while keeping every reset block and the bowl in view.
        "eye_in_tool": (0.07, 0.05, 0.04),
        "target_in_tool": (0.16, 0.02, 0.36),
        "up_axis_in_tool": (0.0, 1.0, 0.0),
        "fov_degrees": 90.0,
    },
}


# Outcome predicates for VLA evaluation.  These intentionally describe the
# meaning of the language task rather than requiring the object to hit the
# single reward-shaping waypoint used by the original RL replay buffer.  They
# are kept explicit and are copied into the locked evaluation protocol.
SEMANTIC_SUCCESS_DEFAULTS = {
    "touch_distance_m": 0.04,
    "pick_min_lift_m": 0.05,
    "move_xy_radius_m": 0.08,
    "move_vertical_below_m": 0.03,
    "move_vertical_above_m": 0.20,
    "pick_place_xy_radius_m": 0.08,
    "pick_place_vertical_below_m": 0.06,
    "pick_place_vertical_above_m": 0.10,
}


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf: Path
    arm_joint_ids: tuple[int, ...]
    mount_link: int
    home_joints: tuple[float, ...]


ROBOT_SPECS = {
    "ur5e": RobotSpec(
        name="ur5e",
        urdf=ASSET_ROOT / "ur5e" / "ur5e.urdf",
        arm_joint_ids=(2, 3, 4, 5, 6, 7),
        mount_link=9,
        home_joints=(
            np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            -np.pi / 2,
            3 * np.pi / 2,
            0.0,
        ),
    ),
    "uf850": RobotSpec(
        name="uf850",
        urdf=ASSET_ROOT / "uf850" / "uf850.urdf",
        arm_joint_ids=(0, 1, 2, 3, 4, 5),
        mount_link=6,
        home_joints=(
            -np.pi / 2,
            -np.pi / 15,
            -2 * np.pi / 9,
            np.pi,
            2 * np.pi / 7,
            0.0,
        ),
    ),
}


class ManipulationEnv:
    """Small deterministic simulator with Gym-like ``reset``/``step`` methods."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        robot: str = "ur5e",
        image_size: int = 256,
        physics_hz: int = 240,
        control_hz: int = 10,
        assist_grasp: bool = True,
        gui: bool = False,
        seed: int = 0,
        required_success_steps: int = 10,
        success_threshold_scale: float = 1.0,
        allow_distance_only_success: bool = False,
        use_semantic_success: bool = False,
        semantic_success_thresholds: Mapping[str, float] | None = None,
        render_observations: bool = True,
    ) -> None:
        if robot not in ROBOT_SPECS:
            raise ValueError(f"robot must be one of {tuple(ROBOT_SPECS)}, got {robot!r}")
        if physics_hz % control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        if required_success_steps < 1:
            raise ValueError("required_success_steps must be at least one")
        if success_threshold_scale <= 0:
            raise ValueError("success_threshold_scale must be positive")
        self.robot_spec = ROBOT_SPECS[robot]
        self.image_size = int(image_size)
        self.physics_hz = int(physics_hz)
        self.control_hz = int(control_hz)
        self.substeps = self.physics_hz // self.control_hz
        self.assist_grasp = assist_grasp
        self.render_observations = bool(render_observations)
        self.rng = np.random.default_rng(seed)
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.task: TaskSpec | None = None
        self.robot_id: int | None = None
        self.gripper: Robotiq2F85 | None = None
        self.object_ids: dict[str, int] = {}
        self._last_segmentation: dict[str, np.ndarray] = {}
        self.pick_id: int | None = None
        self.place_id: int | None = None
        self.step_count = 0
        self.initial_pick_z = 0.0
        self.initial_pick_position = np.zeros(3, dtype=np.float64)
        self.initial_pick_top = np.zeros(3, dtype=np.float64)
        self.initial_pick_bottom = np.zeros(3, dtype=np.float64)
        self.initial_place_top: np.ndarray | None = None
        self._last_success = False
        self.success_streak = 0
        # Demonstration collection deliberately uses ten consecutive success
        # frames, matching Reward.remain: those stable terminal frames prevent
        # successful behavior from being underrepresented during training.
        # Evaluation can use one-shot task completion without changing the
        # collected data or silently changing the underlying reward.xml values.
        self.required_success_steps = int(required_success_steps)
        self.success_threshold_scale = float(success_threshold_scale)
        self.allow_distance_only_success = bool(allow_distance_only_success)
        self.use_semantic_success = bool(use_semantic_success)
        self.semantic_success_thresholds = dict(SEMANTIC_SUCCESS_DEFAULTS)
        if semantic_success_thresholds is not None:
            unknown = set(semantic_success_thresholds) - set(SEMANTIC_SUCCESS_DEFAULTS)
            if unknown:
                raise ValueError(f"Unknown semantic success thresholds: {sorted(unknown)}")
            self.semantic_success_thresholds.update(
                {key: float(value) for key, value in semantic_success_thresholds.items()}
            )
        if any(value < 0 for value in self.semantic_success_thresholds.values()):
            raise ValueError("Semantic success thresholds must be non-negative")
        self.dist_threshold = 0.015
        self.obj_threshold = 0.02

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def __enter__(self) -> "ManipulationEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def reset(self, task: TaskSpec) -> VLAObservation:
        self.task = task
        self.step_count = 0
        self._last_success = False
        self.success_streak = 0
        self.object_ids = {}
        self._last_segmentation = {}
        self.pick_id = None
        self.place_id = None
        self.initial_place_top = None

        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.physics_hz)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            contactERP=0.9,
            solverResidualThreshold=1e-7,
            enableFileCaching=0,
        )

        p.loadURDF(str(ASSET_ROOT / "plane.urdf"), (0, 0, -0.004), useFixedBase=True)
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.45, 0.35, 0.004))
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(0.45, 0.35, 0.004), rgbaColor=(0.2, 0.2, 0.2, 1.0)
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=(0, -0.5, 0),
        )

        self.robot_id = p.loadURDF(
            str(self.robot_spec.urdf),
            (0, 0, 0),
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        for joint_id, home in zip(self.robot_spec.arm_joint_ids, self.robot_spec.home_joints):
            p.resetJointState(self.robot_id, joint_id, home)

        self.pick_id = self._spawn_object(task.pick_name, task.pick_position, fixed=False)
        self.object_ids[task.pick_name] = self.pick_id
        if (
            task.place_name is not None
            and task.place_name not in FIXED_DESTINATION
            and task.place_position is not None
        ):
            self.place_id = self._spawn_object(
                task.place_name,
                task.place_position,
                fixed=task.place_name.endswith(" bowl"),
            )
            self.object_ids[task.place_name] = self.place_id
        for scene_object in task.scene_objects:
            if scene_object.name in self.object_ids:
                raise ValueError(f"Duplicate scene object name {scene_object.name!r}")
            self.object_ids[scene_object.name] = self._spawn_object(
                scene_object.name,
                scene_object.position,
                fixed=scene_object.fixed,
            )

        self.gripper = Robotiq2F85(
            self.robot_id,
            self.robot_spec.mount_link,
            ASSET_ROOT,
            object_ids=(self.pick_id,),
            assist_grasp=self.assist_grasp,
        )

        self._apply_arm_targets(np.asarray(self.robot_spec.home_joints, dtype=np.float32))
        self.gripper.open()
        for _ in range(self.physics_hz):
            self.gripper.apply_motor_control()
            p.stepSimulation()
        self.initial_pick_z = p.getBasePositionAndOrientation(self.pick_id)[0][2]
        self.initial_pick_position = np.asarray(
            p.getBasePositionAndOrientation(self.pick_id)[0], dtype=np.float64
        )
        self.initial_pick_top = self._pick_top()
        self.initial_pick_bottom = self._pick_bottom()
        self.initial_place_top = self._body_top(self.place_id) if self.place_id is not None else None
        base_dist, base_obj = self._read_reward_thresholds(task.task_id)
        self.dist_threshold = base_dist * self.success_threshold_scale
        self.obj_threshold = base_obj * self.success_threshold_scale
        return self.observe()

    def step(self, action: np.ndarray) -> tuple[VLAObservation, float, bool, dict[str, Any]]:
        if self.task is None or self.robot_id is None or self.gripper is None or self.pick_id is None:
            raise RuntimeError("reset(task) must be called before step(action)")
        canonical = CanonicalAction.from_array(action)
        mount_state = p.getLinkState(
            self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True
        )
        current_position = np.asarray(mount_state[4], dtype=np.float32)
        current_orientation = mount_state[5]
        target_position = current_position + canonical.translation
        target_position = np.clip(
            target_position,
            np.array((BOUNDS[0, 0] - 0.12, BOUNDS[1, 0] - 0.05, 0.10)),
            np.array((BOUNDS[0, 1] + 0.12, BOUNDS[1, 1] + 0.05, 0.72)),
        )
        delta_quaternion = p.getQuaternionFromEuler(canonical.rotation.tolist())
        target_orientation = p.multiplyTransforms(
            (0, 0, 0), current_orientation, (0, 0, 0), delta_quaternion
        )[1]

        ik_solution = p.calculateInverseKinematics(
            self.robot_id,
            self.robot_spec.mount_link,
            target_position.tolist(),
            targetOrientation=target_orientation,
            maxNumIterations=120,
            residualThreshold=1e-5,
        )
        arm_targets = np.asarray(ik_solution[: len(self.robot_spec.arm_joint_ids)], dtype=np.float32)
        self.gripper.set_command(canonical.gripper)
        for _ in range(self.substeps):
            self._apply_arm_targets(arm_targets)
            self.gripper.apply_motor_control()
            p.stepSimulation()
            self.gripper.post_step()

        self.step_count += 1
        strict_raw_success = self.is_strict_success()
        semantic_metrics = self.semantic_success_metrics()
        semantic_raw_success = self.is_semantic_success(semantic_metrics)
        distance_only_success = (
            self.task.name in {"pick", "move", "pick_place"}
            and self.object_goal_distance() < self.obj_threshold
        )
        if self.use_semantic_success:
            raw_success = semantic_raw_success
        else:
            raw_success = strict_raw_success or (
                self.allow_distance_only_success and distance_only_success
            )
        self.success_streak = self.success_streak + 1 if raw_success else 0
        success = self.success_streak >= self.required_success_steps
        terminated = success or self.step_count >= self.task.max_steps
        reward = 1.0 if raw_success else self._dense_progress_reward()
        self._last_success = raw_success
        info = {
            "success": success,
            "raw_success": raw_success,
            "strict_raw_success": strict_raw_success,
            "semantic_raw_success": semantic_raw_success,
            "distance_only_success": distance_only_success,
            "success_streak": self.success_streak,
            "task": self.task.name,
            "prompt": self.task.prompt,
            "step": self.step_count,
            "grasped_object": self.gripper.grasped_object,
            "touching_object": self.gripper.is_touching(self.pick_id),
            "ee_object_distance": self.ee_object_distance(),
            "object_goal_distance": self.object_goal_distance(),
            **semantic_metrics,
            "distance_threshold": self.dist_threshold,
            "object_threshold": self.obj_threshold,
        }
        return self.observe(), float(reward), terminated, info

    def observe(self) -> VLAObservation:
        if self.task is None or self.robot_id is None or self.gripper is None:
            raise RuntimeError("Environment has not been reset")
        joint_positions = np.asarray(
            [p.getJointState(self.robot_id, joint)[0] for joint in self.robot_spec.arm_joint_ids],
            dtype=np.float32,
        )
        mount_state = p.getLinkState(
            self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True
        )
        rotation = np.asarray(p.getMatrixFromQuaternion(mount_state[5]), dtype=np.float32).reshape(3, 3)
        # RDT's continuous 6D representation is the first two rotation-matrix
        # columns, transposed and flattened.
        rotation_6d = rotation[:, :2].T.reshape(6)
        if self.render_observations:
            front_rgb = self.render("front")
            wrist_rgb = self.render("wrist")
        else:
            # Physics-only audits need measured proprioception but never feed
            # RGB to a policy. Avoiding TinyRenderer makes exhaustive adapter
            # replay practical without changing any controller or success
            # semantics. Learned-policy evaluation always keeps the default.
            front_rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            wrist_rgb = np.zeros_like(front_rgb)
        return VLAObservation(
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
            joint_positions=joint_positions,
            gripper_width=self.gripper.width(),
            gripper_open=1.0 - self.normalized_gripper_closure(),
            ee_position=np.asarray(mount_state[4], dtype=np.float32),
            ee_rotation_6d=rotation_6d,
            prompt=self.task.prompt,
            step=self.step_count,
        )

    def render(self, camera: str = "front") -> np.ndarray:
        if camera == "front":
            # Pulled back and shifted left so the whole robot, gripper, table,
            # and target remain visible throughout a rollout.
            eye = CAMERA_CONFIG["front"]["eye_world"]
            target = CAMERA_CONFIG["front"]["target_world"]
            up = CAMERA_CONFIG["front"]["up_world"]
        elif camera == "wrist":
            if self.robot_id is None or self.gripper is None:
                raise RuntimeError("Environment has not been reset")
            mount_state = p.getLinkState(
                self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True
            )
            # Rigid hand-eye transform in the robot tool frame.  On the UR5e
            # home pose local +Z points toward the table; the lateral offset
            # keeps both fingers out of the optical centre.
            eye_pose = p.multiplyTransforms(
                mount_state[0],
                mount_state[1],
                CAMERA_CONFIG["wrist"]["eye_in_tool"],
                (0, 0, 0, 1),
            )
            target_pose = p.multiplyTransforms(
                mount_state[0],
                mount_state[1],
                CAMERA_CONFIG["wrist"]["target_in_tool"],
                (0, 0, 0, 1),
            )
            up_pose = p.multiplyTransforms(
                (0, 0, 0),
                mount_state[1],
                CAMERA_CONFIG["wrist"]["up_axis_in_tool"],
                (0, 0, 0, 1),
            )
            eye, target, up = eye_pose[0], target_pose[0], up_pose[0]
        else:
            raise ValueError("camera must be 'front' or 'wrist'")

        view = p.computeViewMatrix(eye, target, up)
        projection = p.computeProjectionMatrixFOV(
            fov=CAMERA_CONFIG[camera]["fov_degrees"],
            aspect=1.0,
            nearVal=0.01,
            farVal=3.0,
        )
        _, _, rgba, _, segmentation = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
        )
        self._last_segmentation[camera] = np.asarray(segmentation, dtype=np.int64).reshape(
            self.image_size, self.image_size
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(
            self.image_size, self.image_size, 4
        )[:, :, :3]

    def target_mask(self, camera: str, target: str = "pick") -> np.ndarray:
        """Return an audit-only segmentation mask for the last rendered observation.

        This method is intentionally absent from :class:`VLAObservation`; it
        verifies camera coverage without leaking simulator IDs to policies.
        """
        if camera not in self._last_segmentation:
            raise RuntimeError(f"Camera {camera!r} has not been rendered for this observation")
        if target == "pick":
            body_id = self.pick_id
        elif target == "place":
            body_id = self.place_id
        else:
            raise ValueError("target must be 'pick' or 'place'")
        if body_id is None:
            return np.zeros_like(self._last_segmentation[camera], dtype=bool)
        packed = self._last_segmentation[camera]
        object_ids = np.where(packed >= 0, packed & ((1 << 24) - 1), -1)
        return object_ids == body_id

    def robot_mask(self, camera: str) -> np.ndarray:
        """Audit-only mask for the UR5e and attached Robotiq in the last frame."""
        if camera not in self._last_segmentation:
            raise RuntimeError(f"Camera {camera!r} has not been rendered for this observation")
        if self.robot_id is None or self.gripper is None:
            raise RuntimeError("Environment has not been reset")
        packed = self._last_segmentation[camera]
        object_ids = np.where(packed >= 0, packed & ((1 << 24) - 1), -1)
        return (object_ids == self.robot_id) | (object_ids == self.gripper.body)

    def is_success(self) -> bool:
        """Configured success used for termination and evaluation."""
        if self.use_semantic_success:
            return self.is_semantic_success()
        strict = self.is_strict_success()
        if strict or not self.allow_distance_only_success or self.task is None:
            return strict
        return (
            self.task.name in {"pick", "move", "pick_place"}
            and self.object_goal_distance() < self.obj_threshold
        )

    def is_strict_success(self) -> bool:
        """Original reward.py predicate, retained as an auditable diagnostic."""
        if self.task is None or self.pick_id is None or self.gripper is None:
            return False
        object_position = np.asarray(p.getBasePositionAndOrientation(self.pick_id)[0])
        if self.task.name == "touch":
            ee_to_object = np.linalg.norm(self.gripper.grasp_center() - self._pick_top())
            return ee_to_object < self.dist_threshold or self.gripper.is_touching(self.pick_id)
        if self.task.name in {"pick", "move", "pick_place"}:
            del object_position
            # individual_reward.py compares the object's AABB bottom point to
            # env.target, not its centre of mass.  Keeping that convention is
            # important for pick: its target is initial object top + 15 cm.
            object_to_goal = self.object_goal_distance()
            # Mirrors individual_reward.py: pick/move/place require active
            # gripper contact and the object-goal error below threshold1.
            return self.gripper.grasped_object == self.pick_id and object_to_goal < self.obj_threshold
        raise ValueError(f"Unsupported task {self.task.name!r}")

    def semantic_success_metrics(self) -> dict[str, float | None]:
        """Task-level outcome measures that do not expose goal pose to a policy."""
        if self.task is None or self.pick_id is None:
            raise RuntimeError("Environment has not been reset")
        pick_bottom = self._pick_bottom()
        place = self._place_reference()
        return {
            "pick_lift_m": float(pick_bottom[2] - self.initial_pick_bottom[2]),
            "goal_xy_distance_m": (
                None if place is None else float(np.linalg.norm(pick_bottom[:2] - place[:2]))
            ),
            "goal_vertical_offset_m": (
                None if place is None else float(pick_bottom[2] - place[2])
            ),
        }

    def is_semantic_success(self, metrics: Mapping[str, float | None] | None = None) -> bool:
        """Outcome predicate analogous to a benchmark target-region check.

        The policy never receives these simulator coordinates.  They are used
        only by the evaluator, just as LIBERO evaluates a language-conditioned
        policy with privileged BDDL predicates.
        """
        if self.task is None or self.pick_id is None or self.gripper is None:
            return False
        thresholds = self.semantic_success_thresholds
        if self.task.name == "touch":
            return bool(
                self.gripper.is_touching(self.pick_id)
                or self.ee_object_distance() <= thresholds["touch_distance_m"]
            )
        values = self.semantic_success_metrics() if metrics is None else metrics
        if self.task.name == "pick":
            return float(values["pick_lift_m"]) >= thresholds["pick_min_lift_m"]
        xy = values["goal_xy_distance_m"]
        vertical = values["goal_vertical_offset_m"]
        if xy is None or vertical is None:
            return False
        if self.task.name == "move":
            return bool(
                float(xy) <= thresholds["move_xy_radius_m"]
                and -thresholds["move_vertical_below_m"]
                <= float(vertical)
                <= thresholds["move_vertical_above_m"]
            )
        if self.task.name == "pick_place":
            return bool(
                float(xy) <= thresholds["pick_place_xy_radius_m"]
                and -thresholds["pick_place_vertical_below_m"]
                <= float(vertical)
                <= thresholds["pick_place_vertical_above_m"]
            )
        raise ValueError(f"Unsupported task {self.task.name!r}")

    def debug_state(self) -> dict[str, Any]:
        """Simulator state for diagnostics/oracle only; never passed to a VLA."""
        if self.pick_id is None or self.gripper is None or self.task is None:
            raise RuntimeError("Environment has not been reset")
        return {
            "pick_position": np.asarray(p.getBasePositionAndOrientation(self.pick_id)[0]),
            "initial_pick_position": self.initial_pick_position.copy(),
            "initial_pick_top": self.initial_pick_top.copy(),
            "initial_pick_bottom": self.initial_pick_bottom.copy(),
            "pick_top": self._pick_top(),
            "pick_bottom": self._pick_bottom(),
            "pick_center": (self._pick_top() + self._pick_bottom()) * 0.5,
            "grasp_center": self.gripper.grasp_center(),
            "grasped": self.gripper.grasped_object == self.pick_id,
            "place_position": self._place_reference(),
            "goal_position": self._object_goal(),
            "reward_target": self._reward_target(),
            "dist_threshold": self.dist_threshold,
            "obj_threshold": self.obj_threshold,
            "semantic_success": self.is_semantic_success(),
            "semantic_success_metrics": self.semantic_success_metrics(),
            "semantic_success_thresholds": dict(self.semantic_success_thresholds),
        }

    def normalized_gripper_closure(self) -> float:
        """Return 0=open, 1=closed for UR5-style VLA proprioception."""
        if self.gripper is None:
            raise RuntimeError("Environment has not been reset")
        return float(np.clip((self.gripper.command + 1.0) * 0.5, 0.0, 1.0))

    def joint_targets_to_canonical(
        self, joint_targets: np.ndarray, gripper_closure: float
    ) -> np.ndarray:
        """Map absolute UR5 joint targets to the shared delta-EEF contract.

        The official openpi UR5 recipe predicts six absolute joint targets plus
        a 0=open/1=closed gripper value. Exact forward kinematics converts the
        target to the same 7D delta-EEF contract used by every other VLA.

        A previous first-order Jacobian conversion had sub-2 mm mean one-step
        error but accumulated enough drift over an episode to miss grasps. FK
        pose differencing is exact for the target/current pair and therefore
        remains stable when a queued action is evaluated against the latest
        measured state.
        """
        if self.robot_id is None:
            raise RuntimeError("Environment has not been reset")
        joint_states = [
            p.getJointState(self.robot_id, joint) for joint in self.robot_spec.arm_joint_ids
        ]
        current = np.asarray([state[0] for state in joint_states], dtype=np.float64)
        current_velocities = np.asarray([state[1] for state in joint_states], dtype=np.float64)
        targets = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        if targets.shape != current.shape:
            raise ValueError(f"Expected {current.size} joint targets, got {targets.shape}")
        current_link = p.getLinkState(
            self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True
        )
        current_position = np.asarray(current_link[4], dtype=np.float64)
        current_orientation = current_link[5]
        try:
            for joint, target in zip(
                self.robot_spec.arm_joint_ids, targets, strict=True
            ):
                p.resetJointState(self.robot_id, joint, float(target))
            target_link = p.getLinkState(
                self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True
            )
        finally:
            for joint, position, velocity in zip(
                self.robot_spec.arm_joint_ids,
                current,
                current_velocities,
                strict=True,
            ):
                p.resetJointState(
                    self.robot_id,
                    joint,
                    float(position),
                    targetVelocity=float(velocity),
                )
        target_position = np.asarray(target_link[4], dtype=np.float64)
        target_orientation = target_link[5]
        translation = target_position - current_position
        inverse_current = p.invertTransform((0.0, 0.0, 0.0), current_orientation)[1]
        relative_orientation = p.multiplyTransforms(
            (0.0, 0.0, 0.0),
            inverse_current,
            (0.0, 0.0, 0.0),
            target_orientation,
        )[1]
        rotation = np.asarray(p.getEulerFromQuaternion(relative_orientation), dtype=np.float64)
        rotation = (rotation + np.pi) % (2.0 * np.pi) - np.pi
        return np.concatenate(
            (
                np.clip(translation, -0.05, 0.05),
                np.clip(rotation, -0.25, 0.25),
                np.array((2.0 * float(gripper_closure) - 1.0,)),
            )
        ).astype(np.float32)

    def _apply_arm_targets(self, targets: np.ndarray) -> None:
        assert self.robot_id is not None
        p.setJointMotorControlArray(
            self.robot_id,
            self.robot_spec.arm_joint_ids,
            p.POSITION_CONTROL,
            targetPositions=targets.tolist(),
            forces=(250.0,) * len(self.robot_spec.arm_joint_ids),
            positionGains=(0.25,) * len(self.robot_spec.arm_joint_ids),
            velocityGains=(1.0,) * len(self.robot_spec.arm_joint_ids),
        )

    def _spawn_object(
        self, name: str, position: tuple[float, float, float], fixed: bool
    ) -> int:
        color_name, object_type = name.split(" ", 1)
        color = COLORS[color_name]
        if object_type == "block":
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.02, 0.02, 0.02))
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=(0.02, 0.02, 0.02), rgbaColor=color
            )
            body = p.createMultiBody(
                baseMass=0.0 if fixed else 0.05,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position,
            )
        elif object_type == "egg":
            collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.03,
                rgbaColor=color,
            )
            body = p.createMultiBody(
                baseMass=0.0 if fixed else 0.01,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position,
            )
        else:
            urdf = ASSET_ROOT / object_type / f"{object_type}.urdf"
            if not urdf.exists():
                raise FileNotFoundError(f"No asset for {name!r}: {urdf}")
            body = p.loadURDF(str(urdf), position, useFixedBase=fixed)
            p.changeVisualShape(body, -1, rgbaColor=color)
        p.changeDynamics(
            body,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.02,
            rollingFriction=0.01,
            restitution=0.0,
            linearDamping=0.25,
            angularDamping=0.25,
        )
        return body

    def _dense_progress_reward(self) -> float:
        if self.pick_id is None or self.gripper is None or self.task is None:
            return 0.0
        obj = np.asarray(p.getBasePositionAndOrientation(self.pick_id)[0])
        tip = self.gripper.grasp_center()
        if self.task.name == "touch":
            return -float(np.linalg.norm(tip - obj))
        if self.gripper.grasped_object != self.pick_id:
            return -float(np.linalg.norm(tip - obj))
        if self.task.place_position is None:
            return float(obj[2] - self.initial_pick_z)
        return -self.object_goal_distance()

    def ee_object_distance(self) -> float:
        """Distance used by the touch success predicate."""
        if self.gripper is None:
            raise RuntimeError("Environment has not been reset")
        return float(np.linalg.norm(self.gripper.grasp_center() - self._pick_top()))

    def object_goal_distance(self) -> float:
        """Exact bottom-of-object to target distance from individual_reward.py."""
        return float(np.linalg.norm(self._pick_bottom() - self._reward_target()))

    def _pick_top(self) -> np.ndarray:
        assert self.pick_id is not None
        aabb_min, aabb_max = p.getAABB(self.pick_id)
        top = (np.asarray(aabb_min, dtype=np.float64) + np.asarray(aabb_max, dtype=np.float64)) * 0.5
        top[2] = aabb_max[2]
        return top

    def _pick_bottom(self) -> np.ndarray:
        assert self.pick_id is not None
        aabb_min, aabb_max = p.getAABB(self.pick_id)
        bottom = (np.asarray(aabb_min, dtype=np.float64) + np.asarray(aabb_max, dtype=np.float64)) * 0.5
        bottom[2] = aabb_min[2]
        return bottom

    @staticmethod
    def _body_top(body_id: int) -> np.ndarray:
        aabb_min, aabb_max = p.getAABB(body_id)
        top = (np.asarray(aabb_min, dtype=np.float64) + np.asarray(aabb_max, dtype=np.float64)) * 0.5
        top[2] = aabb_max[2]
        return top

    def _place_reference(self) -> np.ndarray | None:
        if self.task is None or self.task.place_position is None:
            return None
        if self.initial_place_top is not None:
            return self.initial_place_top.copy()
        return np.asarray(self.task.place_position, dtype=np.float64)

    def _reward_target(self) -> np.ndarray:
        """Reproduce RobotEnv._coor for the four benchmark tasks."""
        assert self.task is not None
        if self.task.name == "pick":
            return self.initial_pick_top + np.array((0.0, 0.0, 0.15))
        if self.task.name == "move":
            place = self._place_reference()
            assert place is not None
            return place + np.array((0.0, 0.0, 0.15))
        if self.task.name == "pick_place":
            place = self._place_reference()
            assert place is not None
            return place + np.array((0.0, 0.0, 0.015))
        return self._pick_top()

    def _object_goal(self) -> np.ndarray:
        assert self.pick_id is not None
        # This centre-of-mass target is only for the oracle controller.  The
        # success check above remains the exact AABB-bottom reward predicate.
        object_position = np.asarray(p.getBasePositionAndOrientation(self.pick_id)[0])
        return object_position + (self._reward_target() - self._pick_bottom())

    @staticmethod
    def _read_reward_thresholds(task_id: int) -> tuple[float, float]:
        root = ET.parse(ASSET_ROOT / "params" / "reward.xml").getroot()
        node = root.find(f"r{task_id}")
        if node is None:
            raise KeyError(f"reward.xml has no r{task_id} entry")
        return float(node.attrib["threshold0"]), float(node.attrib["threshold1"])
