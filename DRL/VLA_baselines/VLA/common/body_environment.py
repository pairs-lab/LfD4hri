"""Opt-in V9 calibrated gripper environment; not a legacy/RL replacement."""
from pathlib import Path
import copy
import numpy as np
import pybullet as p
from VLA.common.provenance import sha256_file

from VLA.common.observation_profile import ProfiledManipulationEnv
from VLA.common.body_gripper import BodyContactRobotiq
from VLA.common.physics_safety import enforce_physics_safety, surface_clearance, physics_diagnostics, PhysicsSafetyViolation

DERIVED_ASSETS = Path(__file__).resolve().parents[1] / "assets/body_grasp_v9"


class BodyGraspEnvironment(ProfiledManipulationEnv):
    def __init__(self, observation_profile, **kwargs):
        super().__init__(observation_profile, **kwargs)
        if self.observation_profile.get("gripper_controller") != "body_contact_v9":
            self.close()
            raise ValueError("Body-contact controller requires a new explicit profile; never reuse a legacy camera/checkpoint hash")
        if self.observation_profile.get("grasp_candidate_scope") != "all_dynamic_scene_objects":
            self.close()
            raise ValueError("V9 requires task-independent grasp candidates across all dynamic objects")
        expected = self.observation_profile.get("derived_gripper_urdf_sha256")
        if expected and sha256_file(DERIVED_ASSETS / "robotiq_2f_85/robotiq_2f_85.urdf") != expected:
            self.close()
            raise ValueError("Derived gripper visual/collision asset differs from the versioned profile")

    def reset(self, task):
        super().reset(task)
        objects = list(self.gripper.object_ids)
        p.removeBody(self.gripper.body)
        self.gripper = BodyContactRobotiq(self.robot_id, self.robot_spec.mount_link, DERIVED_ASSETS, objects,
            mimic_gear=self.observation_profile.get("mimic_coupling") == "gear")
        for _ in range(120):
            self._apply_arm_targets(np.asarray(self.robot_spec.home_joints))
            self.gripper.apply_motor_control()
            p.stepSimulation()
        self.initial_pick_z = p.getBasePositionAndOrientation(self.pick_id)[0][2]
        self.initial_pick_position = np.asarray(p.getBasePositionAndOrientation(self.pick_id)[0])
        self.initial_pick_top, self.initial_pick_bottom = self._pick_top(), self._pick_bottom()
        self.initial_place_top = self._body_top(self.place_id) if self.place_id is not None else None
        self.ground_bodies = [self.table_id] + [p.getBodyUniqueId(i) for i in range(p.getNumBodies())
            if p.getBodyInfo(p.getBodyUniqueId(i))[1].decode() == "plane"]
        state = p.getLinkState(self.robot_id, self.robot_spec.mount_link, computeForwardKinematics=True)
        inverse = p.invertTransform(state[4], state[5])
        relative = p.multiplyTransforms(inverse[0], inverse[1], *p.getBasePositionAndOrientation(self.gripper.body))
        self.rest_mount_relative = (np.array(relative[0]), relative[1])
        self.safety_min_clearance, self.safety_max_mimic_error = float("inf"), 0.
        self.safety_latched = False
        self.gripper.physics_guard = lambda: enforce_physics_safety(self)
        return self.observe()

    def step(self, action):
        if self.safety_latched:
            raise PhysicsSafetyViolation("Reset required after safety abort")
        action = np.asarray(action, dtype=np.float32).copy()
        requested = action.copy()
        clearance = surface_clearance(self)
        # Limit downward translation from the WHOLE gripper's current clearance.
        if action[2] < 0:
            action[2] = max(float(action[2]), -max(0., clearance-.003) * .5)
        snapshot = p.saveState()
        before = {k: copy.deepcopy(getattr(self.gripper, k)) for k in
                  ("grasp_constraint", "grasped_object", "command", "_latched", "_contact_streak", "_lost_contact_ticks")}
        count = self.step_count
        try:
            result = super().step(action)
        except PhysicsSafetyViolation as exc:
            # Capture the offending state BEFORE rollback, not just the safe
            # restored state. This stays in audit metadata, outside model inputs.
            diagnostics = physics_diagnostics(self)
            if before["grasp_constraint"] is None and self.gripper.grasp_constraint is not None:
                self.gripper._release_assisted_grasp()
            restored = True
            try:
                p.restoreState(snapshot)
            except p.error:
                restored = False
            for key, value in before.items():
                setattr(self.gripper, key, value)
            self.safety_latched = True
            self._last_success = False
            self.step_count = count + 1
            metrics = self.semantic_success_metrics()
            info = {"success": False, "raw_success": False, "step": self.step_count,
                "strict_raw_success": False, "semantic_raw_success": False,
                "distance_only_success": False, "task": self.task.name,
                "prompt": self.task.prompt,
                "safety_violation": str(exc), "rollback_restored": restored,
                "safety_diagnostics_before_rollback": diagnostics,
                "grasped_object": self.gripper.grasped_object, "semantic_success_metrics": metrics,
                "ee_object_distance": self.ee_object_distance(), "object_goal_distance": self.object_goal_distance(),
                "touching_object": self.gripper.is_touching(self.pick_id), "success_streak": 0,
                "distance_threshold": self.dist_threshold, "object_threshold": self.obj_threshold,
                **metrics}
            result = (self.observe(), 0., True, info)
        finally:
            p.removeState(snapshot)
        result[3].update(requested_action=requested.tolist(), executed_canonical_action=action.tolist(),
                         safety_clamped_action=bool(np.any(requested != action)))
        return result

    def measured_gripper_closure(self):
        # Encoder/FK aperture estimate, not a direct contact sensor or target ID.
        opening = np.clip((self.gripper.width() - .0063742693) / (.0885956060 - .0063742693), 0, 1)
        return float(1 - opening)

    def normalized_gripper_closure(self):
        if isinstance(self.gripper, BodyContactRobotiq):
            return self.measured_gripper_closure()
        return super().normalized_gripper_closure()

    def commanded_gripper_closure(self):
        return float((self.gripper.command + 1) / 2)

    def is_semantic_success(self, metrics=None):
        reached = super().is_semantic_success(metrics)
        if reached and self.task.name == "pick" and isinstance(self.gripper, BodyContactRobotiq):
            # Same 5cm/one-frame tolerance, but bouncing an unheld object is
            # not a PICK. Evaluator-only identity, never a model input.
            return self.gripper.grasped_object == self.pick_id and self.gripper._bilateral(self.pick_id)
        return reached
