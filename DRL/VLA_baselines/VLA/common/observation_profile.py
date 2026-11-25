"""Opt-in, versioned VLA sensor/home configuration; legacy/RL stays unchanged.

Camera mounts are rigid transforms in the URDF tool *link* frame. No camera
may follow a selected object or use task coordinates. Profiles must travel
with collected data and deployment; changing a profile requires new images.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as p

from .environment import CAMERA_CONFIG, ManipulationEnv, ROBOT_SPECS


def profile_hash(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_profile(profile):
    if profile.get("schema_version") != 1 or profile.get("robot") != "ur5e":
        raise ValueError("Only schema-v1 UR5e profiles supported")
    if profile.get("wrist_frame") != "tool_link":
        raise ValueError("Hand-eye frame must explicitly be tool_link")
    if profile.get("grasp_candidate_scope", "selected_target_legacy") not in (
            "selected_target_legacy", "all_dynamic_scene_objects"):
        raise ValueError("Unknown grasp candidate scope")
    if set(profile["cameras"]) - {"front", "wrist", "overview"}:
        raise ValueError("Only two policy cameras and optional video-only overview supported")
    role = profile.get("front_camera_role", "robot_overview")
    if role not in ("robot_overview", "policy_workspace") or (role == "policy_workspace" and "overview" not in profile["cameras"]):
        raise ValueError("Workspace-focused policy camera needs an explicit video-only overview")
    q = np.asarray(profile["home_joints"], dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Six finite home joints required")
    camera_vectors = [("front", ("eye_world", "target_world", "up_world")),
                      ("wrist", ("eye_in_tool", "target_in_tool", "up_axis_in_tool"))]
    if "overview" in profile["cameras"]:
        camera_vectors.append(("overview", ("eye_world", "target_world", "up_world")))
    for camera, names in camera_vectors:
        cfg = profile["cameras"][camera]
        vectors = [np.asarray(cfg[name], dtype=float) for name in names]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("Finite camera vectors required")
        direction = vectors[1] - vectors[0]
        if np.linalg.norm(direction) < 1e-6 or np.linalg.norm(np.cross(direction, vectors[2])) < 1e-6:
            raise ValueError("Degenerate camera look/up vectors")
        if not 10 <= float(cfg["fov_degrees"]) <= 140:
            raise ValueError("Camera FOV outside supported range")
    profile_hash(profile)  # rejects NaN in any extra metadata
    return copy.deepcopy(profile)


def baseline_profile():
    return {"schema_version": 1, "robot": "ur5e", "name": "legacy_v6_geometry",
            "wrist_frame": "tool_link", "home_joints": list(ROBOT_SPECS["ur5e"].home_joints),
            "cameras": copy.deepcopy(CAMERA_CONFIG)}


def camera_geometry(env, cfg, camera):
    if camera != "wrist":
        eye, target, up = (cfg[key] for key in ("eye_world", "target_world", "up_world"))
    else:
        state = p.getLinkState(env.robot_id, env.robot_spec.mount_link, computeForwardKinematics=True,
                              physicsClientId=env.client_id)
        origin, quat = state[4], state[5]
        eye = p.multiplyTransforms(origin, quat, cfg["eye_in_tool"], (0, 0, 0, 1))[0]
        target = p.multiplyTransforms(origin, quat, cfg["target_in_tool"], (0, 0, 0, 1))[0]
        up = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3) @ cfg["up_axis_in_tool"]
    return np.asarray(eye), np.asarray(target), np.asarray(up)


def render_geometry(env, geometry, fov, size):
    view = p.computeViewMatrix(*[v.tolist() for v in geometry])
    projection = p.computeProjectionMatrixFOV(float(fov), 1., .01, 3.)
    result = p.getCameraImage(size, size, viewMatrix=view, projectionMatrix=projection,
                              renderer=p.ER_TINY_RENDERER,
                              flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                              physicsClientId=env.client_id)
    rgb = np.asarray(result[2], dtype=np.uint8).reshape(size, size, 4)[..., :3]
    seg = np.asarray(result[4], dtype=np.int64).reshape(size, size)
    return rgb, seg, view, projection


class ProfiledManipulationEnv(ManipulationEnv):
    def reset(self, task):
        observation = super().reset(task)
        if self.observation_profile.get("grasp_candidate_scope") == "all_dynamic_scene_objects":
            # Task identity may determine scoring, NEVER which nearby object
            # the same physical gripper is permitted to grasp. Preserve the
            # legacy behavior only for explicitly historical profiles.
            self.gripper.object_ids = sorted(body for body in self.object_ids.values()
                                            if p.getDynamicsInfo(body, -1, physicsClientId=self.client_id)[0] > 0)
        return observation

    def _apply_arm_targets(self, targets):
        # Requested motor targets are action labels; next measured joints are
        # feedback/state. Preserve both instead of introducing servo-lag bias
        # into target-domain absolute-joint action supervision.
        self.requested_arm_targets = np.asarray(targets, dtype=np.float32).copy()
        super()._apply_arm_targets(targets)

    def __init__(self, observation_profile, **kwargs):
        if isinstance(observation_profile, (str, Path)):
            observation_profile = json.loads(Path(observation_profile).read_text())
        self.observation_profile = validate_profile(observation_profile)
        self.observation_profile_sha256 = profile_hash(self.observation_profile)
        super().__init__(**kwargs)
        if self.robot_spec.name != self.observation_profile["robot"]:
            self.close()
            raise ValueError("Robot/profile mismatch")
        self.robot_spec = replace(self.robot_spec, home_joints=tuple(self.observation_profile["home_joints"]))

    def render(self, camera="front"):
        if camera not in self.observation_profile["cameras"]:
            raise ValueError(camera)
        cfg = self.observation_profile["cameras"][camera]
        geometry = camera_geometry(self, cfg, camera)
        rgb, seg, _, _ = render_geometry(self, geometry, cfg["fov_degrees"], self.image_size)
        self._last_segmentation[camera] = seg
        return rgb
