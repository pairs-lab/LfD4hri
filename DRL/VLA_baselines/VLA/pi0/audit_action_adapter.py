#!/usr/bin/env python3
"""Verify that openpi's native UR5 joint targets preserve oracle EEF actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure
import pybullet as p

from VLA.common.environment import ROBOT_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            PROJECT_ROOT
            / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "VLA/audits/camera_v4_vla_oneshot/pi0_action_adapter.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    args = parse_args()
    from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
    profile = json.loads((args.source/"dataset_manifest.json").read_text()).get("observation_profile")
    runtime_fingerprint = environment_runtime_fingerprint(profile)
    index = json.loads((args.source / "index.json").read_text())
    spec = ROBOT_SPECS["ur5e"]
    bullet_client = p.connect(p.DIRECT)
    robot = p.loadURDF(str(spec.urdf), (0, 0, 0), useFixedBase=True)
    expected_rows: list[np.ndarray] = []
    reconstructed_rows: list[np.ndarray] = []
    try:
        for item in index:
            with np.load(args.source / item["file"]) as episode:
                for current, target, closure, expected in zip(
                    episode["joint_positions"],
                    episode["next_joint_targets"],
                    native_gripper_closure(episode),
                    episode["canonical_actions"],
                    strict=True,
                ):
                    current = wrap(np.asarray(current, dtype=np.float64))
                    target = wrap(np.asarray(target, dtype=np.float64))
                    for joint, value in zip(spec.arm_joint_ids, current, strict=True):
                        p.resetJointState(robot, joint, float(value))
                    current_link = p.getLinkState(
                        robot, spec.mount_link, computeForwardKinematics=True
                    )
                    for joint, value in zip(spec.arm_joint_ids, target, strict=True):
                        p.resetJointState(robot, joint, float(value))
                    target_link = p.getLinkState(
                        robot, spec.mount_link, computeForwardKinematics=True
                    )
                    translation = np.asarray(target_link[4]) - np.asarray(current_link[4])
                    inverse_current = p.invertTransform(
                        (0.0, 0.0, 0.0), current_link[5]
                    )[1]
                    relative = p.multiplyTransforms(
                        (0.0, 0.0, 0.0),
                        inverse_current,
                        (0.0, 0.0, 0.0),
                        target_link[5],
                    )[1]
                    rotation = np.asarray(
                        p.getEulerFromQuaternion(relative), dtype=np.float64
                    )
                    rotation = (rotation + np.pi) % (2.0 * np.pi) - np.pi
                    reconstructed = np.concatenate(
                        (
                            np.clip(translation, -0.05, 0.05),
                            np.clip(rotation, -0.25, 0.25),
                            np.asarray((2.0 * float(closure) - 1.0,)),
                        )
                    )
                    expected_rows.append(np.asarray(expected, dtype=np.float64))
                    reconstructed_rows.append(reconstructed)
    finally:
        p.disconnect(bullet_client)

    expected = np.asarray(expected_rows)
    reconstructed = np.asarray(reconstructed_rows)
    error = reconstructed - expected
    correlations = []
    for dimension in range(6):
        if np.std(expected[:, dimension]) == 0 or np.std(reconstructed[:, dimension]) == 0:
            correlations.append(None)
        else:
            correlations.append(
                float(np.corrcoef(expected[:, dimension], reconstructed[:, dimension])[0, 1])
            )
    translation_mae = np.mean(np.abs(error[:, :3]), axis=0)
    rotation_mae = np.mean(np.abs(error[:, 3:6]), axis=0)
    gripper_max_error = float(np.max(np.abs(error[:, 6])))
    passed = bool(
        len(expected) == sum(int(item["steps"]) for item in index)
        and max(translation_mae) < 0.0025
        and max(rotation_mae) < 0.003
        and gripper_max_error == 0.0
        and min(value for value in correlations[:3] if value is not None) > 0.99
    )
    environment_path = PROJECT_ROOT / "VLA/common/environment.py"
    verify_environment_runtime_fingerprint(runtime_fingerprint, profile)
    report = {
        "passed": passed,
        "source": str(args.source.resolve()),
        "source_index_sha256": sha256(args.source / "index.json"),
        "source_manifest_sha256": sha256(args.source / "dataset_manifest.json"),
        "adapter_source": str(environment_path.resolve()),
        "adapter_source_sha256": sha256(environment_path),
        "environment_runtime_fingerprint": runtime_fingerprint,
        "frames": len(expected),
        "method": "exact PyBullet forward-kinematics pose difference at recorded state",
        "translation_mae_m": translation_mae.tolist(),
        "translation_rmse_m": np.sqrt(np.mean(np.square(error[:, :3]), axis=0)).tolist(),
        "rotation_mae_rad": rotation_mae.tolist(),
        "rotation_rmse_rad": np.sqrt(np.mean(np.square(error[:, 3:6]), axis=0)).tolist(),
        "gripper_max_abs_error": gripper_max_error,
        "correlation_by_pose_dimension": correlations,
        "pass_thresholds": {
            "max_translation_mae_m": 0.0025,
            "max_rotation_mae_rad": 0.003,
            "min_translation_correlation": 0.99,
            "gripper_max_abs_error": 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("openpi native-action adapter round-trip failed")


if __name__ == "__main__":
    main()
