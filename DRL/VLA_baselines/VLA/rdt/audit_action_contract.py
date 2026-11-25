#!/usr/bin/env python3
"""Exhaustively verify RDT's absolute-joint action representation with UR5e FK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure
import pybullet as p

from VLA.common.environment import ROBOT_SPECS
from VLA.rdt.training_data import sha256, wrap_joints


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
    profile = json.loads((source/"dataset_manifest.json").read_text()).get("observation_profile")
    runtime_fingerprint = environment_runtime_fingerprint(profile)
    index = json.loads((source / "index.json").read_text())
    spec = ROBOT_SPECS["ur5e"]
    client = p.connect(p.DIRECT)
    robot = p.loadURDF(str(spec.urdf), (0, 0, 0), useFixedBase=True)
    expected_rows: list[np.ndarray] = []
    reconstructed_rows: list[np.ndarray] = []
    wrap_fk_position_errors: list[float] = []
    try:
        for number, item in enumerate(index, start=1):
            with np.load(source / item["file"], allow_pickle=False) as episode:
                iterator = zip(
                    episode["joint_positions"],
                    episode["next_joint_targets"],
                    native_gripper_closure(episode),
                    episode["canonical_actions"],
                    strict=True,
                )
                for current_raw, target_raw, closure, expected in iterator:
                    current = wrap_joints(current_raw).astype(np.float64)
                    target = wrap_joints(target_raw).astype(np.float64)
                    for joint, value in zip(spec.arm_joint_ids, current_raw, strict=True):
                        p.resetJointState(robot, joint, float(value))
                    raw_current_link = p.getLinkState(
                        robot, spec.mount_link, computeForwardKinematics=True
                    )
                    for joint, value in zip(spec.arm_joint_ids, current, strict=True):
                        p.resetJointState(robot, joint, float(value))
                    current_link = p.getLinkState(
                        robot, spec.mount_link, computeForwardKinematics=True
                    )
                    wrap_fk_position_errors.append(
                        float(
                            np.linalg.norm(
                                np.asarray(raw_current_link[4]) - np.asarray(current_link[4])
                            )
                        )
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
                    # RDT predicts gripper-open; the common native adapter
                    # consumes closure = 1-open and emits -1=open/+1=close.
                    gripper_open = 1.0 - float(closure)
                    reconstructed = np.concatenate(
                        (
                            np.clip(translation, -0.05, 0.05),
                            np.clip(rotation, -0.25, 0.25),
                            np.asarray((2.0 * (1.0 - gripper_open) - 1.0,)),
                        )
                    )
                    expected_rows.append(np.asarray(expected, dtype=np.float64))
                    reconstructed_rows.append(reconstructed)
            print(f"{number}/{len(index)} {item['file']}", flush=True)
    finally:
        p.disconnect(client)

    expected = np.asarray(expected_rows)
    reconstructed = np.asarray(reconstructed_rows)
    error = reconstructed - expected
    translation_mae = np.mean(np.abs(error[:, :3]), axis=0)
    rotation_mae = np.mean(np.abs(error[:, 3:6]), axis=0)
    gripper_max_error = float(np.max(np.abs(error[:, 6])))
    correlations: list[float | None] = []
    for dimension in range(6):
        if np.std(expected[:, dimension]) == 0 or np.std(reconstructed[:, dimension]) == 0:
            correlations.append(None)
        else:
            correlations.append(
                float(np.corrcoef(expected[:, dimension], reconstructed[:, dimension])[0, 1])
            )
    valid_translation_correlations = [
        value for value in correlations[:3] if value is not None
    ]
    wrap_fk_max = max(wrap_fk_position_errors, default=float("inf"))
    passed = bool(
        len(expected) == sum(int(item["steps"]) for item in index)
        and max(translation_mae) < 0.0025
        and max(rotation_mae) < 0.003
        and gripper_max_error == 0.0
        and valid_translation_correlations
        and min(valid_translation_correlations) > 0.99
        and wrap_fk_max < 1e-5
    )
    environment_path = PROJECT_ROOT / "VLA/common/environment.py"
    contract_path = PROJECT_ROOT / "VLA/rdt/training_data.py"
    verify_environment_runtime_fingerprint(runtime_fingerprint, profile)
    report = {
        "passed": passed,
        "model": "rdt",
        "source": str(source),
        "source_index_sha256": sha256(source / "index.json"),
        "source_manifest_sha256": sha256(source / "dataset_manifest.json"),
        "environment_sha256": sha256(environment_path),
        "environment_runtime_fingerprint": runtime_fingerprint,
        "training_contract_sha256": sha256(contract_path),
        "frames": len(expected),
        "joint_convention": "wrap_to_minus_pi_pi",
        "native_action": "absolute wrapped joints + absolute gripper-open",
        "method": "exact PyBullet UR5e FK pose difference at every recorded state",
        "wrapped_vs_raw_fk_position_max_error_m": wrap_fk_max,
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
            "wrapped_vs_raw_fk_position_max_error_m": 1e-5,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("RDT native action contract audit failed")


if __name__ == "__main__":
    main()
