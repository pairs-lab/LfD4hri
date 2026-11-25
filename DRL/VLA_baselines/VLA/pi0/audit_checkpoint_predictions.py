#!/usr/bin/env python3
"""Measure openpi checkpoint imitation error on exact audited source frames.

This is deliberately an offline audit: the policy sees only the two RGB
images, wrapped arm joints, gripper closure and language prompt.  Simulator
poses are never sent to the server.  PyBullet is used locally only to apply the
same joint-target-to-canonical-action adapter used during benchmark rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure
import pybullet as p
from openpi_client import image_tools, websocket_client_policy

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "VLA/audits/pi0_cam_v4_ckpt4999/offline_imitation.json",
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


def canonical_from_joint_target(
    robot: int,
    current: np.ndarray,
    target: np.ndarray,
    gripper_closure: float,
) -> np.ndarray:
    """Apply the benchmark's exact FK pose-difference action adapter.

    This intentionally mirrors ``ManipulationEnv.joint_targets_to_canonical``.
    The previous audit used a first-order Jacobian approximation, so its
    reported imitation error did not describe the action actually executed by
    the benchmark.
    """
    spec = ROBOT_SPECS["ur5e"]
    current = wrap(np.asarray(current, dtype=np.float64))
    for joint, value in zip(spec.arm_joint_ids, current, strict=True):
        p.resetJointState(robot, joint, float(value))
    current_link = p.getLinkState(robot, spec.mount_link, computeForwardKinematics=True)
    current_position = np.asarray(current_link[4], dtype=np.float64)
    current_orientation = current_link[5]
    for joint, value in zip(spec.arm_joint_ids, target, strict=True):
        p.resetJointState(robot, joint, float(value))
    target_link = p.getLinkState(robot, spec.mount_link, computeForwardKinematics=True)
    target_position = np.asarray(target_link[4], dtype=np.float64)
    target_orientation = target_link[5]
    inverse_current = p.invertTransform((0.0, 0.0, 0.0), current_orientation)[1]
    relative_orientation = p.multiplyTransforms(
        (0.0, 0.0, 0.0),
        inverse_current,
        (0.0, 0.0, 0.0),
        target_orientation,
    )[1]
    rotation = np.asarray(p.getEulerFromQuaternion(relative_orientation), dtype=np.float64)
    rotation = wrap(rotation)
    return np.concatenate(
        (
            np.clip(target_position - current_position, -0.05, 0.05),
            np.clip(rotation, -0.25, 0.25),
            np.asarray((2.0 * float(gripper_closure) - 1.0,)),
        )
    )


def aggregate(rows: list[dict]) -> dict:
    canonical_error = np.asarray([row["canonical_error"] for row in rows])
    predicted = np.asarray([row["predicted_canonical"] for row in rows])
    expected = np.asarray([row["expected_canonical"] for row in rows])
    native_error = np.asarray([row["native_error"] for row in rows])
    pred_translation = predicted[:, :3]
    expected_translation = expected[:, :3]
    denom = np.linalg.norm(pred_translation, axis=1) * np.linalg.norm(
        expected_translation, axis=1
    )
    cosine = np.divide(
        np.sum(pred_translation * expected_translation, axis=1),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-8,
    )
    return {
        "samples": len(rows),
        "canonical_mae": np.mean(np.abs(canonical_error), axis=0).tolist(),
        "native_absolute_joint_mae_rad": np.mean(
            np.abs(native_error[:, :6]), axis=0
        ).tolist(),
        "native_gripper_closure_mae": float(np.mean(np.abs(native_error[:, 6]))),
        "translation_rmse_m": float(
            np.sqrt(np.mean(np.square(canonical_error[:, :3])))
        ),
        "translation_cosine_mean": float(np.mean(cosine)),
        "translation_cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "gripper_sign_accuracy": float(
            np.mean(np.sign(predicted[:, 6]) == np.sign(expected[:, 6]))
        ),
        "translation_clip_fraction": float(
            np.mean(np.any(np.abs(predicted[:, :3]) >= 0.049999, axis=1))
        ),
    }


def main() -> None:
    args = parse_args()
    if args.episodes_per_task < 1 or args.draws < 1:
        raise ValueError("episodes-per-task and draws must be positive")
    index = json.loads((args.source / "index.json").read_text())
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in index:
        by_task[item["task"]["name"]].append(item)

    selected: list[dict] = []
    for task in ("touch", "pick", "move", "pick_place"):
        episodes = by_task[task]
        episode_indices = np.linspace(
            0, len(episodes) - 1, min(args.episodes_per_task, len(episodes)), dtype=int
        )
        for episode_index in episode_indices:
            selected.append(episodes[int(episode_index)])

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    spec = ROBOT_SPECS["ur5e"]
    bullet_client = p.connect(p.DIRECT)
    robot = p.loadURDF(str(spec.urdf), (0, 0, 0), useFixedBase=True)
    rows: list[dict] = []
    try:
        for item in selected:
            with np.load(args.source / item["file"]) as episode:
                length = len(episode["joint_positions"])
                frame_indices = np.unique(
                    np.rint(np.linspace(0, length - 1, 4)).astype(int)
                )
                for stage_index, frame in enumerate(frame_indices):
                    current = wrap(episode["joint_positions"][frame]).astype(np.float32)
                    request = {
                        "base_rgb": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(episode["front_rgb"][frame], 224, 224)
                        ),
                        "wrist_rgb": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(episode["wrist_rgb"][frame], 224, 224)
                        ),
                        "joints": current,
                        "gripper": np.asarray(
                            (episode["gripper_closure"][frame],), dtype=np.float32
                        ),
                        "prompt": str(episode["prompt"].item()),
                    }
                    expected = np.asarray(episode["canonical_actions"][frame], dtype=np.float64)
                    expected_native = np.concatenate(
                        (
                            wrap(episode["next_joint_targets"][frame]),
                            np.asarray((native_gripper_closure(episode)[frame],)),
                        )
                    ).astype(np.float64)
                    for draw in range(args.draws):
                        noise_seed = int.from_bytes(
                            hashlib.sha256(
                                f"{item['file']}:{int(frame)}:{draw}".encode("utf-8")
                            ).digest()[:8],
                            byteorder="little",
                        )
                        request["__policy_noise__"] = np.random.default_rng(
                            np.random.SeedSequence((args.seed, noise_seed))
                        ).standard_normal((10, 32), dtype=np.float32)
                        actions = np.asarray(client.infer(request)["actions"], dtype=np.float64)
                        if actions.ndim != 2 or actions.shape[1] < 7:
                            raise ValueError(f"Policy returned invalid native action shape {actions.shape}")
                        predicted = canonical_from_joint_target(
                            robot, current, actions[0, :6], actions[0, 6]
                        )
                        predicted_native = np.concatenate(
                            (wrap(actions[0, :6]), np.asarray((actions[0, 6],)))
                        )
                        native_error = predicted_native - expected_native
                        native_error[:6] = wrap(native_error[:6])
                        rows.append(
                            {
                                "task": item["task"]["name"],
                                "file": item["file"],
                                "frame": int(frame),
                                "stage_quartile": int(stage_index),
                                "draw": draw,
                                "expected_canonical": expected.tolist(),
                                "predicted_canonical": predicted.tolist(),
                                "canonical_error": (predicted - expected).tolist(),
                                "expected_native": expected_native.tolist(),
                                "predicted_native": predicted_native.tolist(),
                                "native_error": native_error.tolist(),
                            }
                        )
    finally:
        p.disconnect(bullet_client)

    report = {
        "source": str(args.source.resolve()),
        "source_index_sha256": sha256(args.source / "index.json"),
        "source_manifest_sha256": sha256(args.source / "dataset_manifest.json"),
        "policy_inputs": ["front_rgb", "wrist_rgb", "joint_positions", "gripper", "text"],
        "privileged_pose_sent_to_policy": False,
        "action_adapter": "exact FK pose difference (benchmark-equivalent)",
        "sampling": {
            "episodes_per_task": args.episodes_per_task,
            "quartiles_per_episode": 4,
            "draws_per_frame": args.draws,
            "diffusion_noise_seed": args.seed,
            "diffusion_noise_shape": [10, 32],
        },
        "overall": aggregate(rows),
        "by_task": {
            task: aggregate([row for row in rows if row["task"] == task])
            for task in ("touch", "pick", "move", "pick_place")
        },
        "by_stage_quartile": {
            str(stage): aggregate(
                [row for row in rows if row["stage_quartile"] == stage]
            )
            for stage in range(4)
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
