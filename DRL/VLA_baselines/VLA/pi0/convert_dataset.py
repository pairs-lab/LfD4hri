#!/usr/bin/env python3
"""Convert successful PyBullet demos to openpi's pinned LeRobot v2 format.

The openpi UR5 recipe predicts the next six absolute joint targets and an
absolute gripper-closure value.  This is intentionally a separate dataset from
the canonical delta-EEF SmolVLA dataset: each model receives its native action
representation, while both are executed through the same simulator action
adapter during evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure, copy_requested_gripper_field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLA.common.provenance import build_directory_fingerprint, sha256_file


CONVERSION_MANIFEST_NAME = "conversion_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/pybullet_ur5e_robotiq_openpi")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def wrap_joints(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def main() -> None:
    # This import deliberately targets the LeRobot revision pinned by openpi,
    # not the newer LeRobot used for SmolVLA.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    index = json.loads((args.source / "index.json").read_text())
    audit = json.loads(args.audit_report.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, args.source)
    if not audit.get("passed"):
        raise RuntimeError(f"Refusing unaudited data: {args.audit_report} did not pass")
    if (
        audit.get("dataset") != str(args.source.resolve())
        or audit.get("index_sha256") != sha256_file(args.source / "index.json")
        or audit.get("manifest_sha256") != sha256_file(args.source / "dataset_manifest.json")
    ):
        raise RuntimeError("Audit report does not belong to the current immutable source metadata")
    failed = [item["file"] for item in index if not item.get("success")]
    if failed:
        raise RuntimeError(f"Refusing to train on failed demonstrations: {failed[:10]}")
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}. Choose a new path so prior data is preserved."
        )

    features = {
        "image": {
            "dtype": "image",
            "shape": (384, 384, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (384, 384, 3),
            "names": ["height", "width", "channel"],
        },
        "joints": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"q{i}" for i in range(6)],
        },
        "gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["closure"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": [*[f"next_q{i}" for i in range(6)], "next_gripper_closure"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="ur5e_robotiq_2f85_pybullet",
        fps=args.fps,
        features=features,
        root=args.output,
        use_videos=True,
        image_writer_threads=4,
    )
    frames = 0
    for episode_number, item in enumerate(index):
        path = args.source / item["file"]
        with np.load(path) as archive:
            episode = {key: archive[key] for key in (
                "prompt", "canonical_actions", "joint_positions", "gripper_closure",
                "next_joint_targets", "next_gripper_closure", "front_rgb", "wrist_rgb",
            )}
            prompt = str(episode["prompt"].item())
            copy_requested_gripper_field(archive, episode)
            length = int(episode["canonical_actions"].shape[0])
            required = (
                "joint_positions",
                "gripper_closure",
                "next_joint_targets",
                "next_gripper_closure",
                "front_rgb",
                "wrist_rgb",
            )
            if any(len(episode[key]) != length for key in required):
                raise ValueError(f"Inconsistent array length in {path}")
            for step in range(length):
                action = np.concatenate(
                    (
                        wrap_joints(episode["next_joint_targets"][step]),
                        np.asarray((native_gripper_closure(episode)[step],), dtype=np.float32),
                    )
                )
                dataset.add_frame(
                    {
                        "image": episode["front_rgb"][step],
                        "wrist_image": episode["wrist_rgb"][step],
                        "joints": wrap_joints(episode["joint_positions"][step]),
                        "gripper": np.asarray(
                            (episode["gripper_closure"][step],), dtype=np.float32
                        ),
                        "actions": action,
                        "task": prompt,
                    }
                )
                frames += 1
        dataset.save_episode()
        print(f"episode {episode_number + 1}/{len(index)}: {path.name}", flush=True)
    # The manifest cannot include its own digest.  Every actual LeRobot
    # payload/metadata file is content-hashed once and the manifest path is the
    # sole explicit exclusion.  Downstream audits can cheaply reject drift by
    # checking the recorded path/size/mtime layout before decoding anything.
    converted_fingerprint = build_directory_fingerprint(
        args.output,
        exclude_paths=(CONVERSION_MANIFEST_NAME,),
    )
    (args.output / CONVERSION_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 2,
                "source": str(args.source.resolve()),
                "source_index_sha256": audit["index_sha256"],
                "source_manifest_sha256": audit["manifest_sha256"],
                "audit_report": str(args.audit_report.resolve()),
                "audit_report_sha256": sha256_file(args.audit_report),
                "converted": str(args.output.resolve()),
                "converted_fingerprint": converted_fingerprint,
                "episodes": len(index),
                "frames": frames,
                "state": "6 wrapped absolute UR5e joints + Robotiq closure",
                "action": "next wrapped absolute UR5e joints + Robotiq closure",
                "camera_mapping": {"front": "base_rgb", "wrist": "wrist_rgb"},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(index)} episodes / {frames} frames to {args.output}")


if __name__ == "__main__":
    main()
