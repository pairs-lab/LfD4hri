#!/usr/bin/env python3
"""Convert successful PyBullet demonstrations to a local LeRobot dataset.

Run this module with the isolated LeRobot Python environment.  The conversion
keeps both RGB views, the six UR5e joints plus gripper closure, the common 7D
delta-EEF action, and the natural-language task instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/pybullet_ur5e_robotiq")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    index = json.loads((args.source / "index.json").read_text())
    audit = json.loads(args.audit_report.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, args.source)
    if not audit.get("passed"):
        raise RuntimeError(f"Refusing unaudited data: {args.audit_report} did not pass")
    if (
        audit.get("dataset") != str(args.source.resolve())
        or audit.get("index_sha256") != sha256(args.source / "index.json")
        or audit.get("manifest_sha256") != sha256(args.source / "dataset_manifest.json")
    ):
        raise RuntimeError("Audit report does not belong to the current immutable source metadata")
    failed = [item["file"] for item in index if not item.get("success")]
    if failed:
        raise RuntimeError(f"Refusing to train on failed demonstrations: {failed[:10]}")
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}. Choose a new path so a prior dataset is not overwritten."
        )

    features = {
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["q0", "q1", "q2", "q3", "q4", "q5", "gripper_closure"],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (384, 384, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (384, 384, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="ur5e_robotiq_2f85_pybullet",
        fps=args.fps,
        features=features,
        root=args.output,
    )
    frame_count = 0
    for episode_number, item in enumerate(index):
        path = args.source / item["file"]
        with np.load(path) as episode:
            prompt = str(episode["prompt"].item())
            # Materialize every compressed NPZ member once per episode.  In
            # particular, derive the length from the cached action array so it
            # is not decompressed once for ``shape`` and again below.
            arrays = {
                key: episode[key]
                for key in (
                    "canonical_actions",
                    "joint_positions",
                    "gripper_closure",
                    "front_rgb",
                    "wrist_rgb",
                )
            }
            length = int(arrays["canonical_actions"].shape[0])
            if any(len(value) != length for value in arrays.values()):
                raise ValueError(f"Inconsistent array length in {path}")
            for step in range(length):
                state = np.concatenate(
                    (
                        arrays["joint_positions"][step].astype(np.float32),
                        np.asarray((arrays["gripper_closure"][step],), dtype=np.float32),
                    )
                )
                dataset.add_frame(
                    {
                        "action": arrays["canonical_actions"][step].astype(np.float32),
                        "observation.state": state,
                        "observation.images.front": arrays["front_rgb"][step],
                        "observation.images.wrist": arrays["wrist_rgb"][step],
                        "task": prompt,
                    }
                )
                frame_count += 1
        dataset.save_episode()
        print(f"episode {episode_number + 1}/{len(index)}: {path.name}", flush=True)
    dataset.finalize()
    (args.output / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "source_index_sha256": audit["index_sha256"],
                "source_manifest_sha256": audit["manifest_sha256"],
                "audit_report": str(args.audit_report.resolve()),
                "episodes": len(index),
                "frames": frame_count,
                "state": "6 absolute UR5e joints + Robotiq closure",
                "action": "shared 7D delta-EEF canonical action",
                "camera_mapping": {"front": "camera1", "wrist": "camera2"},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(index)} episodes / {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
