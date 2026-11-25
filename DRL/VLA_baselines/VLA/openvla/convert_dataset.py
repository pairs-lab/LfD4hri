#!/usr/bin/env python3
"""Convert the audited shared demonstrations into a compact OpenVLA dataset.

OpenVLA consumes one RGB image, a language instruction and a normalized 7D
EEF action.  This converter keeps the shared front view and canonical first six
action dimensions, while expressing the gripper in OpenVLA's native convention
(1=open, 0=closed).  It refuses stale/failed source audits and verifies every
written PNG before marking the conversion usable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-key",
        default=None,
        help="OpenVLA normalization key; defaults to the immutable source directory name",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    dataset_key = args.dataset_key or args.source.name
    audit = json.loads(args.audit_report.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, args.source)
    if not audit.get("passed"):
        raise RuntimeError(f"Refusing failed source audit: {args.audit_report}")
    if (
        audit.get("dataset") != str(args.source.resolve())
        or audit.get("index_sha256") != sha256(args.source / "index.json")
        or audit.get("manifest_sha256") != sha256(args.source / "dataset_manifest.json")
    ):
        raise RuntimeError("Source audit path/hash does not match the current dataset")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite converted dataset: {args.output}")

    index = json.loads((args.source / "index.json").read_text())
    if any(not item.get("success") for item in index):
        raise RuntimeError("Source contains an unsuccessful demonstration")
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True)
    samples: list[dict] = []
    native_actions: list[np.ndarray] = []
    max_pixel_error = 0
    global_index = 0
    for episode_index, item in enumerate(index):
        with np.load(args.source / item["file"]) as episode:
            prompt = str(episode["prompt"].item())
            # NPZ members are compressed independently.  Materialize each
            # trajectory member once: indexing ``episode["front_rgb"]`` (or
            # actions) inside the frame loop would decompress the complete
            # member again for every frame.
            front_rgb = episode["front_rgb"].astype(np.uint8, copy=False)
            canonical_actions = episode["canonical_actions"].astype(np.float32, copy=False)
            length = len(canonical_actions)
            for frame_index in range(length):
                relative_image = Path("images") / f"{global_index:06d}.png"
                image_path = args.output / relative_image
                source_rgb = front_rgb[frame_index]
                Image.fromarray(source_rgb, mode="RGB").save(image_path, format="PNG", compress_level=6)
                decoded = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                max_pixel_error = max(
                    max_pixel_error,
                    int(np.max(np.abs(decoded.astype(np.int16) - source_rgb.astype(np.int16)))),
                )
                canonical = canonical_actions[frame_index]
                native = canonical.copy()
                native[6] = (1.0 - canonical[6]) * 0.5  # OpenVLA: 1=open, 0=closed.
                native_actions.append(native)
                samples.append(
                    {
                        "index": global_index,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "image": relative_image.as_posix(),
                        "prompt": prompt,
                        "action": native.tolist(),
                    }
                )
                global_index += 1

    actions = np.asarray(native_actions, dtype=np.float32)
    mask = [True] * 6 + [False]
    stats = {
        dataset_key: {
            "action": {
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
                "min": actions.min(axis=0).tolist(),
                "q01": np.quantile(actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions, 0.99, axis=0).tolist(),
                "mask": mask,
            },
            "num_transitions": len(samples),
            "num_trajectories": len(index),
        }
    }
    (args.output / "samples.json").write_text(json.dumps(samples, separators=(",", ":")) + "\n")
    (args.output / "dataset_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    manifest = {
        "passed": max_pixel_error == 0,
        "source": str(args.source.resolve()),
        "source_audit": str(args.audit_report.resolve()),
        "source_index_sha256": audit["index_sha256"],
        "source_manifest_sha256": audit["manifest_sha256"],
        "episodes": len(index),
        "frames": len(samples),
        "front_rgb_max_pixel_error": max_pixel_error,
        "task_input": "natural-language prompt",
        "action_contract": "canonical delta EEF first 6 + OpenVLA gripper 1=open/0=closed",
        "normalization": "q01/q99 to [-1,1] for first 6; gripper unnormalized",
        "dataset_key": dataset_key,
    }
    (args.output / "conversion_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
