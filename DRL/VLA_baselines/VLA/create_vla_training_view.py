#!/usr/bin/env python3
"""Derive a one-shot-success VLA training view from strict ten-frame demos.

The source trajectories retain ten consecutive terminal-success transitions
to prove stable task completion. For supervised VLA training those near-static
tails overweight stopping behavior. This immutable derived view keeps every
transition through the first successful action, preserving the scene, cameras,
prompts, actions and all pre-success behavior byte-for-byte at array level.
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
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite derived dataset: {output}")
    audit = json.loads(args.source_audit.read_text())
    source_index_path = source / "index.json"
    source_manifest_path = source / "dataset_manifest.json"
    index_hash = sha256(source_index_path)
    manifest_hash = sha256(source_manifest_path)
    if (
        not audit.get("passed")
        or audit.get("failures")
        or Path(audit.get("dataset", "")).resolve() != source
        or audit.get("index_sha256") != index_hash
        or audit.get("manifest_sha256") != manifest_hash
        or not audit.get("all_oracle_success")
    ):
        raise RuntimeError("Strict source dataset/audit is failed or stale")

    source_index = json.loads(source_index_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, source)
    output.mkdir(parents=True)
    derived_index: list[dict] = []
    source_frames = 0
    output_frames = 0
    removed_terminal_frames = 0
    for episode_number, item in enumerate(source_index, start=1):
        source_path = source / item["file"]
        with np.load(source_path) as episode:
            original_steps = int(len(episode["canonical_actions"]))
            source_frames += original_steps
            raw_success = episode["next_raw_success"].astype(bool)
            successful = np.flatnonzero(raw_success)
            if not len(successful):
                raise RuntimeError(f"No successful transition in {source_path}")
            cutoff = int(successful[0]) + 1
            if int(episode["next_success_streak"][cutoff - 1]) != 1:
                raise RuntimeError(f"First-success alignment is invalid in {source_path}")
            arrays = {}
            for key in episode.files:
                value = episode[key]
                if key != "overview_reset_robot_mask" and value.ndim >= 1 and value.shape[0] == original_steps:
                    arrays[key] = value[:cutoff]
                else:
                    arrays[key] = value
            np.savez_compressed(output / item["file"], **arrays)

        derived = dict(item)
        derived["steps"] = cutoff
        derived["success"] = True
        derived["source_steps"] = original_steps
        for key in ("camera_only_derivative_of", "camera_derivative_source_sha256", "camera_derivative_output_sha256"):
            if key in derived:
                derived["parent_"+key] = derived.pop(key)
        derived["removed_terminal_success_steps"] = original_steps - cutoff
        derived["success_contract"] = {
            **item["success_contract"],
            "required_consecutive_steps": 1,
            "derivation": "kept through first raw-success transition",
        }
        derived_index.append(derived)
        output_frames += cutoff
        removed_terminal_frames += original_steps - cutoff
        print(
            f"{episode_number}/{len(source_index)} {item['file']}: "
            f"{original_steps} -> {cutoff}",
            flush=True,
        )

    (output / "index.json").write_text(json.dumps(derived_index, indent=2) + "\n")
    derived_manifest = {
        **source_manifest,
        "required_success_steps": 1,
        "index_sha256": sha256(output / "index.json"),
        "training_view": "vla_one_shot_success",
        "derived_from": str(source),
        "source_index_sha256": index_hash,
        "source_manifest_sha256": manifest_hash,
        "source_audit": str(args.source_audit.resolve()),
        "source_episodes": len(source_index),
        "source_frames": source_frames,
        "frames": output_frames,
        "removed_terminal_success_frames": removed_terminal_frames,
        "derivation": (
            "Every per-transition array is sliced after the first raw-success frame; "
            "all preceding sensor/action values are unchanged"
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(derived_manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "episodes": len(derived_index),
                "source_frames": source_frames,
                "frames": output_frames,
                "removed_terminal_success_frames": removed_terminal_frames,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
