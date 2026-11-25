#!/usr/bin/env python3
"""Round-trip audit for the openpi-pinned LeRobot v2.1 dataset.

The numeric pass checks every frame directly in Parquet.  The image pass
decodes every embedded RGB payload, and a stratified start/middle/end pass for
every episode exercises the exact LeRobot loader consumed by openpi.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure, copy_requested_gripper_field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLA.common.provenance import sha256_file, verify_directory_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wrap_joints(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def decoded_rgb(value) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return np.rint(np.clip(array.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)


def main() -> int:
    import pyarrow.parquet as pq
    from PIL import Image
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    source_root = args.source.resolve()
    dataset_root = args.dataset_root.resolve()
    source_index_path = source_root / "index.json"
    source_manifest_path = source_root / "dataset_manifest.json"
    conversion_path = dataset_root / "conversion_manifest.json"
    source_index_hash = sha256_file(source_index_path)
    source_manifest_hash = sha256_file(source_manifest_path)
    conversion_manifest_hash = sha256_file(conversion_path)
    source_index = json.loads(source_index_path.read_text())
    conversion = json.loads(conversion_path.read_text())
    converted_fingerprint = conversion.get("converted_fingerprint")
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    tasks = {
        int(row["task_index"]): row["task"]
        for row in (
            json.loads(line)
            for line in (dataset_root / "meta" / "tasks.jsonl").read_text().splitlines()
        )
    }
    failures: list[str] = []
    expected_frames = sum(int(item["steps"]) for item in source_index)
    expected_loader_samples = sum(
        len({0, steps // 2, steps - 1}) if steps > 0 else 0
        for steps in (int(item["steps"]) for item in source_index)
    )
    if conversion.get("version") != 2:
        failures.append("conversion manifest does not use provenance schema version 2")
    if Path(conversion.get("source", "")).resolve() != source_root:
        failures.append("conversion manifest names a different source path")
    if Path(conversion.get("converted", "")).resolve() != dataset_root:
        failures.append("conversion manifest names a different converted dataset path")
    if conversion.get("source_index_sha256") != source_index_hash:
        failures.append("conversion manifest has a stale source index hash")
    if conversion.get("source_manifest_sha256") != source_manifest_hash:
        failures.append("conversion manifest has a stale source manifest hash")
    provenance_verification: dict | None = None
    if not isinstance(converted_fingerprint, dict):
        failures.append("conversion manifest has no full converted-data fingerprint")
    else:
        try:
            provenance_verification = verify_directory_fingerprint(
                dataset_root,
                converted_fingerprint,
                full=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"converted-data fingerprint mismatch: {exc}")
    if conversion.get("episodes") != len(source_index) or info.get("total_episodes") != len(source_index):
        failures.append("episode count mismatch")
    if conversion.get("frames") != expected_frames or info.get("total_frames") != expected_frames:
        failures.append("frame count mismatch")
    if info.get("fps") != 10:
        failures.append(f"unexpected fps {info.get('fps')}")
    chunks_size = int(info.get("chunks_size", 0))
    data_path_template = info.get("data_path")
    if chunks_size < 1 or not isinstance(data_path_template, str):
        raise RuntimeError("LeRobot metadata has no valid chunks_size/data_path contract")

    max_joint_error = 0.0
    max_gripper_error = 0.0
    max_action_error = 0.0
    max_front_pixel_error = 0
    max_wrist_pixel_error = 0
    prompt_mismatches = 0
    index_mismatches = 0
    timestamp_max_abs_error = 0.0
    global_frame = 0
    source_frame_offset = 0
    loader_samples = 0
    loader_joint_error = 0.0
    loader_gripper_error = 0.0
    loader_action_error = 0.0
    loader_front_pixel_error = 0
    loader_wrist_pixel_error = 0
    loader_prompt_mismatches = 0
    dataset = LeRobotDataset(args.repo_id, root=dataset_root)
    source_keys = (
        "prompt",
        "canonical_actions",
        "joint_positions",
        "gripper_closure",
        "next_joint_targets",
        "next_gripper_closure",
        "front_rgb",
        "wrist_rgb",
    )
    parquet_keys = (
        "joints",
        "gripper",
        "actions",
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
        "image",
        "wrist_image",
    )
    for episode_index, item in enumerate(source_index):
        episode_start = source_frame_offset
        source_frame_offset += int(item["steps"])
        parquet_path = dataset_root / data_path_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        if not parquet_path.is_file():
            failures.append(f"missing {parquet_path.name}")
            continue
        table = pq.read_table(parquet_path)
        missing_columns = [key for key in parquet_keys if key not in table.column_names]
        if missing_columns:
            failures.append(f"{parquet_path.name}: missing columns {missing_columns}")
            continue
        # Resolve each Parquet column once.  In particular, repeatedly looking
        # up embedded image columns can trigger avoidable chunk traversal.
        columns = {key: table[key] for key in parquet_keys}
        with np.load(source_root / item["file"]) as archive:
            # NpzFile.__getitem__ decompresses an array on every access.  Load
            # each required array exactly once, then reuse it for both the
            # direct Parquet and exact LeRobot-loader checks below.
            source = {key: archive[key] for key in source_keys}
            copy_requested_gripper_field(archive, source)
            length = len(source["canonical_actions"])
            if int(item["steps"]) != length:
                failures.append(
                    f"{item['file']}: source index steps {item['steps']} != archive length {length}"
                )
            if len(table) != length:
                failures.append(f"{item['file']}: frame count {len(table)} != {length}")
                continue
            if length < 1:
                failures.append(f"{item['file']}: empty episode")
                continue
            expected_joints = wrap_joints(source["joint_positions"])
            expected_gripper = source["gripper_closure"].astype(np.float32)
            expected_action = np.concatenate(
                (
                    wrap_joints(source["next_joint_targets"]),
                    native_gripper_closure(source).astype(np.float32)[:, None],
                ),
                axis=1,
            )
            joints = np.asarray(columns["joints"].to_pylist(), dtype=np.float32)
            gripper = np.asarray(columns["gripper"].to_numpy(), dtype=np.float32)
            actions = np.asarray(columns["actions"].to_pylist(), dtype=np.float32)
            max_joint_error = max(max_joint_error, float(np.max(np.abs(joints - expected_joints))))
            max_gripper_error = max(
                max_gripper_error,
                float(np.max(np.abs(gripper - expected_gripper))),
            )
            max_action_error = max(
                max_action_error,
                float(np.max(np.abs(actions - expected_action))),
            )
            episode_indices = np.asarray(columns["episode_index"].to_numpy())
            frame_indices = np.asarray(columns["frame_index"].to_numpy())
            indices = np.asarray(columns["index"].to_numpy())
            timestamps = np.asarray(columns["timestamp"].to_numpy(), dtype=np.float64)
            task_indices = np.asarray(columns["task_index"].to_numpy())
            expected_prompt = str(source["prompt"].item())
            for frame in range(length):
                with Image.open(io.BytesIO(columns["image"][frame].as_py()["bytes"])) as image:
                    front = np.asarray(image.convert("RGB"))
                with Image.open(
                    io.BytesIO(columns["wrist_image"][frame].as_py()["bytes"])
                ) as image:
                    wrist = np.asarray(image.convert("RGB"))
                max_front_pixel_error = max(
                    max_front_pixel_error,
                    int(
                        np.max(
                            np.abs(
                                front.astype(np.int16)
                                - source["front_rgb"][frame].astype(np.int16)
                            )
                        )
                    ),
                )
                max_wrist_pixel_error = max(
                    max_wrist_pixel_error,
                    int(
                        np.max(
                            np.abs(
                                wrist.astype(np.int16)
                                - source["wrist_rgb"][frame].astype(np.int16)
                            )
                        )
                    ),
                )
                if tasks.get(int(task_indices[frame])) != expected_prompt:
                    prompt_mismatches += 1
                if (
                    int(episode_indices[frame]) != episode_index
                    or int(frame_indices[frame]) != frame
                    or int(indices[frame]) != episode_start + frame
                ):
                    index_mismatches += 1
                timestamp_max_abs_error = max(
                    timestamp_max_abs_error,
                    abs(float(timestamps[frame]) - frame / 10.0),
                )
                global_frame += 1

            # Exercise the exact OpenPI/LeRobot loader while this episode's
            # source arrays are still cached.  This removes the former second
            # NPZ open/decompression pass and keeps memory bounded to one
            # episode rather than caching RGB from the whole dataset.
            for frame in sorted({0, length // 2, length - 1}):
                converted = dataset[episode_start + frame]
                loader_samples += 1
                loader_joint_error = max(
                    loader_joint_error,
                    float(
                        np.max(
                            np.abs(converted["joints"].numpy() - expected_joints[frame])
                        )
                    ),
                )
                loader_gripper_error = max(
                    loader_gripper_error,
                    abs(float(converted["gripper"]) - float(expected_gripper[frame])),
                )
                loader_action_error = max(
                    loader_action_error,
                    float(
                        np.max(
                            np.abs(converted["actions"].numpy() - expected_action[frame])
                        )
                    ),
                )
                loader_front_pixel_error = max(
                    loader_front_pixel_error,
                    int(
                        np.max(
                            np.abs(
                                decoded_rgb(converted["image"]).astype(np.int16)
                                - source["front_rgb"][frame].astype(np.int16)
                            )
                        )
                    ),
                )
                loader_wrist_pixel_error = max(
                    loader_wrist_pixel_error,
                    int(
                        np.max(
                            np.abs(
                                decoded_rgb(converted["wrist_image"]).astype(np.int16)
                                - source["wrist_rgb"][frame].astype(np.int16)
                            )
                        )
                    ),
                )
                if converted["task"] != expected_prompt:
                    loader_prompt_mismatches += 1

    if len(dataset) != expected_frames or dataset.num_episodes != len(source_index):
        failures.append("LeRobot loader count mismatch")
    if loader_samples != expected_loader_samples:
        failures.append(
            f"LeRobot loader sample coverage {loader_samples} != {expected_loader_samples}"
        )
    if max_joint_error > 1e-7 or max_gripper_error > 1e-7 or max_action_error > 1e-7:
        failures.append("Parquet numeric round-trip mismatch")
    if max_front_pixel_error or max_wrist_pixel_error:
        failures.append("Parquet RGB round-trip is not lossless")
    if prompt_mismatches or index_mismatches or timestamp_max_abs_error > 1e-5:
        failures.append("Parquet metadata mismatch")
    if loader_joint_error > 1e-7 or loader_gripper_error > 1e-7 or loader_action_error > 1e-7:
        failures.append("LeRobot loader numeric mismatch")
    if loader_front_pixel_error or loader_wrist_pixel_error or loader_prompt_mismatches:
        failures.append("LeRobot loader image/prompt mismatch")

    # Catch a concurrent or accidental artifact mutation during the expensive
    # semantic audit without rereading every payload solely for hashing.
    if isinstance(converted_fingerprint, dict):
        try:
            provenance_verification = verify_directory_fingerprint(
                dataset_root,
                converted_fingerprint,
                full=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            message = f"converted-data fingerprint changed during audit: {exc}"
            if message not in failures:
                failures.append(message)
    if sha256_file(conversion_path) != conversion_manifest_hash:
        failures.append("conversion manifest changed during audit")

    report = {
        "passed": not failures,
        "failures": failures,
        "source": str(source_root),
        "source_index_sha256": source_index_hash,
        "source_manifest_sha256": source_manifest_hash,
        "converted": str(dataset_root),
        "repo_id": args.repo_id,
        "conversion_manifest_sha256": conversion_manifest_hash,
        "converted_fingerprint": converted_fingerprint,
        "converted_fingerprint_verification": provenance_verification,
        "episodes": len(source_index),
        "frames": global_frame,
        "joint_max_abs_error": max_joint_error,
        "gripper_max_abs_error": max_gripper_error,
        "action_max_abs_error": max_action_error,
        "front_rgb_max_pixel_error": max_front_pixel_error,
        "wrist_rgb_max_pixel_error": max_wrist_pixel_error,
        "prompt_mismatches": prompt_mismatches,
        "index_mismatches": index_mismatches,
        "timestamp_max_abs_error": timestamp_max_abs_error,
        "loader_sampling": "unique start/middle/end frame from every episode",
        "loader_expected_samples": expected_loader_samples,
        "loader_samples": loader_samples,
        "loader_joint_max_abs_error": loader_joint_error,
        "loader_gripper_max_abs_error": loader_gripper_error,
        "loader_action_max_abs_error": loader_action_error,
        "loader_front_rgb_max_pixel_error": loader_front_pixel_error,
        "loader_wrist_rgb_max_pixel_error": loader_wrist_pixel_error,
        "loader_prompt_mismatches": loader_prompt_mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
