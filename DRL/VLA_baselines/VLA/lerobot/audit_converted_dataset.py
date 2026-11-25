#!/usr/bin/env python3
"""Round-trip audit from source NPZ frames to the exact LeRobot training tensors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decoded_rgb(value) -> np.ndarray:
    array = value.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import pyarrow.parquet as pq
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    index = json.loads((args.source / "index.json").read_text())
    conversion = json.loads((args.dataset_root / "conversion_manifest.json").read_text())
    expected_frames = sum(int(item["steps"]) for item in index)
    failures: list[str] = []
    data_files = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No converted Parquet data under {args.dataset_root / 'data'}")
    table = pq.read_table(data_files)
    if len(table) != expected_frames or conversion.get("frames") != expected_frames:
        failures.append(
            f"frame count mismatch: source={expected_frames}, converted={len(table)}, "
            f"manifest={conversion.get('frames')}"
        )
    if conversion.get("episodes") != len(index):
        failures.append(
            f"episode count mismatch: source={len(index)}, manifest={conversion.get('episodes')}"
        )
    if conversion.get("source_index_sha256") != sha256(args.source / "index.json"):
        failures.append("conversion manifest has a stale source index hash")
    if conversion.get("source_manifest_sha256") != sha256(args.source / "dataset_manifest.json"):
        failures.append("conversion manifest has a stale source dataset-manifest hash")

    task_table = pq.read_table(args.dataset_root / "meta" / "tasks.parquet").to_pydict()
    prompts_by_index = dict(zip(task_table["task_index"], task_table["task"], strict=True))
    converted_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    converted_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    global_indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)

    max_state_error = 0.0
    max_action_error = 0.0
    max_front_pixel_error = 0
    max_wrist_pixel_error = 0
    global_frame = 0
    prompt_mismatches = 0
    index_mismatches = 0
    timestamp_max_abs_error = 0.0
    loader_sample_indices: set[int] = set()
    for episode_number, item in enumerate(index):
        with np.load(args.source / item["file"]) as archive:
            source = {key: archive[key] for key in (
                "prompt", "canonical_actions", "joint_positions", "gripper_closure",
                "front_rgb", "wrist_rgb",
            )}
            prompt = str(source["prompt"].item())
            length = len(source["canonical_actions"])
            loader_sample_indices.add(global_frame)
            loader_sample_indices.add(global_frame + length // 2)
            loader_sample_indices.add(global_frame + length - 1)
            for frame in range(length):
                expected_state = np.concatenate(
                    (
                        source["joint_positions"][frame].astype(np.float32),
                        np.asarray((source["gripper_closure"][frame],), dtype=np.float32),
                    )
                )
                max_state_error = max(
                    max_state_error,
                    float(np.max(np.abs(converted_states[global_frame] - expected_state))),
                )
                max_action_error = max(
                    max_action_error,
                    float(
                        np.max(
                            np.abs(
                                converted_actions[global_frame]
                                - source["canonical_actions"][frame].astype(np.float32)
                            )
                        )
                    ),
                )
                max_front_pixel_error = max(
                    max_front_pixel_error,
                    int(
                        np.max(
                            np.abs(
                                np.asarray(
                                    Image.open(
                                        io.BytesIO(
                                            table["observation.images.front"][global_frame].as_py()["bytes"]
                                        )
                                    ).convert("RGB")
                                ).astype(np.int16)
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
                                np.asarray(
                                    Image.open(
                                        io.BytesIO(
                                            table["observation.images.wrist"][global_frame].as_py()["bytes"]
                                        )
                                    ).convert("RGB")
                                ).astype(np.int16)
                                - source["wrist_rgb"][frame].astype(np.int16)
                            )
                        )
                    ),
                )
                if prompts_by_index[int(task_indices[global_frame])] != prompt:
                    prompt_mismatches += 1
                if (
                    episode_indices[global_frame] != episode_number
                    or frame_indices[global_frame] != frame
                    or global_indices[global_frame] != global_frame
                ):
                    index_mismatches += 1
                timestamp_max_abs_error = max(
                    timestamp_max_abs_error,
                    abs(float(timestamps[global_frame]) - frame / 10.0),
                )
                global_frame += 1

    # The all-frame pass above audits the exact Parquet values and lossless PNG
    # payloads.  This stratified pass additionally exercises LeRobotDataset's
    # image decoder/tensor conversion at the start, middle and end of every
    # episode—the same API consumed by the trainer.
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    loader_state_error = 0.0
    loader_action_error = 0.0
    loader_front_pixel_error = 0
    loader_wrist_pixel_error = 0
    for global_index in sorted(loader_sample_indices):
        converted = dataset[global_index]
        loader_state_error = max(
            loader_state_error,
            float(np.max(np.abs(converted["observation.state"].numpy() - converted_states[global_index]))),
        )
        loader_action_error = max(
            loader_action_error,
            float(np.max(np.abs(converted["action"].numpy() - converted_actions[global_index]))),
        )
        for key, parquet_key, accumulator in (
            ("observation.images.front", "observation.images.front", "front"),
            ("observation.images.wrist", "observation.images.wrist", "wrist"),
        ):
            raw = np.asarray(
                Image.open(io.BytesIO(table[parquet_key][global_index].as_py()["bytes"])).convert("RGB")
            )
            error = int(np.max(np.abs(decoded_rgb(converted[key]).astype(np.int16) - raw.astype(np.int16))))
            if accumulator == "front":
                loader_front_pixel_error = max(loader_front_pixel_error, error)
            else:
                loader_wrist_pixel_error = max(loader_wrist_pixel_error, error)

    if len(dataset) != expected_frames or dataset.num_episodes != len(index):
        failures.append(
            f"LeRobot API counts differ: frames={len(dataset)}, episodes={dataset.num_episodes}"
        )

    if max_state_error > 1e-7:
        failures.append(f"state max absolute error {max_state_error}")
    if max_action_error > 1e-7:
        failures.append(f"action max absolute error {max_action_error}")
    if max_front_pixel_error != 0 or max_wrist_pixel_error != 0:
        failures.append(
            f"RGB round-trip is not lossless: front={max_front_pixel_error}, wrist={max_wrist_pixel_error}"
        )
    if prompt_mismatches:
        failures.append(f"prompt mismatches: {prompt_mismatches}")
    if index_mismatches:
        failures.append(f"episode/frame index mismatches: {index_mismatches}")
    if timestamp_max_abs_error > 1e-5:
        failures.append(f"timestamp max absolute error: {timestamp_max_abs_error}")
    if loader_state_error > 1e-7 or loader_action_error > 1e-7:
        failures.append(
            f"LeRobot loader numeric mismatch: state={loader_state_error}, action={loader_action_error}"
        )
    if loader_front_pixel_error or loader_wrist_pixel_error:
        failures.append(
            "LeRobot loader RGB mismatch: "
            f"front={loader_front_pixel_error}, wrist={loader_wrist_pixel_error}"
        )

    report = {
        "passed": not failures,
        "failures": failures,
        "episodes": len(index),
        "frames": global_frame,
        "state_max_abs_error": max_state_error,
        "action_max_abs_error": max_action_error,
        "front_rgb_max_pixel_error": max_front_pixel_error,
        "wrist_rgb_max_pixel_error": max_wrist_pixel_error,
        "prompt_mismatches": prompt_mismatches,
        "index_mismatches": index_mismatches,
        "timestamp_max_abs_error": timestamp_max_abs_error,
        "lerobot_loader_samples": len(loader_sample_indices),
        "loader_state_max_abs_error": loader_state_error,
        "loader_action_max_abs_error": loader_action_error,
        "loader_front_rgb_max_pixel_error": loader_front_pixel_error,
        "loader_wrist_rgb_max_pixel_error": loader_wrist_pixel_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
