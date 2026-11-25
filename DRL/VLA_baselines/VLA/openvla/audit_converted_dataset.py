#!/usr/bin/env python3
"""Exhaustive source-to-OpenVLA round-trip and statistics audit."""

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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    dataset = args.dataset_root.resolve()
    index_path = source / "index.json"
    source_manifest_path = source / "dataset_manifest.json"
    conversion_path = dataset / "conversion_audit.json"
    samples_path = dataset / "samples.json"
    statistics_path = dataset / "dataset_statistics.json"
    for path in (index_path, source_manifest_path, conversion_path, samples_path, statistics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_index = json.loads(index_path.read_text())
    conversion = json.loads(conversion_path.read_text())
    samples = json.loads(samples_path.read_text())
    statistics = json.loads(statistics_path.read_text())
    failures: list[str] = []
    index_hash = sha256(index_path)
    manifest_hash = sha256(source_manifest_path)
    expected_frames = sum(int(item["steps"]) for item in source_index)
    dataset_key = conversion.get("dataset_key")
    if (
        not conversion.get("passed")
        or Path(conversion.get("source", "")).resolve() != source
        or conversion.get("source_index_sha256") != index_hash
        or conversion.get("source_manifest_sha256") != manifest_hash
    ):
        failures.append("conversion manifest is failed or stale")
    if conversion.get("episodes") != len(source_index):
        failures.append("conversion episode count mismatch")
    if conversion.get("frames") != expected_frames or len(samples) != expected_frames:
        failures.append("conversion/sample frame count mismatch")
    if dataset_key not in statistics:
        failures.append("dataset statistics key is missing")

    action_max_abs_error = 0.0
    front_rgb_max_pixel_error = 0
    prompt_mismatches = 0
    index_mismatches = 0
    missing_images = 0
    expected_actions: list[np.ndarray] = []
    global_index = 0
    for episode_index, item in enumerate(source_index):
        with np.load(source / item["file"]) as episode:
            prompt = str(episode["prompt"].item())
            actions = episode["canonical_actions"].astype(np.float32).copy()
            # NPZ members are compressed independently.  Materialize the RGB
            # member once per episode; indexing ``episode['front_rgb']`` in
            # the inner loop would decompress the whole trajectory per frame.
            front_rgb = episode["front_rgb"].astype(np.uint8)
            actions[:, 6] = (1.0 - actions[:, 6]) * 0.5
            expected_actions.append(actions)
            for frame_index in range(len(actions)):
                if global_index >= len(samples):
                    failures.append("samples.json ended before source frames")
                    break
                sample = samples[global_index]
                if (
                    sample.get("index") != global_index
                    or sample.get("episode_index") != episode_index
                    or sample.get("frame_index") != frame_index
                ):
                    index_mismatches += 1
                if sample.get("prompt") != prompt:
                    prompt_mismatches += 1
                converted_action = np.asarray(sample.get("action", ()), dtype=np.float32)
                if converted_action.shape != (7,):
                    failures.append(f"sample {global_index}: invalid action shape {converted_action.shape}")
                else:
                    action_max_abs_error = max(
                        action_max_abs_error,
                        float(np.max(np.abs(converted_action - actions[frame_index]))),
                    )
                relative_image = Path(str(sample.get("image", "")))
                expected_relative = Path("images") / f"{global_index:06d}.png"
                if relative_image != expected_relative:
                    index_mismatches += 1
                image_path = dataset / relative_image
                if not image_path.is_file():
                    missing_images += 1
                else:
                    decoded = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                    source_rgb = front_rgb[frame_index]
                    if decoded.shape != source_rgb.shape:
                        failures.append(
                            f"sample {global_index}: RGB shape {decoded.shape} != {source_rgb.shape}"
                        )
                    else:
                        front_rgb_max_pixel_error = max(
                            front_rgb_max_pixel_error,
                            int(
                                np.max(
                                    np.abs(
                                        decoded.astype(np.int16) - source_rgb.astype(np.int16)
                                    )
                                )
                            ),
                        )
                global_index += 1

    expected_pngs = {f"{index:06d}.png" for index in range(expected_frames)}
    actual_pngs = {path.name for path in (dataset / "images").glob("*.png")}
    unexpected_image_files = sorted(actual_pngs - expected_pngs)
    absent_image_files = sorted(expected_pngs - actual_pngs)
    if unexpected_image_files or absent_image_files:
        failures.append(
            f"image file set mismatch: missing={len(absent_image_files)}, "
            f"unexpected={len(unexpected_image_files)}"
        )

    all_actions = np.concatenate(expected_actions, axis=0)
    exact_stats = {
        "mean": all_actions.mean(axis=0),
        "std": all_actions.std(axis=0),
        "max": all_actions.max(axis=0),
        "min": all_actions.min(axis=0),
        "q01": np.quantile(all_actions, 0.01, axis=0),
        "q99": np.quantile(all_actions, 0.99, axis=0),
    }
    statistics_max_abs_error = 0.0
    if dataset_key in statistics:
        stored_action = statistics[dataset_key].get("action", {})
        for key, expected in exact_stats.items():
            stored = np.asarray(stored_action.get(key, ()), dtype=np.float64)
            if stored.shape != (7,):
                failures.append(f"statistics {key}: invalid shape {stored.shape}")
            else:
                statistics_max_abs_error = max(
                    statistics_max_abs_error,
                    float(np.max(np.abs(stored - np.asarray(expected, dtype=np.float64)))),
                )
        if stored_action.get("mask") != [True] * 6 + [False]:
            failures.append("statistics action mask is not the OpenVLA 6D-continuous mask")
        if statistics[dataset_key].get("num_transitions") != expected_frames:
            failures.append("statistics transition count mismatch")
        if statistics[dataset_key].get("num_trajectories") != len(source_index):
            failures.append("statistics trajectory count mismatch")

    if global_index != expected_frames:
        failures.append(f"audited {global_index} frames, expected {expected_frames}")
    if action_max_abs_error != 0.0:
        failures.append(f"action round-trip max error {action_max_abs_error}")
    if front_rgb_max_pixel_error != 0:
        failures.append(f"PNG round-trip max pixel error {front_rgb_max_pixel_error}")
    if prompt_mismatches or index_mismatches or missing_images:
        failures.append(
            "sample metadata mismatch: "
            f"prompt={prompt_mismatches}, index={index_mismatches}, missing_images={missing_images}"
        )
    if statistics_max_abs_error > 1e-12:
        failures.append(f"statistics max absolute error {statistics_max_abs_error}")

    report = {
        "passed": not failures,
        "failures": failures,
        "source": str(source),
        "dataset_root": str(dataset),
        "source_index_sha256": index_hash,
        "source_manifest_sha256": manifest_hash,
        "conversion_audit_sha256": sha256(conversion_path),
        "samples_sha256": sha256(samples_path),
        "dataset_statistics_sha256": sha256(statistics_path),
        "episodes": len(source_index),
        "frames": global_index,
        "action_max_abs_error": action_max_abs_error,
        "front_rgb_max_pixel_error": front_rgb_max_pixel_error,
        "prompt_mismatches": prompt_mismatches,
        "index_mismatches": index_mismatches,
        "missing_images": missing_images,
        "unexpected_image_files": unexpected_image_files,
        "absent_image_files": absent_image_files,
        "statistics_max_abs_error": statistics_max_abs_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
