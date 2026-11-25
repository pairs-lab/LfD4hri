#!/usr/bin/env python3
"""Replace approximate LeRobot numeric statistics with exact source statistics.

LeRobot's streaming accumulator is appropriate for large image datasets, but
``E[x^2] - E[x]^2`` loses precision for an almost-fixed UR5e wrist joint and
its streaming histogram is not an exact quantile estimator.  SmolVLA consumes
the numeric mean/std in ``meta/stats.json``.  This tool recomputes only the
state/action entries from the immutable audited NPZ source and leaves image
statistics untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STATISTICS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


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


def exact(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def max_error(actual: dict, expected: dict) -> dict[str, float]:
    return {
        statistic: float(
            np.max(
                np.abs(
                    np.asarray(actual[statistic], dtype=np.float64)
                    - np.asarray(expected[statistic], dtype=np.float64)
                )
            )
        )
        for statistic in STATISTICS
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    dataset_root = args.dataset_root.resolve()
    index_path = source / "index.json"
    source_manifest_path = source / "dataset_manifest.json"
    conversion_path = dataset_root / "conversion_manifest.json"
    stats_path = dataset_root / "meta" / "stats.json"
    for path in (index_path, source_manifest_path, conversion_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    index = json.loads(index_path.read_text())
    conversion = json.loads(conversion_path.read_text())
    if any(not item.get("success") for item in index):
        raise RuntimeError("Refusing source dataset containing failed demonstrations")
    if (
        Path(conversion.get("source", "")).resolve() != source
        or conversion.get("source_index_sha256") != sha256(index_path)
        or conversion.get("source_manifest_sha256") != sha256(source_manifest_path)
        or conversion.get("episodes") != len(index)
    ):
        raise RuntimeError("Converted dataset lineage does not match the source")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for item in index:
        with np.load(source / item["file"]) as episode:
            states.append(
                np.concatenate(
                    (
                        episode["joint_positions"].astype(np.float64),
                        episode["gripper_closure"].astype(np.float64)[:, None],
                    ),
                    axis=1,
                )
            )
            actions.append(episode["canonical_actions"].astype(np.float64))
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    if state.shape != (conversion.get("frames"), 7) or action.shape != state.shape:
        raise RuntimeError(f"Unexpected numeric arrays: state={state.shape}, action={action.shape}")

    expected = {
        "observation.state": exact(state),
        "action": exact(action),
    }
    stats = json.loads(stats_path.read_text())
    before = {key: max_error(stats[key], value) for key, value in expected.items()}
    for key, value in expected.items():
        stats[key] = value
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    reloaded = json.loads(stats_path.read_text())
    after = {key: max_error(reloaded[key], value) for key, value in expected.items()}
    if any(error != 0.0 for errors in after.values() for error in errors.values()):
        raise RuntimeError(f"Exact statistics did not round-trip: {after}")

    conversion.update(
        {
            "numeric_stats_method": "exact_numpy_float64",
            "numeric_stats_sha256": sha256(stats_path),
            "numeric_stats_source_index_sha256": sha256(index_path),
            "numeric_stats_source_manifest_sha256": sha256(source_manifest_path),
        }
    )
    conversion_path.write_text(json.dumps(conversion, indent=2) + "\n")
    report = {
        "passed": True,
        "source": str(source),
        "dataset_root": str(dataset_root),
        "episodes": len(index),
        "frames": int(state.shape[0]),
        "source_index_sha256": sha256(index_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "statistics_method": "exact_numpy_float64",
        "stats_sha256": sha256(stats_path),
        "before_max_abs_error": before,
        "after_max_abs_error": after,
        "minimum_nonzero_std": {
            key: float(np.min(np.asarray(value["std"])[np.asarray(value["std"]) > 0]))
            for key, value in expected.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
