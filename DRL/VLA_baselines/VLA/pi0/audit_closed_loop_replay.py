#!/usr/bin/env python3
"""Replay demonstrations through the exact openpi inference action adapter.

Array-level Jacobian error is necessary but not sufficient: small one-step
errors can compound and break a grasp. This audit resets the real benchmark
scene for every stored episode, executes either the source canonical actions
or the native absolute-joint targets through ``joint_targets_to_canonical``,
and checks the original strict reward predicate in closed loop.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure

from VLA.common.environment import ManipulationEnv
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.common.tasks import TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "VLA/audits/camera_v4_vla_oneshot/pi0_closed_loop_replay.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("native", "canonical", "both"), default="both")
    parser.add_argument(
        "--native-actions-root", type=Path,
        help="FAST codec export; use only with --mode native (no changes to the scene)",
    )
    parser.add_argument("--episode-files", nargs="+", help="Exact filenames for failure diagnosis")
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Deterministic per-task prefix for a quick pilot; default audits all episodes",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent PyBullet replay processes; use >1 for exhaustive audits",
    )
    parser.add_argument(
        "--native-success-mode",
        choices=("semantic", "strict"),
        default="semantic",
        help="Native adapter gate; canonical control replay is always held to strict reward success",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_items(index: list[dict], maximum: int | None) -> list[dict]:
    if maximum is None:
        return index
    if maximum < 1:
        raise ValueError("--max-episodes-per-task must be positive")
    counts: Counter[str] = Counter()
    selected = []
    for item in index:
        task_name = item["task"]["name"]
        if counts[task_name] < maximum:
            selected.append(item)
            counts[task_name] += 1
    return selected


def replay(
    source: Path,
    item: dict,
    mode: str,
    image_size: int,
    native_success_mode: str,
    native_actions_root: Path | None = None,
) -> dict:
    task = TaskSpec.from_dict(item["task"])
    with np.load(source / item["file"]) as episode:
        canonical = np.asarray(episode["canonical_actions"], dtype=np.float32)
        targets = np.asarray(episode["next_joint_targets"], dtype=np.float32)
        closures = np.asarray(native_gripper_closure(episode), dtype=np.float32)
    if native_actions_root is not None:
        with np.load(native_actions_root / item["file"]) as decoded:
            targets = np.asarray(decoded["next_joint_targets"], dtype=np.float32)
            closures = np.asarray(decoded["next_gripper_closure"], dtype=np.float32)
        if targets.shape != (len(canonical), 6) or closures.shape != (len(canonical),):
            raise ValueError(f"FAST codec/source transition mismatch: {item['file']}")

    with audit_environment(source,
        robot="ur5e",
        image_size=image_size,
        control_hz=10,
        assist_grasp=True,
        seed=0,
        required_success_steps=1,
        success_threshold_scale=1.0,
        use_semantic_success=False,
        render_observations=False,
    ) as env:
        reset_source_episode(env, source, item)
        initial_strict_success = env.is_strict_success()
        success = False
        ever_strict = False
        ever_semantic = False
        final_info = None
        conversion_errors: list[float] = []
        translation_errors, rotation_errors = [], []
        for step in range(len(canonical)):
            if mode == "canonical":
                action = canonical[step]
            else:
                action = env.joint_targets_to_canonical(targets[step], float(closures[step]))
                conversion_errors.append(
                    float(np.max(np.abs(action - canonical[step])))
                )
                translation_errors.append(float(np.max(np.abs(action[:3]-canonical[step, :3]))))
                rotation_errors.append(float(np.max(np.abs(action[3:6]-canonical[step, 3:6]))))
            _observation, _reward, terminated, final_info = env.step(action)
            ever_strict = ever_strict or bool(final_info["strict_raw_success"])
            ever_semantic = ever_semantic or bool(final_info["semantic_raw_success"])
            success = success or bool(final_info["success"])
            if terminated:
                break
        if final_info is None:
            raise RuntimeError(f"Empty episode: {item['file']}")
        scored_success = (
            ever_strict
            if mode == "canonical" or native_success_mode == "strict"
            else ever_semantic
        )
        return {
            "file": item["file"],
            "task": task.name,
            "mode": mode,
            "stored_steps": int(len(canonical)),
            "executed_steps": int(final_info["step"]),
            "initial_strict_success": bool(initial_strict_success),
            "success": bool(scored_success),
            "strict_termination_success": bool(success),
            "ever_strict_success": bool(ever_strict),
            "ever_semantic_success": bool(ever_semantic),
            "final_strict_success": bool(final_info["strict_raw_success"]),
            "final_semantic_success": bool(final_info["semantic_raw_success"]),
            "final_ee_object_distance_m": float(final_info["ee_object_distance"]),
            "final_object_goal_distance_m": float(final_info["object_goal_distance"]),
            "max_online_canonical_error": (
                None if not conversion_errors else float(max(conversion_errors))
            ),
            "max_online_translation_error_m": max(translation_errors, default=None),
            "max_online_rotation_error_rad": max(rotation_errors, default=None),
        }


def replay_job(arguments: tuple) -> dict:
    """Pickle-friendly wrapper for independent PyBullet replay workers."""
    return replay(*arguments)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    source = args.source.resolve()
    from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
    from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint
    from VLA.pi0.codec_provenance import verify_fast_codec_binding
    profile = json.loads((source/"dataset_manifest.json").read_text()).get("observation_profile")
    runtime_fingerprint = environment_runtime_fingerprint(profile)
    index = json.loads((source / "index.json").read_text())
    source_payload = source_payload_fingerprint(source, index)
    initial_index_hash = sha256(source / "index.json")
    initial_manifest_hash = sha256(source / "dataset_manifest.json")
    items = select_items(index, args.max_episodes_per_task)
    requested_files = None
    if args.episode_files is not None:
        if args.max_episodes_per_task is not None:
            raise ValueError("Use --episode-files or --max-episodes-per-task, not both")
        requested_files = set(args.episode_files)
        items = [item for item in items if item["file"] in requested_files]
        if len(items) != len(requested_files) or not items:
            raise ValueError("Requested diagnostic episodes are missing or duplicated")
    codec_manifest = None
    native_actions_root = None
    if args.native_actions_root is not None:
        if args.mode != "native":
            raise ValueError("--native-actions-root requires --mode native")
        native_actions_root = args.native_actions_root.resolve()
        codec_manifest = json.loads((native_actions_root / "codec_manifest.json").read_text())
        verify_fast_codec_binding(codec_manifest, source, native_actions_root)
        if (
            not codec_manifest.get("passed")
            or Path(codec_manifest.get("source", "")).resolve() != source
            or codec_manifest.get("source_index_sha256") != sha256(source / "index.json")
            or codec_manifest.get("source_manifest_sha256") != sha256(source / "dataset_manifest.json")
            or codec_manifest.get("tokenizer_source_sha256") != sha256(PROJECT_ROOT / "VLA/pi0/fast_tokenizer.py")
        ):
            raise RuntimeError("FAST codec export is failed, stale or for a different source")
        selected_files = set(codec_manifest["selected_files"])
        if requested_files is not None:
            if not requested_files <= selected_files:
                raise RuntimeError("Requested files were not exported by the FAST codec")
            selected_files = requested_files
        items = [item for item in items if item["file"] in selected_files]
        if len(items) != len(selected_files) or not items:
            raise RuntimeError("FAST codec replay does not cover every exported episode")
    modes = ("canonical", "native") if args.mode == "both" else (args.mode,)
    rows: list[dict] = []
    for mode in modes:
        jobs = [
            (source, item, mode, args.image_size, args.native_success_mode, native_actions_root)
            for item in items
        ]
        if args.workers == 1:
            iterator = map(replay_job, jobs)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            iterator = executor.map(replay_job, jobs)
        try:
            for number, (item, row) in enumerate(zip(items, iterator, strict=True), start=1):
                rows.append(row)
                print(
                    f"{mode} {number}/{len(items)} {item['file']}: "
                    f"success={row['success']}",
                    flush=True,
                )
        finally:
            if executor is not None:
                executor.shutdown()

    summaries = {}
    failures = []
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        by_task = {}
        for task_name in ("touch", "pick", "move", "pick_place"):
            task_rows = [row for row in selected if row["task"] == task_name]
            by_task[task_name] = {
                "episodes": len(task_rows),
                "successes": sum(int(row["success"]) for row in task_rows),
                "success_rate": (
                    sum(int(row["success"]) for row in task_rows) / len(task_rows)
                    if task_rows
                    else None
                ),
            }
        summaries[mode] = {
            "episodes": len(selected),
            "successes": sum(int(row["success"]) for row in selected),
            "success_rate": sum(int(row["success"]) for row in selected) / len(selected),
            "initial_false_positives": sum(
                int(row["initial_strict_success"]) for row in selected
            ),
            "by_task": by_task,
        }
        failures.extend(
            f"{mode}:{row['file']}" for row in selected if not row["success"]
        )
        failures.extend(
            f"initial_success:{mode}:{row['file']}"
            for row in selected
            if row["initial_strict_success"]
        )

    # Canonical replay must be exact. Native replay is the actual policy path,
    # so any failure is treated as a setup defect, not blamed on the model.
    verify_environment_runtime_fingerprint(runtime_fingerprint, profile)
    verify_source_payload_fingerprint(source_payload, source, index)
    if (sha256(source / "index.json") != initial_index_hash
            or sha256(source / "dataset_manifest.json") != initial_manifest_hash):
        raise RuntimeError("Source metadata changed during closed-loop replay")
    if codec_manifest is not None:
        verify_fast_codec_binding(codec_manifest, source, native_actions_root)
    report = {
        "passed": not failures,
        "source": str(source),
        "source_index_sha256": sha256(source / "index.json"),
        "source_manifest_sha256": sha256(source / "dataset_manifest.json"),
        "source_payload_fingerprint": source_payload,
        "codec_payload_fingerprint": None if codec_manifest is None else codec_manifest.get("codec_payload_fingerprint"),
        "environment_sha256": sha256(PROJECT_ROOT / "VLA/common/environment.py"),
        "environment_runtime_fingerprint": runtime_fingerprint,
        "adapter_sha256": sha256(PROJECT_ROOT / "VLA/pi0/adapter.py"),
        "mode": args.mode,
        "native_success_mode": args.native_success_mode,
        "workers": args.workers,
        "all_episodes": len(items) == len(index),
        "native_actions_root": None if native_actions_root is None else str(native_actions_root),
        "codec_manifest_sha256": (
            None if native_actions_root is None
            else sha256(native_actions_root / "codec_manifest.json")
        ),
        "codec_action_horizon": None if codec_manifest is None else codec_manifest["action_horizon"],
        "summaries": summaries,
        "failures": failures,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if failures:
        raise RuntimeError("Closed-loop demonstration replay failed")


if __name__ == "__main__":
    main()
