#!/usr/bin/env python3
"""Run and aggregate one locked split of the VLA evaluation protocol.

This script intentionally defaults to the development split.  Final-test
seeds require an explicit flag and exactly one already-selected replan value,
which makes accidental test-set tuning harder.  Development sweeps do not
encode video unless the protocol explicitly locks a development video seed;
the final split records the protocol-designated combined video and skips
video for the remaining seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from VLA.common.evaluation_statistics import success_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "VLA/evaluation_protocol.json"
DEFAULT_ORACLE_AUDIT = (
    PROJECT_ROOT / "VLA/audits/evaluation_protocol_oracle_v3_spatial_fixed.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("pi0", "pi0_fast", "openvla", "cogact", "rdt", "smolvla"),
        required=True,
    )
    parser.add_argument("--split", choices=("development", "final_test"), default="development")
    parser.add_argument(
        "--replan-steps",
        type=int,
        nargs="+",
        default=None,
        help="Defaults to all locked dev candidates; required as one frozen value for final_test",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--oracle-audit", type=Path, default=DEFAULT_ORACLE_AUDIT)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Immutable label appended to every test id and aggregate directory "
            "(for example ckpt29999). Required for non-dry runs so checkpoint "
            "evaluations cannot overwrite one another."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "VLA/evaluations",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_minimum(task: dict) -> float:
    if task.get("min_normalized_semantic_gap") is not None:
        return float(task["min_normalized_semantic_gap"])
    # Backward-compatible fallback for pre-v2 diagnostic artifacts. New
    # protocol runs always persist the paired XY/Z semantic gap above.
    if task["name"] == "touch":
        value = task["min_ee_object_distance_m"]
        threshold = task["semantic_thresholds"]["touch_distance_m"]
    elif task["name"] == "pick":
        lift = task["max_pick_lift_m"]
        threshold = task["semantic_thresholds"]["pick_min_lift_m"]
        if lift is None or threshold <= 0:
            return float("inf")
        return max(0.0, 1.0 - float(lift) / float(threshold))
    else:
        value = task["min_goal_xy_distance_m"]
        threshold = task["semantic_thresholds"][f"{task['name']}_xy_radius_m"]
    if value is None or threshold <= 0:
        return float("inf")
    return float(value) / float(threshold)


def validate_protocol(protocol_path: Path, oracle_path: Path, requested_split="development") -> tuple[dict, str]:
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = sha256(protocol_path)
    audit = json.loads(oracle_path.read_text())
    if (
        not protocol.get("locked_before_final_evaluation")
        or not audit.get("passed")
        or audit.get("failures")
        or Path(audit.get("protocol", "")).resolve() != protocol_path.resolve()
        or audit.get("protocol_sha256") != protocol_hash
        or audit.get("environment_sha256")
        != sha256(PROJECT_ROOT / "VLA/common/environment.py")
        or audit.get("tasks_sha256") != sha256(PROJECT_ROOT / "VLA/common/tasks.py")
        or audit.get("spatial_sha256") != sha256(PROJECT_ROOT / "VLA/common/spatial.py")
    ):
        raise RuntimeError("Evaluation protocol is unlocked, stale, or failed its oracle audit")
    if requested_split not in audit.get("validated_splits", ["development", "final_test"]):
        raise RuntimeError("Oracle audit did not qualify the requested split")
    if protocol.get("observation_profile_sha256"):
        from VLA.common.observation_profile import profile_hash, validate_profile
        profile = validate_profile(json.loads(Path(protocol["observation_profile_file"]).read_text()))
        if (profile_hash(profile) != protocol["observation_profile_sha256"]
                or audit.get("observation_profile_sha256") != protocol["observation_profile_sha256"]
                or audit.get("profiled_environment_sha256") != sha256(PROJECT_ROOT / "VLA/common/observation_profile.py")
                or audit.get("gripper_sha256") != sha256(PROJECT_ROOT / "VLA/common/robotiq.py")):
            raise RuntimeError("Versioned controller/camera oracle audit is missing or stale")
        if protocol.get("evaluation_scope") == "postfit_10_per_action_pilot":
            from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
            if not audit.get("environment_runtime_fingerprint"):
                raise RuntimeError("Postfit pilot requires ALL runtime sources, not only the outer environment hash")
            verify_environment_runtime_fingerprint(audit["environment_runtime_fingerprint"], profile)
    return protocol, protocol_hash


def validate_final_grasp_scope(split, sensor_profile):
    if split == "final_test" and (sensor_profile or {}).get("grasp_candidate_scope") != "all_dynamic_scene_objects":
        raise ValueError("Final comparison requires task-independent grasp physics; legacy target-only assistance is diagnostic only")


def locked_split_seeds(protocol, split):
    key = "development_seeds" if split == "development" else "final_test_seeds"
    seeds = [int(seed) for seed in protocol[key]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Requested split must contain nonempty unique locked seeds")
    other = "final_test_seeds" if split == "development" else "development_seeds"
    if set(seeds) & set(map(int, protocol[other])) or int(protocol["train_dataset_seed"]) in seeds:
        raise ValueError("Requested evaluation seeds overlap train or the other split")
    return seeds


def locked_video_seed(protocol, split, seeds):
    key = "development_video_seed" if split == "development" else "combined_video_seed"
    value = protocol["reporting"].get(key)
    if value is None and split == "development":
        return None  # Historical development sweeps stay video-free.
    if value is None or int(value) not in seeds:
        raise ValueError("Locked video seed must belong to the requested split")
    return int(value)


def main() -> None:
    args = parse_args()
    if args.run_tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        raise ValueError("--run-tag must contain only letters, digits, '_', '-' or '.'")
    if not args.dry_run and args.run_tag is None:
        raise ValueError("--run-tag is required for a non-dry protocol run")
    protocol_path = args.protocol.resolve()
    oracle_path = args.oracle_audit.resolve()
    protocol, protocol_hash = validate_protocol(protocol_path, oracle_path, args.split)
    from VLA.common.profile_binding import checkpoint_profile
    from VLA.common.observation_profile import profile_hash
    sensor_profile = checkpoint_profile(args.checkpoint)
    sensor_hash = profile_hash(sensor_profile) if sensor_profile is not None else None
    if protocol.get("observation_profile_sha256") != sensor_hash:
        raise ValueError("Checkpoint camera/home differs from locked protocol; qualify and lock a matching new protocol before evaluation")
    validate_final_grasp_scope(args.split, sensor_profile)

    allowed = [int(value) for value in protocol["action_chunk_selection"]["development_candidates"]]
    # Vanilla OpenVLA predicts exactly one action; it has no native chunk to
    # queue.  Repeating rp1/rp2/rp5 produces identical controls and should not
    # masquerade as an architecture-specific chunk selection experiment.
    default_replans = [1] if args.model == "openvla" else allowed
    replans = default_replans if args.replan_steps is None else [int(value) for value in args.replan_steps]
    if args.model == "openvla" and replans != [1]:
        raise ValueError("Vanilla OpenVLA predicts one action; use --replan-steps 1")
    if not replans or len(set(replans)) != len(replans) or any(value not in allowed for value in replans):
        raise ValueError(f"replan steps must be unique members of locked candidates {allowed}")
    if args.split == "final_test" and len(replans) != 1:
        raise ValueError("final_test requires exactly one frozen --replan-steps value")

    seeds = locked_split_seeds(protocol, args.split)
    video_seed = locked_video_seed(protocol, args.split, seeds)
    run_tag = args.run_tag or "dry_run"
    run_root = args.output_root.resolve() / args.model / args.split / run_tag
    rows: list[dict] = []
    commands: list[list[str]] = []

    for replan in replans:
        for seed in seeds:
            test_id = f"protocol_{args.split}_{run_tag}_seed{seed}_rp{replan}"
            command = [
                sys.executable,
                "-m",
                "VLA.benchmark",
                "--model",
                args.model,
                "--robot",
                str(protocol["robot"]),
                "--tasks",
                ",".join(protocol["tasks"]),
                "--test-id",
                test_id,
                "--control-hz",
                str(protocol["control_hz"]),
                "--seed",
                str(seed),
                "--scene-profile",
                str(protocol["scene_profile"]),
                "--replan-steps",
                str(replan),
                "--required-success-steps",
                str(protocol["success"]["required_consecutive_steps"]),
                "--success-threshold-scale",
                str(protocol["success"]["reward_xml_threshold_scale"]),
                "--success-mode",
                "semantic",
                "--host",
                args.host,
            ]
            for name, value in protocol["success"]["semantic_thresholds"].items():
                command.extend(("--semantic-" + name.replace("_", "-"), str(value)))
            if args.port is not None:
                command.extend(("--port", str(args.port)))
            if args.checkpoint is not None:
                command.extend(("--checkpoint", args.checkpoint))
            record_video = seed == video_seed
            if not record_video:
                command.append("--no-video")
            commands.append(command)
            if args.dry_run:
                print(" ".join(command))
                continue

            result_path = PROJECT_ROOT / "VLA" / args.model / test_id / "results.json"
            if result_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite an existing protocol result: {result_path}"
                )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            result = json.loads(result_path.read_text())
            if result.get("model") != args.model or len(result.get("tasks", ())) != len(protocol["tasks"]):
                raise RuntimeError(f"Malformed benchmark result: {result_path}")
            if any(task.get("error") is not None for task in result["tasks"]):
                raise RuntimeError(f"Integration error is not a task failure: {result_path}")
            for task in result["tasks"]:
                rows.append(
                    {
                        "seed": seed,
                        "replan_steps": replan,
                        "task": task["name"],
                        "success": bool(task["success"]),
                        "strict_success": bool(task["ever_strict_raw_success"]),
                        "normalized_minimum_semantic_gap": normalized_minimum(task),
                        "result": str(result_path.resolve()),
                        "video_recorded": record_video,
                    }
                )

    if args.dry_run:
        return

    summaries = []
    for replan in replans:
        selected = [row for row in rows if row["replan_steps"] == replan]
        summaries.append(
            {
                "replan_steps": replan,
                "trials": len(selected),
                "successes": sum(int(row["success"]) for row in selected),
                "success_rate": sum(int(row["success"]) for row in selected) / len(selected),
                "uncertainty": success_statistics(selected),
                "strict_success_rate": sum(int(row["strict_success"]) for row in selected)
                / len(selected),
                "mean_normalized_minimum_semantic_gap": sum(
                    row["normalized_minimum_semantic_gap"] for row in selected
                )
                / len(selected),
            }
        )
    report = {
        "model": args.model,
        "split": args.split,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "oracle_audit": str(oracle_path),
        "checkpoint": args.checkpoint,
        "run_tag": run_tag,
        "replan_steps": replans,
        "seeds": seeds,
        "summaries": summaries,
        "rows": rows,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / "aggregate.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Aggregate: {output}")


if __name__ == "__main__":
    main()
