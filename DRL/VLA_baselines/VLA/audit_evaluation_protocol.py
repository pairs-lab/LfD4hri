#!/usr/bin/env python3
"""Validate every locked dev/test scene with the privileged physics oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from VLA.common.environment import ManipulationEnv
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.scripted_policy import ScriptedPolicy
from VLA.common.tasks import randomize_rl_scene, select_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "VLA/evaluation_protocol.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "VLA/audits/evaluation_protocol_oracle.json",
    )
    parser.add_argument("--development-only", action="store_true",
                        help="Qualify development without reading/executing final scenes during pipeline repair")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_teacher(protocol, profile):
    selection = protocol.get("offline_qualification_teacher")
    if selection not in (None, "central_section_clearance"):
        raise ValueError("Unknown offline qualification teacher")
    body_v9 = profile and profile.get("gripper_controller") == "body_contact_v9"
    if selection is not None:
        if not body_v9:
            raise ValueError("Section-clearance qualification requires matching V9 body physics")
        from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
        return CentralSectionClearanceTeacher
    if body_v9:
        from VLA.collision_planned_teacher import CollisionPlannedTeacher
        return CollisionPlannedTeacher
    return ScriptedPolicy


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite a preserved protocol audit")
    protocol = json.loads(args.protocol.read_text())
    profile = None
    if protocol.get("observation_profile_file"):
        from VLA.common.observation_profile import validate_profile, profile_hash
        profile = validate_profile(json.loads(Path(protocol["observation_profile_file"]).read_text()))
        if profile_hash(profile) != protocol.get("observation_profile_sha256"):
            raise ValueError("Protocol observation/controller profile changed")
    dev = [int(seed) for seed in protocol["development_seeds"]]
    test = [int(seed) for seed in protocol["final_test_seeds"]]
    if len(set(dev + test)) != len(dev) + len(test):
        raise ValueError("Development and final-test seeds must be unique and disjoint")
    if int(protocol["train_dataset_seed"]) in dev + test:
        raise ValueError("Training seed leaked into evaluation seeds")
    from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
    runtime = environment_runtime_fingerprint(profile)
    protected = {str(args.protocol.resolve()): sha256(args.protocol),
                 str(Path(__file__).resolve()): sha256(Path(__file__))}
    if protocol.get("offline_qualification_teacher") == "central_section_clearance":
        for name in ("central_section_clearance_teacher", "yaw_clear_touch_teacher", "obstacle_aware_touch_teacher",
            "staged_acquisition_teacher", "surface_limited_touch_teacher", "robust_touch_teacher",
            "geometry_clearance_teacher", "ray_refined_clearance_teacher", "adaptive_clearance_teacher",
            "neighbour_clearance_teacher", "clearance_correction_teacher", "collision_planned_teacher",
            "collision_grasp_planner", "grasp_contact_prototype"):
            path = Path(__file__).with_name(name+".py")
            protected[str(path.resolve())] = sha256(path)

    rows: list[dict] = []
    selected_splits = (("development", dev),) if args.development_only else (("development", dev), ("final_test", test))
    for split, seeds in selected_splits:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tasks = [randomize_rl_scene(task, rng) for task in select_tasks("all")]
            factory = environment_class_for_profile(profile)
            profile_kwargs = {} if profile is None else {"observation_profile": profile}
            with factory(**profile_kwargs,
                robot=protocol["robot"],
                image_size=64,
                control_hz=int(protocol["control_hz"]),
                seed=seed,
                required_success_steps=int(protocol["success"]["required_consecutive_steps"]),
                success_threshold_scale=float(protocol["success"]["reward_xml_threshold_scale"]),
                use_semantic_success=True,
                semantic_success_thresholds=protocol["success"]["semantic_thresholds"],
                render_observations=False,
            ) as env:
                teacher = protocol_teacher(protocol, profile)
                policy = teacher(env)
                for task in tasks:
                    observation = env.reset(task)
                    initial_semantic_success = env.is_semantic_success()
                    initial_metrics = env.semantic_success_metrics()
                    done = False
                    info = {"success": False}
                    exception = None
                    try:
                        policy.reset(task)
                        while not done:
                            observation, _reward, done, info = env.step(policy.predict(observation))
                    except Exception as exc:
                        exception = {"type": type(exc).__name__, "message": str(exc)}
                    if exception:
                        rows.append({"split": split, "seed": seed, "task": task.name,
                            "prompt": task.prompt, "initial_semantic_success": bool(initial_semantic_success),
                            "success": False, "steps": env.step_count, "exception": exception})
                        print(json.dumps(rows[-1]), flush=True)
                        continue
                    rows.append(
                        {
                            "split": split,
                            "seed": seed,
                            "task": task.name,
                            "prompt": task.prompt,
                            "pick_name": task.pick_name,
                            "place_name": task.place_name,
                            "initial_semantic_success": bool(initial_semantic_success),
                            "initial_pick_lift_m": float(initial_metrics["pick_lift_m"]),
                            "initial_goal_xy_distance_m": initial_metrics["goal_xy_distance_m"],
                            "initial_goal_vertical_offset_m": initial_metrics[
                                "goal_vertical_offset_m"
                            ],
                            "success": bool(info["success"]),
                            "steps": int(info["step"]),
                            "strict_raw_success": bool(info["strict_raw_success"]),
                            "semantic_raw_success": bool(info["semantic_raw_success"]),
                            "distance_only_success": bool(info["distance_only_success"]),
                            "ee_object_distance_m": float(info["ee_object_distance"]),
                            "object_goal_distance_m": float(info["object_goal_distance"]),
                            "pick_lift_m": float(info["pick_lift_m"]),
                            "goal_xy_distance_m": info["goal_xy_distance_m"],
                            "goal_vertical_offset_m": info["goal_vertical_offset_m"],
                            "safety_violation": info.get("safety_violation"),
                        }
                    )
                    print(split, seed, task.name, info["success"], info.get("safety_violation"), flush=True)

    failures = [
        row
        for row in rows
        if not row["success"] or row["initial_semantic_success"]
    ]
    expected = 4 * sum(len(seeds) for _, seeds in selected_splits)
    verify_environment_runtime_fingerprint(runtime, profile)
    if any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise RuntimeError("Protocol/offline teacher changed during qualification")
    report = {
        "passed": len(rows) == expected and not failures,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "environment_sha256": sha256(PROJECT_ROOT / "VLA/common/environment.py"),
        "tasks_sha256": sha256(PROJECT_ROOT / "VLA/common/tasks.py"),
        "spatial_sha256": sha256(PROJECT_ROOT / "VLA/common/spatial.py"),
        "scenes": sum(len(seeds) for _, seeds in selected_splits),
        "validated_splits": [name for name, _ in selected_splits],
        "final_test_executed": not args.development_only,
        "observation_profile_sha256": protocol.get("observation_profile_sha256"),
        "profiled_environment_sha256": sha256(PROJECT_ROOT / "VLA/common/observation_profile.py") if profile else None,
        "gripper_sha256": sha256(PROJECT_ROOT / "VLA/common/robotiq.py"),
        "trials": len(rows),
        "expected_trials": expected,
        "environment_runtime_fingerprint": runtime,
        "protected_sources_sha256": protected,
        "offline_teacher": protocol.get("offline_qualification_teacher", "legacy_default"),
        "learned_policy_called": False,
        "teacher_outcomes_used_to_change_seeds": False,
        "successes": sum(int(row["success"]) for row in rows),
        "initial_false_positives": sum(
            int(row["initial_semantic_success"]) for row in rows
        ),
        "failures": failures,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["passed"]:
        raise RuntimeError("At least one locked evaluation scene failed oracle validation")


if __name__ == "__main__":
    main()
