"""Collect independent V9 TRAIN candidates with frozen contracts, no promotion.

Same random_env scene sampler/assets, no teacher/model outcome-based scene
filtering, no final-test data. Stop and preserve the FIRST failed/unsafe episode.
Only numerical source contracts are inherited; parent camera/runtime is unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pybullet as p

from config import PICK_TARGETS, PLACE_TARGETS
from VLA.audit_training_readiness import physical_scene_key
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.common.tasks import randomize_rl_scene, select_tasks
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.dataset_replay_context import audit_environment
from VLA.evaluate_smolvla_v9_source import TASKS, verify_source_audit


def frozen_thresholds(rows):
    result = {}
    for task in TASKS:
        values = {recorded_distance_thresholds(row) for row in rows if row["task"]["name"] == task}
        if len(values) != 1:
            raise ValueError("Missing or mixed frozen reward distances: " + task)
        result[task] = values.pop()
    return result


def stratified_names(task, episode_number):
    pick = list(PICK_TARGETS)[episode_number % len(PICK_TARGETS)]
    if task not in {"move", "pick_place"}:
        return pick, None
    places = list(PLACE_TARGETS)
    index = episode_number % len(places)
    while places[index] == pick:
        index = (index + 1) % len(places)
    return pick, places[index]


class FrozenRewardReset:
    def __init__(self, env, thresholds):
        self.env, self.thresholds = env, thresholds

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        observation = self.env.reset(task)
        self.env.dist_threshold, self.env.obj_threshold = self.thresholds[task.name]
        return observation


def bootstrap_prefix(source, plan, reproduction_path):
    """Retain a contiguous immutable prefix, never skip its failed next scene."""
    source = Path(source).resolve()
    old_plan = json.loads((source/"collection_plan.json").read_text())
    state = json.loads((source/"collection_state.json").read_text())
    rows = json.loads((source/"index.json").read_text())
    reproduction = json.loads(Path(reproduction_path).read_text())
    keys = ("seed", "episodes_per_task", "contract_source", "contract_index_sha256", "contract_manifest_sha256",
            "validation", "validation_index_sha256", "validation_manifest_sha256", "frozen_reward_distances",
            "observation_profile_sha256", "environment_runtime_fingerprint", "pybullet_binary_sha256")
    if any(old_plan[key] != plan[key] for key in keys):
        raise ValueError("Bootstrap changes the physical/data/seed contract")
    if (not rows or len(rows) != state["completed_episodes"]
            or sha256_file(source/"index.json") != state["index_sha256"]
            or reproduction["state_sha256"] != sha256_file(source/"collection_state.json")
            or not reproduction["exception"]):
        raise ValueError("Bootstrap lacks exact preserved exception/RNG evidence")
    rng = np.random.default_rng(plan["seed"])
    jobs = [(task, number) for task in select_tasks("all") for number in range(plan["episodes_per_task"])]
    for row, (task, number) in zip(rows, jobs, strict=False):
        pick, place = stratified_names(task.name, number)
        sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
        if (row["file"] != f"{task.name}_{number:05d}.npz" or not row["success"] or row.get("safety_violation")
                or json.loads(json.dumps(sampled.to_dict())) != row["task"]):
            raise ValueError("Bootstrap is not the exact safe contiguous scene prefix")
    if len(rows) >= len(jobs) or rng.bit_generator.state != state["rng_state"]:
        raise ValueError("Bootstrap RNG is not the exact next-scene checkpoint")
    task, number = jobs[len(rows)]
    pick, place = stratified_names(task.name, number)
    sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
    if (Path(reproduction["parent_source"]).resolve() != source
            or reproduction["index_sha256"] != state["index_sha256"]
            or json.loads(json.dumps(sampled.to_dict())) != reproduction["task"]):
        raise ValueError("Exception evidence is not the exact next failed scene")
    metadata = {"source": str(source), "episodes": len(rows), "index_sha256": state["index_sha256"],
                "manifest_sha256": sha256_file(source/"dataset_manifest.json"),
                "state_sha256": sha256_file(source/"collection_state.json"),
                "reproduction_sha256": sha256_file(reproduction_path),
                "old_teacher": old_plan["expert_teacher"], "new_teacher": plan["expert_teacher"],
                "same_seed_and_scene_sequence_verified": True,
                "failed_next_scene_skipped": False,
                "payload_sha256": {row["file"]: sha256_file(source/row["file"]) for row in rows}}
    return rows, state["rng_state"], metadata


def bootstrap_repaired_prefix(source, plan, repair_report_path):
    """Retain safe labels; REPEAT the exact failed scene under an audited repair.

    Unlike legacy exception bootstrap, this handles a committed failed normal
    or recovery episode. The old failure stays immutable. No RNG reroll, failed
    layout filtering, edited payload or runtime relaxation is allowed.
    """
    source = Path(source).resolve()
    old_plan = json.loads((source/"collection_plan.json").read_text())
    state = json.loads((source/"collection_state.json").read_text())
    rows = json.loads((source/"index.json").read_text())
    repair = json.loads(Path(repair_report_path).read_text())
    keys = ("seed", "episodes_per_task", "contract_source", "contract_index_sha256", "contract_manifest_sha256",
            "validation", "validation_index_sha256", "validation_manifest_sha256", "frozen_reward_distances",
            "observation_profile_sha256", "environment_runtime_fingerprint", "pybullet_binary_sha256",
            "closed_hover_recovery_every", "planned_ordinary_episodes", "planned_closed_hover_recovery_episodes",
            "planned_episodes")
    if any(old_plan.get(key) != plan.get(key) for key in keys):
        raise ValueError("Repair bootstrap changes physical/data/seed/recovery contract")
    if (state["status"] != "stopped_on_failed_or_unsafe_episode" or len(rows) != state["completed_episodes"]
            or not rows or state["last_file"] != rows[-1]["file"]
            or sha256_file(source/"index.json") != state["index_sha256"]
            or rows[-1]["success"] and not rows[-1].get("safety_violation")
            or any(not row["success"] or row.get("safety_violation") for row in rows[:-1])):
        raise ValueError("Requires exactly one preserved final failure and an immutable safe prefix")
    failed = rows[-1]
    pose_repair = plan["expert_teacher"] == "staged_pose_revalidated_acquisition"
    teacher_path = Path(__file__).with_name("pose_revalidated_acquisition_teacher.py" if pose_repair
                                           else "obstacle_aware_touch_teacher.py").resolve()
    reset_verified = repair.get("reset_exact") is True
    if pose_repair:
        # Preserve raw float64 error in the report. RGB/joints must be BIT
        # exact and encoder exact at actual float32 policy precision. The
        # original 7.8e-16 encoder rounding is NOT a scene/reset perturbation.
        errors = repair.get("reset_errors", {})
        reset_verified = (repair.get("reset_policy_inputs_exact") is True
            and all(errors.get(k) == 0 for k in ("front_rgb", "wrist_rgb", "joint_positions"))
            and isinstance(errors.get("gripper_closure"), (int, float))
            and 0 <= errors["gripper_closure"] <= 1e-12)
    expected_teacher = {"staged_obstacle_touch": "obstacle_aware",
        "staged_open_obstacle_touch": "open_verified",
        "staged_pose_revalidated_acquisition":
            "VLA.pose_revalidated_acquisition_teacher:PoseRevalidatedAcquisitionTeacher"}.get(plan["expert_teacher"])
    if (Path(repair["source"]).resolve() != source or repair["episode"] != failed["file"]
            or repair["source_index_sha256"] != state["index_sha256"]
            or repair["source_state_sha256"] != sha256_file(source/"collection_state.json")
            or repair["source_payload_sha256"] != sha256_file(source/failed["file"])
            or repair.get("task") != failed["task"] or repair.get("passed") is not True
            or not reset_verified or repair.get("runtime_changed") is not False
            or repair.get("training_performed") is not False or repair.get("final_test_used") is not False
            or repair.get("teacher") != expected_teacher
            or repair["protected_hashes"].get(str(teacher_path)) != sha256_file(teacher_path)
            or not repair["metadata"]["success"] or repair["metadata"].get("safety_violation")):
        raise ValueError("Missing exact safe same-reset repair evidence; no skipped/modified failure")
    replay_path = None
    if pose_repair:
        replay_path = Path(repair.get("canonical_native_replay_report", ""))
        if (not replay_path.is_file() or repair.get("canonical_native_replay_sha256") != sha256_file(replay_path)):
            raise ValueError("Pose repair requires hash-bound canonical AND native physical replay")
        replay = json.loads(replay_path.read_text())
        repaired_source = Path(repair_report_path).resolve().parent
        if (replay.get("passed") is not True or replay.get("mode") != "both"
                or replay.get("native_success_mode") != "strict" or replay.get("failures")
                or Path(replay["source"]).resolve() != repaired_source
                or replay.get("source_index_sha256") != sha256_file(repaired_source / "index.json")
                or replay.get("source_manifest_sha256") != sha256_file(repaired_source / "dataset_manifest.json")
                or any(replay["summaries"][mode]["episodes"] != 1
                       or replay["summaries"][mode]["successes"] != 1
                       or replay["summaries"][mode]["initial_false_positives"] != 0
                       for mode in ("canonical", "native"))):
            raise ValueError("Pose repair physical replay did not qualify the exact preserved scene")
    rng = np.random.default_rng(plan["seed"])
    cursor, continuation_rng = 0, None
    every = plan.get("closed_hover_recovery_every", 0)
    for task in select_tasks("all"):
        for number in range(plan["episodes_per_task"]):
            before = copy.deepcopy(rng.bit_generator.state)
            pick, place = stratified_names(task.name, number)
            sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
            files = [f"{task.name}_{number:05d}.npz"]
            if every and (number+1)%every == 0:
                files.append(f"{task.name}_{number:05d}_closedhover.npz")
            for filename in files:
                if cursor >= len(rows):
                    break
                row = rows[cursor]
                if (row["file"] != filename
                        or row["task"] != json.loads(json.dumps(sampled.to_dict()))):
                    raise ValueError("Repair source is not the exact contiguous declared scene sequence")
                cursor += 1
                if cursor == len(rows):
                    continuation_rng = before
                    break
            if continuation_rng is not None:
                break
        if continuation_rng is not None:
            break
    if cursor != len(rows) or continuation_rng is None or rng.bit_generator.state != state["rng_state"]:
        raise ValueError("Repair RNG checkpoint does not reproduce the failed scene")
    safe_rows = rows[:-1]
    metadata = {"source": str(source), "episodes": len(safe_rows), "index_sha256": state["index_sha256"],
        "manifest_sha256": sha256_file(source/"dataset_manifest.json"),
        "state_sha256": sha256_file(source/"collection_state.json"),
        "repair_report_sha256": sha256_file(repair_report_path), "repair_report": str(Path(repair_report_path).resolve()),
        "failed_file": failed["file"], "failed_payload_sha256": sha256_file(source/failed["file"]),
        "old_teacher": old_plan["expert_teacher"], "new_teacher": plan["expert_teacher"],
        "same_seed_and_scene_sequence_verified": True, "failed_scene_skipped": False,
        "failed_episode_recollected_at_same_pose": True, "original_failure_preserved": True,
        "payload_sha256": {row["file"]: sha256_file(source/row["file"]) for row in safe_rows}}
    if replay_path:
        metadata.update(repair_replay_report=str(replay_path.resolve()), repair_replay_sha256=sha256_file(replay_path))
    return safe_rows, continuation_rng, metadata


def bootstrap_repaired_exception(source, plan, reproduction_path, repair_report_path):
    """Retain ALL safe normal/recovery rows and repeat an uncommitted exception.

    Requires both exact old failure reproduction AND a safe same-scene repair.
    The RNG is regenerated over the complete declared schedule, never inferred
    from the number of rows (which includes recovery copies).
    """
    source = Path(source).resolve()
    old_plan = json.loads((source/"collection_plan.json").read_text())
    state = json.loads((source/"collection_state.json").read_text())
    rows = json.loads((source/"index.json").read_text())
    reproduced = json.loads(Path(reproduction_path).read_text())
    repaired = json.loads(Path(repair_report_path).read_text())
    keys = ("seed", "episodes_per_task", "contract_source", "contract_index_sha256", "contract_manifest_sha256",
            "validation", "validation_index_sha256", "validation_manifest_sha256", "frozen_reward_distances",
            "observation_profile_sha256", "environment_runtime_fingerprint", "pybullet_binary_sha256",
            "closed_hover_recovery_every", "planned_ordinary_episodes", "planned_closed_hover_recovery_episodes",
            "planned_episodes")
    if any(old_plan.get(key) != plan.get(key) for key in keys):
        raise ValueError("Exception repair changes physical/data/seed/recovery contract")
    if (state["status"] != "stopped_on_collection_exception" or len(rows) != state["completed_episodes"]
            or sha256_file(source/"index.json") != state["index_sha256"]
            or any(not row["success"] or row.get("safety_violation") for row in rows)
            or Path(reproduced["parent_source"]).resolve() != source
            or reproduced["state_sha256"] != sha256_file(source/"collection_state.json")
            or reproduced["index_sha256"] != state["index_sha256"]
            or reproduced.get("task") != state["task"]
            or not reproduced.get("exception") or reproduced["exception"]["message"] != state["message"]):
        raise ValueError("Exact preserved exception and immutable safe prefix required")
    repair_teachers = {"staged_yaw_clear_touch": ("yaw_clear", "yaw_clear_touch_teacher.py"),
        "staged_central_section_clearance": ("central_section_clearance", "central_section_clearance_teacher.py")}
    teacher_name, teacher_file = repair_teachers.get(plan["expert_teacher"], (None, None))
    teacher = Path(__file__).with_name(teacher_file or "unqualified_teacher").resolve()
    if (teacher_name is None
            or Path(repaired["source"]).resolve() != source or repaired["episode"] != state["file"]
            or repaired["source_state_sha256"] != sha256_file(source/"collection_state.json")
            or repaired["source_index_sha256"] != state["index_sha256"]
            or repaired["reproduction_sha256"] != sha256_file(reproduction_path)
            or repaired.get("task") != state["task"] or repaired.get("passed") is not True
            or repaired.get("reset_exact") is not True or repaired.get("runtime_changed") is not False
            or repaired.get("training_performed") is not False or repaired.get("final_test_used") is not False
            or repaired.get("teacher") != teacher_name
            or repaired["protected_hashes"].get(str(teacher)) != sha256_file(teacher)
            or not repaired["metadata"]["success"] or repaired["metadata"].get("safety_violation")):
        raise ValueError("Exact safe same-reset exception repair required")
    rng, cursor, found = np.random.default_rng(plan["seed"]), 0, False
    for task in select_tasks("all"):
        for number in range(plan["episodes_per_task"]):
            before = copy.deepcopy(rng.bit_generator.state)
            pick, place = stratified_names(task.name, number)
            sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
            sampled_json = json.loads(json.dumps(sampled.to_dict()))
            files = [f"{task.name}_{number:05d}.npz"]
            every = plan.get("closed_hover_recovery_every", 0)
            if every and (number+1)%every == 0:
                files.append(f"{task.name}_{number:05d}_closedhover.npz")
            for filename in files:
                if cursor < len(rows):
                    row = rows[cursor]
                    if row["file"] != filename or row["task"] != sampled_json:
                        raise ValueError("Exception source is not the exact contiguous declared scene sequence")
                    cursor += 1
                else:
                    if (filename != state["file"] or before != state["rng_state"]
                            or sampled_json != state["task"] or rng.bit_generator.state != state["rng_after_sampling"]):
                        raise ValueError("Exception checkpoint would skip/reroll its exact failed scene")
                    continuation_rng, found = before, True
                    break
            if found:
                break
        if found:
            break
    if not found:
        raise ValueError("No declared next failed scene after safe prefix")
    metadata = {"source": str(source), "episodes": len(rows), "index_sha256": state["index_sha256"],
        "manifest_sha256": sha256_file(source/"dataset_manifest.json"),
        "state_sha256": sha256_file(source/"collection_state.json"), "failed_file": state["file"],
        "exception_sha256": sha256_file(source/"collection_exception.json"),
        "reproduction_sha256": sha256_file(reproduction_path), "repair_report_sha256": sha256_file(repair_report_path),
        "repair_report": str(Path(repair_report_path).resolve()),
        "old_teacher": old_plan["expert_teacher"], "new_teacher": plan["expert_teacher"],
        "same_seed_and_scene_sequence_verified": True, "failed_scene_skipped": False,
        "original_exception_preserved": True,
        "payload_sha256": {row["file"]: sha256_file(source/row["file"]) for row in rows}}
    return rows, continuation_rng, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026100902)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher", choices=("collision", "neighbour_clearance", "adaptive_neighbour", "ray_refined", "geometry_clearance", "staged", "staged_obstacle_touch", "staged_open_obstacle_touch", "staged_yaw_clear_touch", "staged_central_section_clearance", "staged_pose_revalidated_acquisition"), default="collision",
                        help="Explicit expert-only ablation; never modifies learned controls or an existing candidate")
    parser.add_argument("--closed-hover-recovery-every", type=int, choices=(0, 2), default=0,
                        help="Staged only: add a separate closed-empty hover recovery on every second scene")
    parser.add_argument("--bootstrap-from", type=Path, help="Retain an exact contiguous immutable prefix in a NEW declaration")
    parser.add_argument("--bootstrap-reproduction", type=Path, help="Required exact uncommitted exception evidence")
    parser.add_argument("--bootstrap-failed-episode-from", type=Path,
                        help="NEW declaration: retain safe prefix and repeat the exact committed failed scene")
    parser.add_argument("--bootstrap-repair-report", type=Path,
                        help="Exact same-reset safe teacher repair; does not qualify the resulting dataset")
    parser.add_argument("--bootstrap-exception-from", type=Path)
    parser.add_argument("--bootstrap-exception-reproduction", type=Path)
    parser.add_argument("--bootstrap-exception-repair", type=Path)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_task <= 600:
        raise ValueError("Bounded candidate collection supports 1..600 episodes per task")
    if args.closed_hover_recovery_every and (args.teacher not in {"staged", "staged_obstacle_touch", "staged_open_obstacle_touch", "staged_yaw_clear_touch", "staged_central_section_clearance", "staged_pose_revalidated_acquisition"} or args.bootstrap_from):
        raise ValueError("Closed-hover supplementation requires staged labels, not legacy exception bootstrap")
    if bool(args.bootstrap_failed_episode_from) != bool(args.bootstrap_repair_report):
        raise ValueError("Failed-episode parent and exact repair evidence are both required")
    if args.bootstrap_failed_episode_from and (args.bootstrap_from or args.resume or args.teacher not in {"staged_obstacle_touch", "staged_open_obstacle_touch", "staged_pose_revalidated_acquisition"}):
        raise ValueError("Failed-episode bootstrap needs a NEW explicit obstacle-aware teacher declaration")
    exception_flags = (args.bootstrap_exception_from, args.bootstrap_exception_reproduction, args.bootstrap_exception_repair)
    if any(exception_flags) and (not all(exception_flags) or args.resume or args.bootstrap_from
            or args.bootstrap_failed_episode_from or args.teacher not in {"staged_yaw_clear_touch", "staged_central_section_clearance"}):
        raise ValueError("Exception repair requires all three proofs and a NEW explicit repaired teacher declaration")
    recovery_per_task = args.episodes_per_task//2 if args.closed_hover_recovery_every else 0
    planned_episodes = 4*(args.episodes_per_task+recovery_per_task)
    source, validation, output = args.contract_source.resolve(), args.validation.resolve(), args.output.resolve()
    source_audit, parent = verify_source_audit(source, args.source_audit)
    val_audit, val_manifest = verify_source_audit(validation, args.validation_audit)
    if parent.get("split") != "train" or val_manifest.get("split") not in {"val", "validation"}:
        raise ValueError("Requires frozen TRAIN contracts and VALIDATION exclusions, never TEST")
    reference = json.loads((source / "index.json").read_text())
    thresholds = frozen_thresholds(reference)
    policy_factory = CollisionPlannedTeacher
    if args.teacher in {"neighbour_clearance", "adaptive_neighbour"}:
        from VLA.neighbour_clearance_teacher import NeighbourClearanceTeacher
        policy_factory = NeighbourClearanceTeacher
    if args.teacher == "adaptive_neighbour":
        from VLA.adaptive_clearance_teacher import AdaptiveClearanceTeacher
        policy_factory = AdaptiveClearanceTeacher
    if args.teacher == "ray_refined":
        from VLA.ray_refined_clearance_teacher import RayRefinedClearanceTeacher
        policy_factory = RayRefinedClearanceTeacher
    if args.teacher == "geometry_clearance":
        from VLA.geometry_clearance_teacher import GeometryClearanceTeacher
        policy_factory = GeometryClearanceTeacher
    if args.teacher in {"staged", "staged_obstacle_touch", "staged_open_obstacle_touch"}:
        from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher
        policy_factory = StagedAcquisitionTeacher
    if args.teacher == "staged_obstacle_touch":
        from VLA.obstacle_aware_touch_teacher import ObstacleAwareTouchTeacher
        policy_factory = ObstacleAwareTouchTeacher
    if args.teacher == "staged_open_obstacle_touch":
        from VLA.obstacle_aware_touch_teacher import OpenVerifiedTouchTeacher
        policy_factory = OpenVerifiedTouchTeacher
    if args.teacher == "staged_yaw_clear_touch":
        from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher
        policy_factory = YawClearTouchTeacher
    if args.teacher == "staged_central_section_clearance":
        from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
        policy_factory = CentralSectionClearanceTeacher
    if args.teacher == "staged_pose_revalidated_acquisition":
        from VLA.pose_revalidated_acquisition_teacher import PoseRevalidatedAcquisitionTeacher
        policy_factory = PoseRevalidatedAcquisitionTeacher
    val_groups = {physical_scene_key(row["task"]) for row in json.loads((validation / "index.json").read_text())}
    parent_groups = {physical_scene_key(row["task"]) for row in reference}
    protected_paths = (Path(__file__), Path(__file__).with_name("collect_demos.py"),
                       Path(__file__).with_name("collision_planned_teacher.py"),
                       Path(__file__).with_name("collision_grasp_planner.py"),
                       Path(__file__).with_name("common") / "tasks.py")
    if args.teacher in {"neighbour_clearance", "adaptive_neighbour", "ray_refined", "geometry_clearance"}:
        protected_paths += (Path(__file__).with_name("neighbour_clearance_teacher.py"),
                            Path(__file__).with_name("clearance_correction_teacher.py"))
    if args.teacher in {"adaptive_neighbour", "ray_refined", "geometry_clearance"}:
        protected_paths += (Path(__file__).with_name("adaptive_clearance_teacher.py"),)
    if args.teacher in {"ray_refined", "geometry_clearance"}:
        protected_paths += (Path(__file__).with_name("ray_refined_clearance_teacher.py"),)
    if args.teacher == "geometry_clearance":
        protected_paths += (Path(__file__).with_name("geometry_clearance_teacher.py"),)
    if args.teacher in {"staged", "staged_obstacle_touch", "staged_open_obstacle_touch", "staged_yaw_clear_touch", "staged_central_section_clearance", "staged_pose_revalidated_acquisition"}:
        protected_paths += tuple(Path(__file__).with_name(name+".py") for name in
            ("staged_acquisition_teacher", "surface_limited_touch_teacher", "robust_touch_teacher",
             "geometry_clearance_teacher", "ray_refined_clearance_teacher", "adaptive_clearance_teacher",
             "neighbour_clearance_teacher", "clearance_correction_teacher", "offline_closed_hover_recovery"))
    if args.teacher in {"staged_obstacle_touch", "staged_open_obstacle_touch", "staged_yaw_clear_touch", "staged_central_section_clearance", "staged_pose_revalidated_acquisition"}:
        protected_paths += (Path(__file__).with_name("obstacle_aware_touch_teacher.py"),)
    if args.teacher in {"staged_yaw_clear_touch", "staged_central_section_clearance", "staged_pose_revalidated_acquisition"}:
        protected_paths += (Path(__file__).with_name("yaw_clear_touch_teacher.py"),)
    if args.teacher in {"staged_central_section_clearance", "staged_pose_revalidated_acquisition"}:
        protected_paths += (Path(__file__).with_name("central_section_clearance_teacher.py"),)
    if args.teacher == "staged_pose_revalidated_acquisition":
        protected_paths += (Path(__file__).with_name("pose_revalidated_acquisition_teacher.py"),)
    plan = {"scope": "Independent TRAIN candidate collection, NOT learned results or qualified training data",
        "contract_source": str(source), "contract_index_sha256": source_audit["index_sha256"],
        "contract_manifest_sha256": source_audit["manifest_sha256"],
        "validation": str(validation), "validation_index_sha256": val_audit["index_sha256"],
        "validation_manifest_sha256": val_audit["manifest_sha256"],
        "seed": args.seed, "episodes_per_task": args.episodes_per_task,
        "required_success_steps": 1, "frozen_reward_distances": thresholds,
        "observation_profile_sha256": parent["observation_profile_sha256"],
        "environment_runtime_fingerprint": parent["environment_runtime_fingerprint"],
        "collection_sources_sha256": {str(path.resolve()): sha256_file(path) for path in protected_paths},
        "pybullet_binary_sha256": sha256_file(Path(p.__file__)),
        "expert_teacher": args.teacher,
        "scene_selection_uses_model_or_teacher_outcomes": False,
        "source_robot_objects_cameras_safeguards_changed": False, "final_test_used": False}
    if args.closed_hover_recovery_every:
        plan.update(closed_hover_recovery_every=args.closed_hover_recovery_every,
                    planned_ordinary_episodes=4*args.episodes_per_task,
                    planned_closed_hover_recovery_episodes=4*recovery_per_task,
                    planned_episodes=planned_episodes,
                    recovery_scope="mid-height closed-empty only, NOT near-body/slip/wrong-object recovery",
                    recovery_startup_controls_are_training_labels=False)
    # JSON canonicalization makes tuples comparable to a persisted declaration.
    plan = json.loads(json.dumps(plan))
    rng, rows = np.random.default_rng(args.seed), []
    if bool(args.bootstrap_from) != bool(args.bootstrap_reproduction):
        raise ValueError("Bootstrap source and exception reproduction are both required")
    if args.bootstrap_from:
        rows, rng_state, bootstrap = bootstrap_prefix(args.bootstrap_from, plan, args.bootstrap_reproduction)
        rng.bit_generator.state = rng_state
        plan["bootstrap"] = bootstrap
    if args.bootstrap_failed_episode_from:
        rows, rng_state, bootstrap = bootstrap_repaired_prefix(args.bootstrap_failed_episode_from, plan, args.bootstrap_repair_report)
        rng.bit_generator.state = rng_state
        plan["bootstrap_repaired_failure"] = bootstrap
    if args.bootstrap_exception_from:
        rows, rng_state, bootstrap = bootstrap_repaired_exception(args.bootstrap_exception_from, plan,
            args.bootstrap_exception_reproduction, args.bootstrap_exception_repair)
        rng.bit_generator.state = rng_state
        plan["bootstrap_repaired_exception"] = bootstrap
    index_path, state_path = output / "index.json", output / "collection_state.json"
    if args.resume:
        saved = json.loads((output / "collection_plan.json").read_text())
        state = json.loads(state_path.read_text())
        if (saved != plan or state["status"] != "in_progress"
                or (output/"COLLECTION_EXCEPTION.md").exists()):
            raise ValueError("Resume contract changed, already completed, or stopped on an unresolved failure")
        rows = json.loads(index_path.read_text())
        if state["index_sha256"] != sha256_file(index_path) or len(rows) != state["completed_episodes"]:
            raise ValueError("Resume index differs from the committed checkpoint")
        if any(not row["success"] or row.get("safety_violation") is not None for row in rows):
            raise ValueError("Resolve preserved failed collection before any continuation")
        rng.bit_generator.state = state["rng_state"]
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        (output / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        if args.bootstrap_from or args.bootstrap_failed_episode_from or args.bootstrap_exception_from:
            prefix_source = args.bootstrap_from or args.bootstrap_failed_episode_from or args.bootstrap_exception_from
            for row in rows:
                os.link(prefix_source/row["file"], output/row["file"])
            index_path.write_text(json.dumps(rows, indent=2)+"\n")
            state_path.write_text(json.dumps({"status": "in_progress", "completed_episodes": len(rows),
                "planned_episodes": planned_episodes, "last_file": rows[-1]["file"] if rows else None,
                "index_sha256": sha256_file(index_path), "rng_state": rng.bit_generator.state}, indent=2)+"\n")
    manifest = copy.deepcopy(parent)
    for key in ("split", "readiness_report", "readiness_report_sha256", "training_qualification",
                "qualification_parent", "qualification_parent_index_sha256", "qualification_parent_manifest_sha256"):
        manifest.pop(key, None)
    manifest.update(candidate_only=True, training_certified=False, seed=args.seed,
        episodes_per_task=args.episodes_per_task, stratified_target_vocabulary=True,
        required_success_steps=1, position_noise_m=0., minimum_object_spacing_m=.13,
        purpose="New independent TRAIN-only candidate scenes; NOT final data sufficiency or benchmark",
        final_test_used=False, collection_plan_sha256=sha256_file(output / "collection_plan.json"))
    completed = {row["file"] for row in rows}
    seen = parent_groups | {physical_scene_key(row["task"]) for row in rows}
    with audit_environment(source, image_size=384, seed=args.seed, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        proxy = FrozenRewardReset(env, thresholds)
        for task in select_tasks("all"):
            for number in range(args.episodes_per_task):
                filename = f"{task.name}_{number:05d}.npz"
                recovery_filename = (f"{task.name}_{number:05d}_closedhover.npz"
                                     if args.closed_hover_recovery_every and (number+1)%args.closed_hover_recovery_every == 0 else None)
                if filename in completed and (recovery_filename is None or recovery_filename in completed):
                    continue
                pick, place = stratified_names(task.name, number)
                scene_rng_before = copy.deepcopy(rng.bit_generator.state)
                sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
                group = physical_scene_key(sampled.to_dict())
                partial_normal = next((row for row in rows if row["file"] == filename), None)
                if group in val_groups or (group in seen and partial_normal is None):
                    raise ValueError("Duplicate/TRAIN/VALIDATION scene encountered; no silent resampling")
                if partial_normal is not None and partial_normal["task"] != json.loads(json.dumps(sampled.to_dict())):
                    raise ValueError("Pending recovery changed its original normal scene")
                jobs = [] if filename in completed else [(filename, proxy, False)]
                if recovery_filename is not None and recovery_filename not in completed:
                    from VLA.offline_closed_hover_recovery import ClosedHoverRecoveryReset
                    jobs.append((recovery_filename, ClosedHoverRecoveryReset(proxy), True))
                for episode_name, resetter, recovery in jobs:
                    try:
                        arrays, meta = collect_episode(resetter, sampled, policy_factory=policy_factory)
                    except Exception as exc:
                        failure = {"status": "stopped_on_collection_exception", "file": episode_name,
                            "task": sampled.to_dict(), "exception_type": type(exc).__name__, "message": str(exc),
                            "rng_before_sampling": scene_rng_before, "rng_after_sampling": rng.bit_generator.state,
                            "completed_episodes": len(rows), "failed_scene_skipped": False,
                            "partial_episode_is_training_data": False}
                        (output/"collection_exception.json").write_text(json.dumps(failure, indent=2)+"\n")
                        (output/"collection_exception_traceback.txt").write_text(traceback.format_exc())
                        state_path.write_text(json.dumps({**failure, "planned_episodes": planned_episodes,
                            "index_sha256": sha256_file(index_path) if index_path.exists() else None,
                            "rng_state": scene_rng_before}, indent=2)+"\n")
                        raise
                    np.savez_compressed(output / episode_name, **arrays)
                    meta.update(file=episode_name, scene_group=group, candidate_only=True,
                        collection_pybullet_binary_sha256=plan["pybullet_binary_sha256"])
                    if recovery:
                        meta.update(variant=0, canonical_source=str(output), paired_scene_source=filename,
                            paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
                            recovery_start_actions_not_training_labels=[a.tolist() for a in resetter.startup_actions],
                            recovery_start_kind=resetter.startup_evidence["kind"],
                            recovery_start_evidence=resetter.startup_evidence)
                        checks = {"initial_measured_closed_empty": float(arrays["gripper_closure"][0]) >= .95,
                            "first_corrective_label_open": float(arrays["canonical_actions"][0, 6]) == -1,
                            "first_post_action_not_false_grasp": not bool(arrays["next_grasped_object"][0]),
                            "terminal_verified_grasp": task.name == "touch" or bool(arrays["next_verified_grasp"][-1])}
                        meta["closed_hover_recovery_checks"] = checks
                        if not all(checks.values()):
                            meta["success"] = False
                            meta["expert_quality_failure"] = "closed_hover_recovery_feedback_or_labels_failed"
                    rows.append(meta); seen.add(group)
                    index_path.write_text(json.dumps(rows, indent=2) + "\n")
                    safe = meta["success"] and meta.get("safety_violation") is None
                    status = "in_progress" if safe else "stopped_on_failed_or_unsafe_episode"
                    state_path.write_text(json.dumps({"status": status, "completed_episodes": len(rows),
                        "planned_episodes": planned_episodes, "last_file": episode_name,
                        "index_sha256": sha256_file(index_path), "rng_state": rng.bit_generator.state}, indent=2) + "\n")
                    manifest.update(episodes=len(rows), frames=sum(row["steps"] for row in rows),
                        declared_task_counts=dict(Counter(row["task"]["name"] for row in rows)),
                        index_sha256=sha256_file(index_path))
                    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    print(f"{len(rows)}/{planned_episodes} {episode_name}: success={meta['success']} steps={meta['steps']} safety={meta.get('safety_violation')}", flush=True)
                    if not safe:
                        raise RuntimeError("Failed scene/control preserved. Diagnose; do not filter it out or train from this candidate")
    verify_source_audit(source, args.source_audit)
    verify_source_audit(validation, args.validation_audit)
    verify_environment_runtime_fingerprint(plan["environment_runtime_fingerprint"], parent["observation_profile"])
    if any(sha256_file(Path(path)) != digest for path, digest in plan["collection_sources_sha256"].items()):
        raise RuntimeError("Collection implementation changed during the run")
    if plan.get("bootstrap_repaired_failure"):
        lineage = plan["bootstrap_repaired_failure"]
        prefix_source = Path(lineage["source"])
        if (sha256_file(prefix_source/"index.json") != lineage["index_sha256"]
                or sha256_file(prefix_source/"dataset_manifest.json") != lineage["manifest_sha256"]
                or sha256_file(prefix_source/"collection_state.json") != lineage["state_sha256"]
                or sha256_file(prefix_source/lineage["failed_file"]) != lineage["failed_payload_sha256"]
                or any(sha256_file(prefix_source/name) != digest or sha256_file(output/name) != digest
                       for name, digest in lineage["payload_sha256"].items())):
            raise RuntimeError("Preserved failure or hardlinked safe prefix changed during collection")
    if plan.get("bootstrap_repaired_exception"):
        lineage = plan["bootstrap_repaired_exception"]
        prefix_source = Path(lineage["source"])
        if (sha256_file(prefix_source/"index.json") != lineage["index_sha256"]
                or sha256_file(prefix_source/"dataset_manifest.json") != lineage["manifest_sha256"]
                or sha256_file(prefix_source/"collection_state.json") != lineage["state_sha256"]
                or sha256_file(prefix_source/"collection_exception.json") != lineage["exception_sha256"]
                or any(sha256_file(prefix_source/name) != digest or sha256_file(output/name) != digest
                       for name, digest in lineage["payload_sha256"].items())):
            raise RuntimeError("Preserved exception or hardlinked safe prefix changed during collection")
    state = json.loads(state_path.read_text())
    state["status"] = "completed_candidate_not_qualified"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print("Collection completed; full camera/feedback/native replay/split/conversion gates still required.", flush=True)


if __name__ == "__main__":
    main()
