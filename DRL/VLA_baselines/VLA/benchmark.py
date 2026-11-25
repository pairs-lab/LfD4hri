#!/usr/bin/env python3
"""Run one deterministic four-task VLA benchmark and save one combined video."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet
import torch

from VLA.common.environment import CAMERA_CONFIG, SEMANTIC_SUCCESS_DEFAULTS, ManipulationEnv
from VLA.common.observation_profile import ProfiledManipulationEnv
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.profile_binding import resolve_profile
from VLA.common.policy import load_policy
from VLA.common.recorder import NullVideoRecorder, VideoRecorder
from VLA.common.tasks import perturb_task, randomize_rl_scene, select_tasks


VLA_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("oracle", "pi0", "pi0_fast", "openvla", "cogact", "rdt", "smolvla"),
        required=True,
    )
    parser.add_argument("--robot", choices=("ur5e", "uf850"), default="ur5e")
    parser.add_argument("--tasks", default="all", help="all or comma-separated touch,pick,move,pick_place")
    parser.add_argument("--test-id", default="test_1")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--observation-profile", type=Path,
                        help="Versioned camera/home; learned checkpoints must bind this exact profile in source lineage (auto-loaded when available)")
    parser.add_argument("--control-hz", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--position-noise",
        type=float,
        default=0.0,
        help="Uniform XY task-pose perturbation in metres; coordinates remain hidden from the policy",
    )
    parser.add_argument(
        "--scene-profile",
        choices=("single_target", "rl_random_env"),
        default="single_target",
        help="Integration scene or the multi-object distribution from utils/random_env.py",
    )
    parser.add_argument("--execute-steps", type=int, default=1, help="Actions consumed from each predicted chunk")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=1,
        help="Internal queued actions before replanning (used by policies such as SmolVLA)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host for openpi-style adapters")
    parser.add_argument("--port", type=int, default=None, help="Policy server port (model-specific default if omitted)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--audit-observation-every", type=int, default=0,
        help="Save exact, unannotated front/wrist RGB at reset/end and every N steps (0 disables)",
    )
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--no-assist-grasp", action="store_true")
    parser.add_argument(
        "--required-success-steps",
        type=int,
        default=1,
        help="Consecutive raw-success frames required for evaluation; RL replay collection uses 10",
    )
    parser.add_argument(
        "--success-threshold-scale",
        type=float,
        default=1.25,
        help="Evaluation-only multiplier on reward.xml distance thresholds, shared by every policy",
    )
    parser.add_argument(
        "--require-active-grasp",
        action="store_true",
        help="Deprecated strict-mode alias; use --success-mode strict_reward",
    )
    parser.add_argument(
        "--success-mode",
        choices=("semantic", "reward_distance", "strict_reward"),
        default="semantic",
        help="Semantic target-region outcome (default) or legacy reward predicates",
    )
    for name, default in SEMANTIC_SUCCESS_DEFAULTS.items():
        parser.add_argument(
            "--semantic-" + name.replace("_", "-"),
            type=float,
            default=default,
        )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 encoding for development sweeps; final reported runs must retain video",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensor_profile = resolve_profile(args.model, args.checkpoint, args.observation_profile)
    if args.audit_observation_every < 0:
        raise ValueError("--audit-observation-every cannot be negative")
    if args.require_active_grasp:
        if args.success_mode != "semantic":
            raise ValueError("Use either --require-active-grasp or --success-mode, not both")
        args.success_mode = "strict_reward"
    if args.scene_profile == "rl_random_env" and args.position_noise != 0.0:
        raise ValueError("--position-noise is only valid for --scene-profile single_target")
    rng = np.random.default_rng(args.seed)
    tasks = [
        randomize_rl_scene(task, rng)
        if args.scene_profile == "rl_random_env"
        else perturb_task(task, rng, args.position_noise)
        for task in select_tasks(args.tasks)
    ]
    output_dir = VLA_ROOT / args.model / args.test_id
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "rollout.mp4"
    result_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"
    trace_path = output_dir / "trajectory.json"
    environment_class = environment_class_for_profile(sensor_profile)
    profile_kwargs = {"observation_profile": sensor_profile} if sensor_profile is not None else {}
    env = environment_class(
        **profile_kwargs,
        robot=args.robot,
        image_size=args.image_size,
        control_hz=args.control_hz,
        assist_grasp=not args.no_assist_grasp,
        seed=args.seed,
        required_success_steps=args.required_success_steps,
        success_threshold_scale=args.success_threshold_scale,
        allow_distance_only_success=args.success_mode == "reward_distance",
        use_semantic_success=args.success_mode == "semantic",
        semantic_success_thresholds={
            name: float(getattr(args, "semantic_" + name))
            for name in SEMANTIC_SUCCESS_DEFAULTS
        },
    )
    policy = load_policy(
        args.model,
        env,
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        replan_steps=args.replan_steps,
        seed=args.seed,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "robot": args.robot,
                "gripper": "Robotiq 2F-85",
                "action_contract": [
                    "dx_m",
                    "dy_m",
                    "dz_m",
                    "droll_rad",
                    "dpitch_rad",
                    "dyaw_rad",
                    "gripper_-1_open_+1_close",
                ],
                "control_hz": args.control_hz,
                "replan_steps": args.replan_steps,
                "seed": args.seed,
                "position_noise_m": args.position_noise,
                "scene_profile": args.scene_profile,
                "scene_source": "utils/random_env.py" if args.scene_profile == "rl_random_env" else None,
                "valid_scene_conditioning": (
                    "13cm clearance around commanded fixed destination"
                    if args.scene_profile == "rl_random_env"
                    else None
                ),
                "image_size": args.image_size,
                "cameras": env.observation_profile["cameras"] if sensor_profile is not None else CAMERA_CONFIG,
                "home_joints": list(env.robot_spec.home_joints),
                "observation_profile": getattr(env, "observation_profile", None),
                "observation_profile_sha256": getattr(env, "observation_profile_sha256", None),
                "task_input": "natural-language prompt",
                "observation_available": [
                    "front_rgb",
                    "wrist_rgb",
                    "6_joint_positions",
                    "gripper_state",
                    "end_effector_pose",
                ],
                "model_inputs_consumed": list(policy.input_modalities),
                "video_recorded": not args.no_video,
                "video_only_overview": bool(sensor_profile is not None and "overview" in sensor_profile["cameras"]),
                "observation_snapshot_interval": args.audit_observation_every,
                "success_mode": args.success_mode,
                "privileged_object_pose_given_to_vla": False,
                "checkpoint": _checkpoint_fingerprint(args.checkpoint),
                "reward_contract": _reward_contract(env, tasks),
                "tasks": [task.to_dict() for task in tasks],
            },
            indent=2,
        )
        + "\n"
    )
    run = {
        "model": args.model,
        "comparable": policy.comparable,
        "robot": args.robot,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tasks": [],
    }
    trajectories: dict[str, list[dict]] = {}
    try:
        recorder_context = (
            NullVideoRecorder() if args.no_video else VideoRecorder(video_path, fps=args.control_hz,
                overview_render=(lambda: env.render("overview"))
                if sensor_profile is not None and "overview" in sensor_profile["cameras"] else None)
        )
        with recorder_context as recorder:
            for task in tasks:
                print(f"Starting {args.model} | {task.name} | {task.prompt}", flush=True)
                observation = env.reset(task)
                initial_semantic_success = bool(env.is_semantic_success())
                policy.reset(task)
                if args.audit_observation_every:
                    _save_observation_snapshot(output_dir, task.name, observation, env)
                recorder.add(observation, f"{args.model} | {task.prompt}", repeat=args.control_hz)
                done = False
                success = False
                rewards: list[float] = []
                latencies: list[float] = []
                executed_actions: list[np.ndarray] = []
                task_trace: list[dict] = []
                actions_executed = 0
                error = None
                error_traceback = None
                final_info = {
                    "raw_success": False,
                    "strict_raw_success": False,
                    "semantic_raw_success": False,
                    "distance_only_success": False,
                    "success_streak": 0,
                    "grasped_object": None,
                    "touching_object": False,
                    "ee_object_distance": env.ee_object_distance(),
                    "object_goal_distance": env.object_goal_distance(),
                    **env.semantic_success_metrics(),
                    "distance_threshold": env.dist_threshold,
                    "object_threshold": env.obj_threshold,
                }
                try:
                    while not done:
                        policy_observation = _observation_telemetry(observation, env)
                        interaction_before = _interaction_audit(env)
                        if args.audit_observation_every and observation.step % args.audit_observation_every == 0:
                            _save_observation_snapshot(output_dir, task.name, observation, env)
                        started = time.perf_counter()
                        predicted = np.asarray(policy.predict(observation), dtype=np.float32)
                        latencies.append(time.perf_counter() - started)
                        if predicted.ndim == 1:
                            predicted = predicted[None, :]
                        if predicted.ndim != 2 or predicted.shape[1] != 7:
                            raise ValueError(f"Policy returned invalid action shape {predicted.shape}; expected (7,) or (H, 7)")
                        for action in predicted[: args.execute_steps]:
                            executed_actions.append(np.asarray(action, dtype=np.float32).copy())
                            observation, reward, done, info = env.step(action)
                            rewards.append(reward)
                            actions_executed += 1
                            if actions_executed % 40 == 0:
                                print(f"{task.name}: {actions_executed} actions executed", flush=True)
                            success = bool(info["success"])
                            final_info = info
                            task_trace.append(
                                {
                                    "step": actions_executed,
                                    "policy_observation": policy_observation,
                                    "result_observation": _observation_telemetry(observation, env),
                                    "interaction_before_audit_only": interaction_before,
                                    "interaction_after_audit_only": _interaction_audit(env),
                                    "action": np.asarray(action, dtype=np.float32).tolist(),
                                    "reward": float(reward),
                                    "raw_success": bool(info["raw_success"]),
                                    "strict_raw_success": bool(info["strict_raw_success"]),
                                    "semantic_raw_success": bool(info["semantic_raw_success"]),
                                    "distance_only_success": bool(info["distance_only_success"]),
                                    "success": bool(info["success"]),
                                    "success_streak": int(info["success_streak"]),
                                    "safety_violation": info.get("safety_violation"),
                                    "safety_rollback_restored": info.get("rollback_restored"),
                                    "safety_diagnostics_before_rollback": info.get("safety_diagnostics_before_rollback"),
                                    "touching_object": bool(info["touching_object"]),
                                    "grasped_object": info["grasped_object"] is not None,
                                    "ee_object_distance_m": float(info["ee_object_distance"]),
                                    "object_goal_distance_m": float(info["object_goal_distance"]),
                                    "pick_lift_m": float(info["pick_lift_m"]),
                                    "goal_xy_distance_m": info["goal_xy_distance_m"],
                                    "goal_vertical_offset_m": info["goal_vertical_offset_m"],
                                    "normalized_semantic_gap": _normalized_semantic_gap(
                                        task.name,
                                        info,
                                        env.semantic_success_thresholds,
                                    ),
                                }
                            )
                            recorder.add(observation, f"{args.model} | {task.prompt} | step {actions_executed}")
                            if done:
                                break
                except Exception as exc:  # Persist partial evidence before failing the suite.
                    error = f"{type(exc).__name__}: {exc}"
                    error_traceback = traceback.format_exc()
                    done = True
                recorder.add(
                    observation,
                    f"{args.model} | {task.name} | {'SUCCESS' if success else 'FAIL'}",
                    repeat=args.control_hz,
                )
                if args.audit_observation_every:
                    _save_observation_snapshot(output_dir, task.name, observation, env)
                trajectories[task.name] = task_trace
                print(
                    f"Finished {task.name}: success={success}, steps={actions_executed}, error={error}",
                    flush=True,
                )
                run["tasks"].append(
                    {
                        "name": task.name,
                        "prompt": task.prompt,
                        "success": success,
                        "initial_semantic_success": initial_semantic_success,
                        "steps": actions_executed,
                        "return": float(np.sum(rewards)),
                        "mean_inference_ms": None if not latencies else float(np.mean(latencies) * 1000),
                        "action_diagnostics": _action_diagnostics(executed_actions),
                        "final_ee_object_distance_m": float(final_info["ee_object_distance"]),
                        "final_object_goal_distance_m": float(final_info["object_goal_distance"]),
                        "final_raw_success": bool(final_info["raw_success"]),
                        "final_strict_raw_success": bool(final_info["strict_raw_success"]),
                        "final_semantic_raw_success": bool(final_info["semantic_raw_success"]),
                        "final_distance_only_success": bool(final_info["distance_only_success"]),
                        "final_success_streak": int(final_info["success_streak"]),
                        "safety_violation": final_info.get("safety_violation"),
                        "safety_rollback_restored": final_info.get("rollback_restored"),
                        "final_touching_object": bool(final_info["touching_object"]),
                        "final_grasped_object": final_info["grasped_object"] is not None,
                        "final_pick_lift_m": float(final_info["pick_lift_m"]),
                        "final_goal_xy_distance_m": final_info["goal_xy_distance_m"],
                        "final_goal_vertical_offset_m": final_info["goal_vertical_offset_m"],
                        "min_ee_object_distance_m": _trace_min(task_trace, "ee_object_distance_m"),
                        "min_object_goal_distance_m": _trace_min(task_trace, "object_goal_distance_m"),
                        "max_pick_lift_m": _trace_max_float(task_trace, "pick_lift_m"),
                        "min_goal_xy_distance_m": _trace_min_optional(
                            task_trace, "goal_xy_distance_m"
                        ),
                        "min_normalized_semantic_gap": _trace_min(
                            task_trace, "normalized_semantic_gap"
                        ),
                        "max_success_streak": _trace_max(task_trace, "success_streak"),
                        "ever_strict_raw_success": any(
                            item["strict_raw_success"] for item in task_trace
                        ),
                        "ever_semantic_raw_success": any(
                            item["semantic_raw_success"] for item in task_trace
                        ),
                        "touch_threshold_m": float(final_info["distance_threshold"]),
                        "object_threshold_m": float(final_info["object_threshold"]),
                        "semantic_thresholds": dict(env.semantic_success_thresholds),
                        "error": error,
                        "error_traceback": error_traceback,
                    }
                )
    finally:
        policy.close()
        env.close()
    run["success_rate"] = float(np.mean([item["success"] for item in run["tasks"]])) if run["tasks"] else 0.0
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    result_path.write_text(json.dumps(run, indent=2) + "\n")
    trace_path.write_text(json.dumps(trajectories, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print(f"Video: {video_path if not args.no_video else 'disabled for this sweep'}")
    print(f"Results: {result_path}")
    print(f"Trajectory: {trace_path}")
    return 0 if all(item["error"] is None for item in run["tasks"]) else 2


def _interaction_audit(env) -> dict:
    """Evaluator-only identities/poses; NEVER added to VLAObservation."""
    held = env.gripper.grasped_object
    state = env.debug_state()
    return {
        "requested_object": env.task.pick_name,
        "held_object": next((name for name, body in env.object_ids.items() if body == held), None),
        "correct_object_held": held == env.pick_id,
        "object_base_xyz_m": np.asarray(state["pick_position"]).tolist(),
        "object_center_xyz_m": np.asarray(state["pick_center"]).tolist(),
        "pad_grasp_center_xyz_m": np.asarray(state["grasp_center"]).tolist(),
        "goal_xyz_m": np.asarray(state["goal_position"]).tolist(),
        "privileged_policy_input": False,
    }


def _observation_telemetry(observation, env) -> dict:
    """Sensor values only; no object or goal coordinates enter policy telemetry."""
    return {
        "step": int(observation.step),
        "joint_positions_rad": observation.joint_positions.tolist(),
        "gripper_command_closure": float(env.normalized_gripper_closure()),
        # Robotiq.width() measures pad-link centre separation, not the inner
        # jaw aperture; avoid misleadingly comparing this with nominal 85mm.
        "measured_pad_center_distance_m": float(observation.gripper_width),
        "ee_position_m": observation.ee_position.tolist(),
        "ee_rotation_6d": observation.ee_rotation_6d.tolist(),
    }


def _save_observation_snapshot(output_dir, task_name, observation, env) -> None:
    from PIL import Image

    directory = output_dir / "observations" / task_name
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step_{observation.step:04d}"
    visibility = {}
    for camera, pixels in (("front", observation.front_rgb), ("wrist", observation.wrist_rgb)):
        # Save the actual model sensor frame, not the labelled/compressed video.
        Image.fromarray(pixels).save(directory / f"{stem}_{camera}.png")
        if hasattr(env, "target_mask"):
            from VLA.audit_finetune_data import resized_mask_metrics, mask_contrast

            visibility[camera] = {}
            for role in ("pick", "place"):
                mask = env.target_mask(camera, role)
                area, width, height = resized_mask_metrics(mask)
                visibility[camera][role] = {"visible_pixels_224": area, "bbox_width_224": width,
                                             "bbox_height_224": height, "rgb_local_contrast": mask_contrast(pixels, mask)}
                # Segmentation is evaluator-only evidence, never a policy input.
                Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{stem}_{camera}_audit_{role}_mask.png")
    (directory / f"{stem}.json").write_text(json.dumps({
        **_observation_telemetry(observation, env), "prompt": observation.prompt,
        "front_shape": list(observation.front_rgb.shape), "wrist_shape": list(observation.wrist_rgb.shape),
        "visibility_audit_only_not_policy_input": visibility,
    }, indent=2) + "\n")


def _action_diagnostics(actions: list[np.ndarray]) -> dict | None:
    """Compact evidence that a policy is producing usable canonical actions."""
    if not actions:
        return None
    values = np.asarray(actions, dtype=np.float32)
    translation = values[:, :3]
    rotation = values[:, 3:6]
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "mean_translation_norm_m": float(np.linalg.norm(translation, axis=1).mean()),
        "mean_rotation_norm_rad": float(np.linalg.norm(rotation, axis=1).mean()),
        "near_zero_fraction": float(np.mean(np.linalg.norm(values[:, :6], axis=1) < 1e-5)),
        "translation_clip_fraction": float(np.mean(np.any(np.abs(translation) >= 0.05, axis=1))),
        "rotation_clip_fraction": float(np.mean(np.any(np.abs(rotation) >= 0.25, axis=1))),
        "gripper_close_fraction": float(np.mean(values[:, 6] > 0.2)),
    }


def _trace_min(trace: list[dict], key: str) -> float | None:
    return None if not trace else float(min(item[key] for item in trace))


def _trace_max(trace: list[dict], key: str) -> int | None:
    return None if not trace else int(max(item[key] for item in trace))


def _trace_max_float(trace: list[dict], key: str) -> float | None:
    return None if not trace else float(max(item[key] for item in trace))


def _trace_min_optional(trace: list[dict], key: str) -> float | None:
    values = [item[key] for item in trace if item[key] is not None]
    return None if not values else float(min(values))


def _normalized_semantic_gap(
    task_name: str,
    info: dict,
    thresholds: dict[str, float],
) -> float:
    """Dimensionless distance outside the task's accepted outcome region.

    Zero means the semantic predicate is satisfied. For move/place, XY and Z
    violations from the *same frame* are combined, preventing a rollout that
    crosses the right XY while at an invalid height from winning a tie-break.
    """
    if bool(info["semantic_raw_success"]):
        return 0.0
    if task_name == "touch":
        threshold = max(float(thresholds["touch_distance_m"]), 1e-12)
        return max(float(info["ee_object_distance"]) / threshold - 1.0, 0.0)
    if task_name == "pick":
        threshold = max(float(thresholds["pick_min_lift_m"]), 1e-12)
        return max(1.0 - float(info["pick_lift_m"]) / threshold, 0.0)
    if task_name not in {"move", "pick_place"}:
        raise ValueError(f"Unsupported task {task_name!r}")
    xy = info["goal_xy_distance_m"]
    vertical = info["goal_vertical_offset_m"]
    if xy is None or vertical is None:
        return float("inf")
    prefix = "move" if task_name == "move" else "pick_place"
    radius = max(float(thresholds[f"{prefix}_xy_radius_m"]), 1e-12)
    below = float(thresholds[f"{prefix}_vertical_below_m"])
    above = float(thresholds[f"{prefix}_vertical_above_m"])
    xy_gap = max(float(xy) / radius - 1.0, 0.0)
    if float(vertical) < -below:
        vertical_gap = (-below - float(vertical)) / max(below, 1e-12)
    elif float(vertical) > above:
        vertical_gap = (float(vertical) - above) / max(above, 1e-12)
    else:
        vertical_gap = 0.0
    return float(np.hypot(xy_gap, vertical_gap))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_fingerprint(checkpoint: str | None) -> dict | None:
    if checkpoint is None:
        return None
    root = Path(checkpoint).resolve()
    if not root.is_dir():
        return {"path": str(root), "exists": False}
    cached_path = root / "benchmark_fingerprint.json"
    if cached_path.is_file():
        from VLA.fingerprint_checkpoint import layout_records, layout_sha256

        cached = json.loads(cached_path.read_text())
        current_layout = layout_sha256(layout_records(root))
        if (
            cached.get("version") != 1
            or Path(cached.get("checkpoint", "")).resolve() != root
            or cached.get("layout_sha256") != current_layout
        ):
            raise RuntimeError(
                f"Checkpoint changed after full fingerprint was created: {root}"
            )
        return {
            "path": str(root),
            "exists": True,
            "full_content_fingerprint": cached.get("content_tree_sha256"),
            "files": cached.get("files"),
            "bytes": cached.get("bytes"),
            "layout_verified": True,
            "fingerprint_report_sha256": _sha256(cached_path),
        }
    artifacts = {}
    for name in (
        "config.json",
        "train_config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        path = root / name
        if path.is_file():
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {"path": str(root), "exists": True, "artifacts": artifacts}


def _reward_contract(env: ManipulationEnv, tasks: list) -> dict:
    reward_path = VLA_ROOT.parent / "asset" / "params" / "reward.xml"
    per_task = {}
    for task in tasks:
        base_touch, base_object = env._read_reward_thresholds(task.task_id)
        per_task[task.name] = {
            "base_touch_threshold_m": base_touch,
            "base_object_goal_threshold_m": base_object,
            "effective_touch_threshold_m": base_touch * env.success_threshold_scale,
            "effective_object_goal_threshold_m": base_object * env.success_threshold_scale,
        }
    return {
        "source": str(reward_path.resolve()),
        "sha256": _sha256(reward_path),
        "required_consecutive_steps": env.required_success_steps,
        "threshold_scale": env.success_threshold_scale,
        "allow_distance_only_success": env.allow_distance_only_success,
        "use_semantic_success": env.use_semantic_success,
        "semantic_success_thresholds": dict(env.semantic_success_thresholds),
        "per_task_thresholds": per_task,
        "touch": "ee-object distance < threshold0 OR physical Robotiq contact",
        "strict_pick_move_place": "active Robotiq grasp AND object-goal distance < threshold1",
        "scored_pick_move_place": (
            "task-semantic lift/target-region predicate"
            if env.use_semantic_success
            else (
                "strict predicate OR object-goal distance < effective threshold1"
                if env.allow_distance_only_success
                else "strict predicate"
            )
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
