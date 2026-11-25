#!/usr/bin/env python3
"""Collect in-domain oracle demonstrations for target-domain VLA fine-tuning.

Each episode stores the same two RGB streams and proprioception exposed during
evaluation, plus both the shared delta-EEF action and the next absolute UR5e
joint target expected by openpi's published UR5 recipe.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from config import PICK_TARGETS, PLACE_TARGETS
from VLA.common.environment import CAMERA_CONFIG, ManipulationEnv
from VLA.common.observation_profile import ProfiledManipulationEnv, profile_hash, validate_profile
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.scripted_policy import ScriptedPolicy
from VLA.common.spatial import FIXED_DESTINATION
from VLA.common.tasks import TaskSpec, perturb_task, randomize_rl_scene, select_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("VLA/datasets/pybullet_ur5e_robotiq"))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-profile", type=Path,
                        help="Opt-in rigid camera/home JSON; new dataset required, no RL/default changes")
    parser.add_argument("--position-noise", type=float, default=0.0, help="Uniform XY perturbation in metres")
    parser.add_argument(
        "--scene-profile",
        choices=("single_target", "rl_random_env"),
        default="single_target",
        help="Use the integration scene or reproduce random_env.py object density/workspace",
    )
    parser.add_argument(
        "--stratify-targets",
        action="store_true",
        help="Cycle through all pick/destination names while retaining random_env scene sampling",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted collection from its checkpointed index/RNG state",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=1,
        help="Background NPZ compression workers; 2 overlaps rendering and compression safely",
    )
    return parser.parse_args()


def collect_episode(env: ManipulationEnv, task: TaskSpec, *, policy_factory=ScriptedPolicy) -> tuple[dict[str, np.ndarray], dict]:
    policy = policy_factory(env)
    observation = env.reset(task)
    overview_mask = None
    if hasattr(env, "observation_profile") and "overview" in env.observation_profile["cameras"]:
        env.render("overview")
        overview_mask = env.robot_mask("overview")
    policy.reset(task)
    buffers: dict[str, list] = {
        "front_rgb": [],
        "wrist_rgb": [],
        # Simulator segmentation is retained only for fail-closed camera
        # audits.  Converters never include these masks in model inputs.
        "front_pick_mask": [],
        "wrist_pick_mask": [],
        "front_place_mask": [],
        "wrist_place_mask": [],
        "front_robot_mask": [],
        "wrist_robot_mask": [],
        "joint_positions": [],
        "gripper_closure": [],
        "gripper_open": [],
        "ee_position": [],
        "ee_rotation_6d": [],
        "canonical_actions": [],
        "next_joint_targets": [],
        "next_observed_joint_positions": [],
        "next_gripper_closure": [],
        "next_requested_gripper_closure": [],
        # Post-action quantities are audit-only labels.  Dataset converters do
        # not expose them to a learned policy; they let us independently
        # reconstruct every reward.xml success decision before fine-tuning.
        "next_ee_object_distance": [],
        "next_object_goal_distance": [],
        "next_touching_object": [],
        "next_grasped_object": [],
        "next_raw_success": [],
        "next_success_streak": [],
        "rewards": [],
    }
    body_v9 = getattr(env, "observation_profile", {}).get("gripper_controller") == "body_contact_v9"
    if body_v9:
        buffers["next_bilateral_pad_contact"] = []
        buffers["next_verified_grasp"] = []
    done = False
    info = {"success": False}
    while not done:
        action = policy.predict(observation)
        buffers["front_rgb"].append(observation.front_rgb)
        buffers["wrist_rgb"].append(observation.wrist_rgb)
        buffers["front_pick_mask"].append(env.target_mask("front", "pick"))
        buffers["wrist_pick_mask"].append(env.target_mask("wrist", "pick"))
        buffers["front_place_mask"].append(env.target_mask("front", "place"))
        buffers["wrist_place_mask"].append(env.target_mask("wrist", "place"))
        buffers["front_robot_mask"].append(env.robot_mask("front"))
        buffers["wrist_robot_mask"].append(env.robot_mask("wrist"))
        buffers["joint_positions"].append(observation.joint_positions)
        buffers["gripper_closure"].append(env.normalized_gripper_closure())
        buffers["gripper_open"].append(observation.gripper_open)
        buffers["ee_position"].append(observation.ee_position)
        buffers["ee_rotation_6d"].append(observation.ee_rotation_6d)
        buffers["canonical_actions"].append(action)
        observation, reward, done, info = env.step(action)
        if "executed_canonical_action" in info:
            buffers["canonical_actions"][-1] = np.asarray(info["executed_canonical_action"], dtype=np.float32)
        buffers["next_joint_targets"].append(getattr(env, "requested_arm_targets", observation.joint_positions))
        buffers["next_observed_joint_positions"].append(observation.joint_positions)
        buffers["next_gripper_closure"].append(env.normalized_gripper_closure())
        buffers["next_requested_gripper_closure"].append(
            env.commanded_gripper_closure() if hasattr(env, "commanded_gripper_closure") else env.normalized_gripper_closure())
        buffers["next_ee_object_distance"].append(info["ee_object_distance"])
        buffers["next_object_goal_distance"].append(info["object_goal_distance"])
        buffers["next_touching_object"].append(info["touching_object"])
        buffers["next_grasped_object"].append(info["grasped_object"] == env.pick_id)
        if body_v9:
            bilateral = env.gripper._bilateral(env.pick_id)
            buffers["next_bilateral_pad_contact"].append(bilateral)
            buffers["next_verified_grasp"].append(bilateral and info["grasped_object"] == env.pick_id)
        buffers["next_raw_success"].append(info["raw_success"])
        buffers["next_success_streak"].append(info["success_streak"])
        buffers["rewards"].append(reward)
    arrays = {key: np.asarray(value) for key, value in buffers.items()}
    arrays["prompt"] = np.asarray(task.prompt)
    if overview_mask is not None:
        arrays["overview_reset_robot_mask"] = overview_mask
    metadata = {
        "task": task.to_dict(),
        "success": bool(info["success"]),
        "steps": int(len(arrays["canonical_actions"])),
        "safety_violation": info.get("safety_violation"),
        "safety_rollback_restored": info.get("rollback_restored"),
        "safety_diagnostics_before_rollback": info.get("safety_diagnostics_before_rollback"),
        "teacher_policy": type(policy).__module__+"."+type(policy).__name__,
        "action_contract": "[dx,dy,dz,droll,dpitch,dyaw,gripper_-1_open_+1_close]",
        "openpi_action_contract": "[next_q0..next_q5,next_gripper_0_open_1_closed]",
        "native_joint_action_kind": "requested_joint_position_command" if hasattr(env, "requested_arm_targets") else "measured_next_joint_position",
        "success_contract": {
            "required_consecutive_steps": env.required_success_steps,
            "touch_distance_threshold_m": env.dist_threshold,
            "object_goal_threshold_m": env.obj_threshold,
            "source": "asset/params/reward.xml",
        },
    }
    if hasattr(env, "observation_profile_sha256"):
        metadata["observation_profile_sha256"] = env.observation_profile_sha256
        metadata["gripper_state_contract"] = env.observation_profile.get("gripper_state_contract", "commanded_closure")
        metadata["native_gripper_action_kind"] = "requested_close_intent"
    if body_v9:
        metadata["safety_min_clearance_m"] = env.safety_min_clearance
        metadata["safety_max_mimic_error_rad"] = env.safety_max_mimic_error
    return arrays, metadata


def main() -> int:
    args = parse_args()
    if args.write_workers < 1:
        raise ValueError("--write-workers must be positive")
    index_path = args.output / "index.json"
    state_path = args.output / "collection_state.json"
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing non-empty output {args.output}; choose a new path or use --resume"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    sensor_profile = (validate_profile(json.loads(args.observation_profile.read_text()))
                      if args.observation_profile else None)
    sensor_hash = profile_hash(sensor_profile) if sensor_profile is not None else None
    if args.scene_profile == "rl_random_env" and args.position_noise != 0.0:
        raise ValueError("--position-noise is only valid for --scene-profile single_target")
    if args.stratify_targets and args.scene_profile != "rl_random_env":
        raise ValueError("--stratify-targets requires --scene-profile rl_random_env")
    rng = np.random.default_rng(args.seed)
    selected = select_tasks(args.tasks)
    index: list[dict] = []
    if args.resume:
        if not index_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("Resume requires checkpointed index.json and collection_state.json")
        index = json.loads(index_path.read_text())
        state = json.loads(state_path.read_text())
        if state.get("observation_profile_sha256") != sensor_hash:
            raise ValueError("Resume observation profile differs from original collection")
        rng.bit_generator.state = state["rng_state"]
    completed_files = {item["file"] for item in index}
    pending: list[tuple[Future, dict, str, dict]] = []

    def commit_oldest() -> None:
        future, metadata, filename, rng_state = pending.pop(0)
        future.result()
        metadata["file"] = filename
        index.append(metadata)
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        state_path.write_text(
            json.dumps(
                {
                    "status": "in_progress",
                    "completed_episodes": len(index),
                    "rng_state": rng_state,
                    "observation_profile_sha256": sensor_hash,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"{filename}: success={metadata['success']} steps={metadata['steps']}",
            flush=True,
        )

    with (
        (environment_class_for_profile(sensor_profile)(sensor_profile, robot="ur5e", image_size=args.image_size, seed=args.seed)
         if sensor_profile is not None else ManipulationEnv(robot="ur5e", image_size=args.image_size, seed=args.seed)) as env,
        ThreadPoolExecutor(max_workers=args.write_workers) as writer,
    ):
        for task in selected:
            for episode_number in range(args.episodes_per_task):
                filename = f"{task.name}_{episode_number:05d}.npz"
                if filename in completed_files:
                    continue
                if args.scene_profile == "rl_random_env":
                    forced_pick = None
                    forced_place = None
                    if args.stratify_targets:
                        forced_pick = list(PICK_TARGETS)[episode_number % len(PICK_TARGETS)]
                        if task.name in {"move", "pick_place"}:
                            place_names = list(PLACE_TARGETS)
                            place_index = episode_number % len(place_names)
                            while place_names[place_index] == forced_pick:
                                place_index = (place_index + 1) % len(place_names)
                            forced_place = place_names[place_index]
                    episode_task = randomize_rl_scene(
                        task,
                        rng,
                        forced_pick_name=forced_pick,
                        forced_place_name=forced_place,
                    )
                else:
                    episode_task = perturb_task(task, rng, args.position_noise)
                teacher = ScriptedPolicy
                if sensor_profile and sensor_profile.get("gripper_controller") == "body_contact_v9":
                    from VLA.collision_planned_teacher import CollisionPlannedTeacher
                    teacher = CollisionPlannedTeacher
                arrays, metadata = collect_episode(env, episode_task, policy_factory=teacher)
                pending.append(
                    (
                        writer.submit(np.savez_compressed, args.output / filename, **arrays),
                        metadata,
                        filename,
                        copy.deepcopy(rng.bit_generator.state),
                    )
                )
                if len(pending) >= args.write_workers:
                    commit_oldest()
        while pending:
            commit_oldest()
    (args.output / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "robot": "ur5e",
                "robot_urdf": "asset/ur5e/ur5e.urdf",
                "gripper": "Robotiq 2F-85",
                "gripper_asset": ("VLA/assets/body_grasp_v9/robotiq_2f_85"
                                  if sensor_profile and sensor_profile.get("gripper_controller") == "body_contact_v9"
                                  else "asset/robotiq_2f_85"),
                "image_size": args.image_size,
                "physics_hz": 240,
                "control_hz": 10,
                "required_success_steps": env.required_success_steps,
                "seed": args.seed,
                "position_noise_m": args.position_noise,
                "scene_profile": args.scene_profile,
                "scene_source": "utils/random_env.py" if args.scene_profile == "rl_random_env" else None,
                "stratified_target_vocabulary": bool(args.stratify_targets),
                "pick_count_range": [3, 4] if args.scene_profile == "rl_random_env" else [1, 1],
                "place_count_range": [2, 3] if args.scene_profile == "rl_random_env" else [0, 1],
                "minimum_object_spacing_m": 0.13 if args.scene_profile == "rl_random_env" else None,
                "fixed_destination_mapping": FIXED_DESTINATION,
                "spatial_language_contract": "VLA/common/spatial.py; named region and coordinate are one-to-one",
                "valid_scene_conditioning": (
                    "13cm clearance around commanded fixed destination"
                    if args.scene_profile == "rl_random_env"
                    else None
                ),
                "episodes_per_task": args.episodes_per_task,
                "cameras": sensor_profile["cameras"] if sensor_profile else CAMERA_CONFIG,
                "home_joints": list(env.robot_spec.home_joints),
                "observation_profile": sensor_profile,
                "observation_profile_sha256": sensor_hash,
                "front_camera_role": sensor_profile.get("front_camera_role", "robot_overview") if sensor_profile else "robot_overview",
                "overview_supplied_to_policy": False,
                "policy_task_input": "natural-language prompt only",
                "privileged_pose_in_policy_input": False,
                "candidate_only": bool(sensor_profile and sensor_profile.get("gripper_controller") == "body_contact_v9"),
                "gripper_state_contract": sensor_profile.get("gripper_state_contract", "commanded_closure") if sensor_profile else "commanded_closure",
                "native_gripper_action_kind": "requested_close_intent",
                "native_joint_action_kind": ("requested_joint_position_command"
                                             if sensor_profile is not None else "measured_next_joint_position"),
                "action_contract": "[dx,dy,dz,droll,dpitch,dyaw,gripper_-1_open_+1_close]",
            },
            indent=2,
        )
        + "\n"
    )
    state_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_episodes": len(index),
                "rng_state": rng.bit_generator.state,
                "observation_profile_sha256": sensor_hash,
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if all(item["success"] for item in index) else 2


if __name__ == "__main__":
    raise SystemExit(main())
