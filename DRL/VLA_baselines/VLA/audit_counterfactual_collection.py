"""Collect a source-only paired-instruction pilot, preserving scene/robot/assets.

This is teacher/data-pipeline evidence, not a VLA evaluation. Production data
and final-test seeds are never changed or used here.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw
from config import PLACE_TARGETS, INSTRUCTION_FORM

from VLA.collect_demos import collect_episode
from VLA.common.environment import ManipulationEnv
from VLA.common.observation_profile import validate_profile
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import SceneObjectSpec, TaskSpec
from VLA.common.spatial import FIXED_DESTINATION
from VLA.common.training_selection import select_probe_rows
from VLA.common.provenance import sha256_file


def physical_scene_spec(task):
    objects = {task.pick_name: (tuple(task.pick_position), False)}
    if task.place_name is not None and task.place_name not in FIXED_DESTINATION:
        objects[task.place_name] = (tuple(task.place_position), task.place_name.endswith(" bowl"))
    for item in task.scene_objects:
        if item.name in objects:
            raise ValueError("Duplicate physical scene object")
        objects[item.name] = (tuple(item.position), item.fixed)
    return objects


class CanonicalSceneResetEnvironment:
    """Collection-only proxy: stable spawn order, then retarget teacher labels.

    Ordinary reset spawns the requested object first. Changing which object
    is requested therefore changes body/constraint ordering and settling.
    Reuse one canonical task for spawning both branches; retarget only after
    the SAME physical reset. No benchmark environment code is changed.
    """
    def __init__(self, env, canonical_task, startup_actions=()):
        self.env, self.canonical_task = env, canonical_task
        self.startup_actions = startup_actions

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        if physical_scene_spec(task) != physical_scene_spec(self.canonical_task):
            raise ValueError("Paired task changed a physical object, pose or fixed/dynamic flag")
        if (task.name, task.task_id) != (self.canonical_task.name, self.canonical_task.task_id):
            raise ValueError("Paired retargeting cannot change the primitive")
        self.env.reset(self.canonical_task)
        self.env.task = task
        self.env.pick_id = self.env.object_ids[task.pick_name]
        self.env.place_id = self.env.object_ids.get(task.place_name)
        self.env.initial_place_top = self.env._body_top(self.env.place_id) if self.env.place_id is not None else None
        if getattr(self.env, "observation_profile", {}).get("grasp_candidate_scope") != "all_dynamic_scene_objects":
            self.env.gripper.object_ids = [self.env.pick_id]
        position = p.getBasePositionAndOrientation(self.env.pick_id)[0]
        self.env.initial_pick_z = position[2]
        self.env.initial_pick_position = np.asarray(position, dtype=np.float64)
        self.env.initial_pick_top = self.env._pick_top()
        self.env.initial_pick_bottom = self.env._pick_bottom()
        for action in self.startup_actions:
            _, _, done, info = self.env.step(action)
            if done or info.get("grasped_object") is not None:
                raise RuntimeError("Recovery perturbation terminated/grasped before teacher collection")
        if len(self.startup_actions):
            # The demonstration starts at this off-path state; perturbation
            # commands are NOT expert labels and never enter the dataset.
            self.env.step_count = 0
            self.env.success_streak = 0
            self.env._last_success = False
            position = p.getBasePositionAndOrientation(self.env.pick_id)[0]
            self.env.initial_pick_z = position[2]
            self.env.initial_pick_position = np.asarray(position, dtype=np.float64)
            self.env.initial_pick_top = self.env._pick_top()
            self.env.initial_pick_bottom = self.env._pick_bottom()
        return self.env.observe()


def alternate_target(task, choice=0):
    candidates = [item for item in task.scene_objects if item.source_role == "pick" and not item.fixed]
    if not candidates:
        raise ValueError("Scene contains no alternate graspable target")
    candidates.sort(key=lambda item: (not item.name.endswith(" block"), item.name))
    chosen = candidates[choice % len(candidates)]
    old_target = SceneObjectSpec(task.pick_name, task.pick_position, False, "pick")
    return replace(task, pick_name=chosen.name, pick_position=chosen.position,
                   prompt=task.prompt.replace(task.pick_name, chosen.name),
                   scene_objects=tuple(item for item in task.scene_objects if item != chosen) + (old_target,))


def alternate_destination(task):
    """Retarget only an already configured goal; preserve every physical body."""
    if task.name not in ("move", "pick_place"):
        raise ValueError("Destination intervention is only defined for move/place")
    positions = {item.name: item.position for item in task.scene_objects}
    choices = [name for name in task.config_place_items if name != task.place_name]
    if not choices:
        raise ValueError("No alternative configured destination")
    def position(name):
        return FIXED_DESTINATION[name] if name in FIXED_DESTINATION else positions[name]
    # Avoid nearly overlapping regions when an existing more distant goal is
    # available. Selection is geometry-only, BEFORE model inference/results.
    name = max(choices, key=lambda item: float(np.linalg.norm(
        np.asarray(position(item))[:2] - np.asarray(task.place_position)[:2])))
    objects = tuple(item for item in task.scene_objects if item.name != name)
    if task.place_name not in FIXED_DESTINATION:
        objects += (SceneObjectSpec(task.place_name, task.place_position,
                                    task.place_name.endswith(" bowl"), "place"),)
    result = replace(task, place_name=name, place_position=tuple(position(name)),
                     config_selected_place=name, scene_objects=objects,
                     prompt=INSTRUCTION_FORM[task.task_id].format(task.pick_name, name))
    if physical_scene_spec(result) != physical_scene_spec(task):
        raise ValueError("Destination intervention changed the physical scene")
    return result


def recovery_visibility_names(task, object_ids):
    """All configured physical picks AND destinations, not selected task IDs."""
    names = tuple(dict.fromkeys((*task.config_pick_items, *task.config_place_items)))
    names = tuple(name for name in names if name not in FIXED_DESTINATION)
    missing = set(names) - set(object_ids)
    if missing:
        raise ValueError(f"Configured recovery objects are absent: {sorted(missing)}")
    return names


def closed_empty_workspace_start(env, canonical_task):
    """Collection-only recovery pose chosen from ALL bodies, not target ID."""
    from VLA.audit_finetune_data import resized_mask_metrics, mask_contrast
    env.reset(canonical_task)
    visibility_names = recovery_visibility_names(canonical_task, env.object_ids)
    bounds = [p.getAABB(body) for body in env.object_ids.values()]
    minimum = np.asarray([b[0][:2] for b in bounds])
    maximum = np.asarray([b[1][:2] for b in bounds])
    candidates = [(x, y) for x in np.linspace(-.22, .22, 13) for y in np.linspace(-.72, -.28, 13)]
    def clearance(xy):
        distances = np.maximum(np.maximum(minimum - xy, xy - maximum), 0)
        return float(np.linalg.norm(distances, axis=1).min())
    candidates.sort(key=clearance, reverse=True)
    separated = []
    for xy in candidates:
        if clearance(xy) >= .08 and all(np.linalg.norm(np.asarray(xy) - other) >= .10 for other in separated):
            separated.append(xy)
        if len(separated) == 4:
            break
    if not separated:
        raise RuntimeError("No sufficiently empty recovery location; preserve failure evidence, do not move objects")
    render_before = env.render_observations
    try:
        # Audit ALL possible picks/destinations, never adapt to a selected ID.
        # Suppress only discarded planning renders; collected observations are
        # rendered normally during canonical replay of the chosen controls.
        env.render_observations = False
        for height in (.08, .16, .24, .32):
            for xy in separated:
                env.reset(canonical_task)
                actions = []
                unsafe = False
                target = np.asarray([*xy, height])
                for _ in range(40):
                    delta = target - env.gripper.grasp_center()
                    if np.linalg.norm(delta) < .012:
                        break
                    action = np.r_[np.clip(.65 * delta, -.045, .045), np.zeros(3), -1].astype(np.float32)
                    _, _, done, info = env.step(action)
                    if done or info["grasped_object"] is not None:
                        unsafe = True
                        break
                    actions.append(action)
                else:
                    continue
                if unsafe:
                    continue
                close = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                closed_ready = False
                for _ in range(15):
                    _, _, done, info = env.step(close)
                    if done or info["grasped_object"] is not None:
                        break
                    actions.append(np.asarray(info.get("executed_canonical_action", close), dtype=np.float32))
                    # One 100ms CLOSE only moves the velocity-limited motor
                    # about0.15rad, not its full0.78rad stroke. Commanded=closed
                    # is not physical closed-empty feedback.
                    actual_angle = p.getJointState(env.gripper.body, 1)[0]
                    feedback_ready = (not hasattr(env, "measured_gripper_closure")
                                      or env.measured_gripper_closure() >= .95)
                    if actual_angle >= .95*env.gripper.CLOSED_ANGLE and feedback_ready:
                        closed_ready = True
                        break
                if not closed_ready:
                    continue
                rgb = env.render("front")
                packed = env._last_segmentation["front"]
                body_ids = np.where(packed >= 0, packed & ((1 << 24) - 1), -1)
                visible = True
                for name in visibility_names:
                    mask = body_ids == env.object_ids[name]
                    pixels, width, height_px = resized_mask_metrics(mask)
                    if pixels < 48 or min(width, height_px) < 7 or mask_contrast(rgb, mask) < 25:
                        visible = False
                        break
                if visible:
                    return actions
        raise RuntimeError("No collision-free closed-empty pose retained clear front observations for all physical pick/place candidates")
    finally:
        env.render_observations = render_before


def trajectory_action_difference(first, second):
    """Different targets can share their first velocity-saturated move."""
    length = min(len(first), len(second))
    return {"common_prefix_max_action_difference": float(np.max(np.abs(first[:length] - second[:length]))),
            "different_trajectory_lengths": len(first) != len(second)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observation-profile", type=Path,
                        help="Versioned camera/home; preserve existing production data")
    parser.add_argument("--layouts-per-task", type=int, default=1)
    parser.add_argument("--include-fixed-destinations", action="store_true",
                        help="Add missing named-region training layouts for move/place")
    parser.add_argument("--include-all-destinations", action="store_true",
                        help="Also cover every physical destination, not only named regions")
    parser.add_argument("--source-episodes", help="Optional comma-separated training episode files for regression probes")
    parser.add_argument("--recovery-start-steps", type=int, default=0,
                        help="Collect corrective expert trajectories from small off-path robot states; bad-start commands excluded")
    parser.add_argument("--closed-empty-workspace-start", action="store_true",
                        help="Teacher-only corrective data from a task-independent empty near-table pose with closed gripper")
    parser.add_argument("--closed-empty-near-object-start", action="store_true",
                        help="Teacher-only corrective data after an audited collision-free near-object missed close")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.layouts_per_task < 1:
        raise ValueError("layouts-per-task must be positive")
    if not 0 <= args.recovery_start_steps <= 2:
        raise ValueError("This audited perturbation pilot supports only zero, one or two steps")
    if sum((args.closed_empty_workspace_start, args.closed_empty_near_object_start,
            bool(args.recovery_start_steps))) > 1:
        raise ValueError("Select one recovery protocol, not multiple")
    args.output.mkdir(parents=True)
    source_index = json.loads((args.source / "index.json").read_text())
    index, pairs, cells = [], [], []
    profile = (validate_profile(json.loads(args.observation_profile.read_text()))
               if args.observation_profile else None)
    environment_class = environment_class_for_profile(profile)
    profile_kwargs = {"observation_profile": profile} if profile is not None else {}
    with environment_class(**profile_kwargs, robot="ur5e", image_size=384, seed=20261007,
                         required_success_steps=1) as env:
        selected = []
        selected_files = set(args.source_episodes.split(",")) if args.source_episodes else None
        for task_name in ("touch", "pick", "move", "pick_place"):
            candidates = [item for item in source_index if item["task"]["name"] == task_name]
            if selected_files is not None:
                candidates = [item for item in candidates if item["file"] in selected_files]
            if len(candidates) < args.layouts_per_task:
                raise ValueError(f"Not enough source layouts for {task_name}")
            chosen = select_probe_rows(candidates, args.layouts_per_task, args.include_fixed_destinations,
                                       tuple(PLACE_TARGETS) if args.include_all_destinations else None)
            selected.extend((task_name, number, item) for number, item in enumerate(chosen))
        planned_episodes = 2 * len(selected)
        for task_name, layout_number, item in selected:
            first = TaskSpec.from_dict(item["task"])
            startup_actions = []
            startup_evidence = None
            if args.closed_empty_workspace_start:
                startup_actions = closed_empty_workspace_start(env, first)
            elif args.closed_empty_near_object_start:
                from VLA.near_object_recovery_start import closed_empty_near_object_start
                startup_actions, startup_evidence = closed_empty_near_object_start(env, first)
            elif args.recovery_start_steps:
                rng = np.random.default_rng(20261007 + first.task_id * 10000 + layout_number)
                angle, distance = rng.uniform(0, 2*np.pi), rng.uniform(.015, .025)
                action = np.asarray([distance*np.cos(angle), distance*np.sin(angle), rng.uniform(-.005, .005),
                                     0., 0., 0., -1.], dtype=np.float32)
                startup_actions = [action.copy() for _ in range(args.recovery_start_steps)]
            collection_env = CanonicalSceneResetEnvironment(env, first, startup_actions)
            if startup_evidence is not None:
                from VLA.near_object_recovery_start import validated_near_object_reset_environment
                collection_env = validated_near_object_reset_environment(env, first, startup_actions, startup_evidence)
            from VLA.recovery_teacher import RecoveryTeacher
            from VLA.common.scripted_policy import ScriptedPolicy
            teacher = RecoveryTeacher if (args.closed_empty_workspace_start or args.closed_empty_near_object_start
                                          or args.recovery_start_steps) else ScriptedPolicy
            if profile and profile.get("gripper_controller") == "body_contact_v9":
                from VLA.collision_planned_teacher import CollisionPlannedTeacher
                teacher = CollisionPlannedTeacher
            pair = []
            for variant, task in enumerate((first, alternate_target(first, choice=layout_number))):
                arrays, metadata = collect_episode(collection_env, task, policy_factory=teacher)
                filename = (f"{task_name}_variant{variant}.npz" if args.layouts_per_task == 1
                            else f"{task_name}_layout{layout_number:04d}_variant{variant}.npz")
                np.savez_compressed(args.output / filename, **arrays)
                metadata.update(file=filename, paired_scene_source=item["file"], variant=variant,
                                training_data_candidate_only=True)
                metadata["paired_reset_strategy"] = "canonical_spawn_order_then_teacher_retarget"
                metadata["teacher_policy"] = teacher.__module__+"."+teacher.__name__
                metadata["recovery_start_actions_not_training_labels"] = [action.tolist() for action in startup_actions]
                if startup_evidence is not None:
                    metadata["near_object_recovery_start_evidence"] = startup_evidence
                    metadata["near_object_recovery_replay_evidence"] = collection_env.near_object_recovery_replay_evidence
                    metadata["near_object_recovery_checks"] = {
                        "initial_actual_closure_at_least_95pct": bool(arrays["gripper_closure"][0] >= .95),
                        "first_expert_label_reopens": bool(arrays["canonical_actions"][0, -1] == -1
                                                          and arrays["next_requested_gripper_closure"][0] == 0),
                        "no_false_grasp_on_first_action": bool(not arrays["next_verified_grasp"][0]),
                        "verified_carry_at_task_end": bool(task.name == "touch" or (
                            arrays["next_verified_grasp"][-1] and arrays["next_bilateral_pad_contact"][-1])),
                        "no_safety_abort": metadata["safety_violation"] is None,
                    }
                metadata["initial_joint_distance_from_default_home_rad"] = float(np.linalg.norm(
                    arrays["joint_positions"][0] - np.asarray(env.robot_spec.home_joints)))
                index.append(metadata)
                pair.append(arrays)
                cell = Image.new("RGB", (448, 260), "black")
                for view_index, name in enumerate(("front_rgb", "wrist_rgb")):
                    cell.paste(Image.fromarray(arrays[name][0]).resize((224, 224)), (view_index * 224, 36))
                ImageDraw.Draw(cell).text((4, 3), task.prompt, fill="white")
                ImageDraw.Draw(cell).text((4, 18), f"expert success={metadata['success']} | steps={metadata['steps']}", fill="white")
                if layout_number < 2:
                    cells.append(cell)
                print(f"{filename}: success={metadata['success']} steps={metadata['steps']}", flush=True)
            pairs.append({
                "task": task_name, "source_episode": item["file"],
                "front_initial_max_pixel_error": int(np.max(np.abs(pair[0]["front_rgb"][0].astype(int) - pair[1]["front_rgb"][0].astype(int)))),
                "wrist_initial_max_pixel_error": int(np.max(np.abs(pair[0]["wrist_rgb"][0].astype(int) - pair[1]["wrist_rgb"][0].astype(int)))),
                "initial_joint_max_abs_error": float(np.max(np.abs(pair[0]["joint_positions"][0] - pair[1]["joint_positions"][0]))),
                "first_action_translation_difference_m": float(np.linalg.norm(pair[0]["canonical_actions"][0, :3] - pair[1]["canonical_actions"][0, :3])),
                "both_expert_success": all(item["success"] for item in index[-2:]),
                **trajectory_action_difference(pair[0]["canonical_actions"], pair[1]["canonical_actions"]),
            })
            # Preserve progress/evidence even if a later candidate fails.
            (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
            (args.output / "collection_state.json").write_text(json.dumps({
                "status": "in_progress", "completed_episodes": len(index),
                "planned_episodes": planned_episodes,
                "source": str(args.source.resolve()), "production_dataset_changed": False,
                "observation_profile_sha256": getattr(env, "observation_profile_sha256", None),
            }, indent=2) + "\n")
    sheet = Image.new("RGB", (896, 260 * ((len(cells) + 1) // 2)), "black")
    for cell_index, cell in enumerate(cells):
        sheet.paste(cell, ((cell_index % 2) * 448, (cell_index // 2) * 260))
    sheet.save(args.output / "initial_model_observations.png")
    report = {
        "passed": all(item["both_expert_success"] and item["front_initial_max_pixel_error"] == 0
                      and item["wrist_initial_max_pixel_error"] == 0 and item["initial_joint_max_abs_error"] == 0
                      and (item["common_prefix_max_action_difference"] > .001 or item["different_trajectory_lengths"])
                      for item in pairs),
        "source": str(args.source.resolve()), "episodes": len(index), "pairs": pairs,
        "source_index_sha256": sha256_file(args.source / "index.json"),
        "includes_all_named_regions": bool(args.include_fixed_destinations or args.include_all_destinations),
        "includes_all_destinations": bool(args.include_all_destinations),
        "native_joint_action_kind": index[0]["native_joint_action_kind"],
        "grasp_candidate_scope": getattr(env, "observation_profile", {}).get("grasp_candidate_scope", "selected_target_legacy"),
        "layouts_per_task": args.layouts_per_task,
        "paired_reset_strategy": "canonical_spawn_order_then_teacher_retarget",
        "recovery_start_steps": "per_episode" if (args.closed_empty_workspace_start or args.closed_empty_near_object_start) else args.recovery_start_steps,
        "recovery_start_kind": ("closed_empty_near_object_missed_close" if args.closed_empty_near_object_start
                                else "closed_empty_workspace" if args.closed_empty_workspace_start
                                else "open_translation_perturbation"),
        "startup_action_count_range": [min(len(r["recovery_start_actions_not_training_labels"]) for r in index),
                                       max(len(r["recovery_start_actions_not_training_labels"]) for r in index)],
        "observation_profile_sha256": getattr(env, "observation_profile_sha256", None),
        "observation_profile": getattr(env, "observation_profile", None),
        "scope": "Teacher paired-instruction collection pilot only; not VLA success rate or sufficiently large training dataset",
        "production_dataset_changed": False, "final_test_used": False,
        "policy_task_input": "text only; target IDs/masks/poses belong to teacher and audit, not the learner",
        "known_missing_checks": ["Full camera/conversion/native replay audits before any training use", "Recovery-trajectory collection", "Broader target/destination/layout coverage"],
    }
    if args.closed_empty_near_object_start:
        report["near_object_recovery_checks_passed"] = all(
            all(row["near_object_recovery_checks"].values()) for row in index)
        report["passed"] = report["passed"] and report["near_object_recovery_checks_passed"]
        report["recovery_scope"] = ("Deliberate physically closed-empty missed close beside a configured pick; "
                                    "not contact-slip/all natural failed grasps, not runtime oracle fallback")
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "collection_state.json").write_text(json.dumps({
        "status": "completed_candidate_audit_passed" if report["passed"] else "completed_candidate_audit_failed",
        "completed_episodes": len(index), "planned_episodes": planned_episodes,
        "observation_profile_sha256": getattr(env, "observation_profile_sha256", None),
        "production_dataset_changed": False, "not_qualified_for_training_yet": True,
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
