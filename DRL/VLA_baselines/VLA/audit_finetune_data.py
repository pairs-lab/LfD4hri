#!/usr/bin/env python3
"""Fail-closed audit of PyBullet VLA demonstrations and camera observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from config import BOUNDS, INSTRUCTION_FORM, PICK_TARGETS, PLACE_TARGETS
from VLA.common.spatial import FIXED_DESTINATION
from VLA.common.tasks import TASKS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# A 7x7 nominal target spans 49 pixels before rasterization.  Square resize
# and the >=128 binary coverage test can drop one boundary pixel while its
# actual bounding dimensions remain >=7.  Keep a one-pixel raster tolerance;
# the independent dimension, contrast and synchronized-visibility gates still
# apply.  This value is shared with the evaluation-observation audit.
MIN_FRONT_RESET_TARGET_PIXELS_224 = 48


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-per-task", type=int, default=50)
    parser.add_argument("--expected-task-counts", help="Exact JSON counts for a declared nonuniform diagnostic pilot; default balanced gate unchanged")
    parser.add_argument("--allow-stricter-frozen-reward-contract", action="store_true",
                        help="Audit immutable historical labels under their recorded, no-looser-than-live thresholds; report live mismatch, never relabel")
    parser.add_argument("--recovery-reset-observability", choices=("front", "dual_view"), default="front",
                        help="Explicit dual-camera model contract for actual closed-empty recovery resets only; ordinary/front-only gates unchanged")
    return parser.parse_args()


def reset_view_quality(metrics):
    """ALL geometry/contrast criteria must hold in ONE real camera."""
    return (metrics["raw_pixels"] >= 25 and
            metrics["pixels_224"] >= MIN_FRONT_RESET_TARGET_PIXELS_224 and
            metrics["minimum_dimension_224"] >= 7 and metrics["contrast"] >= 25.)


def reset_observability_failures(samples, *, dual_recovery=False):
    failures = []
    for sample in samples:
        views = sample["views"]
        allowed = ("front", "wrist") if dual_recovery and sample["closed_empty_recovery"] else ("front",)
        if not any(view in views and reset_view_quality(views[view]) for view in allowed):
            failures.append(f"{sample['file']}: reset object lacks size/dimension/contrast in one required camera ({','.join(allowed)})")
    return failures


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    rgb = image.astype(np.int16)
    r, g, b = (rgb[..., i] for i in range(3))
    if color == "red":
        return (r > 130) & (r > g * 1.35) & (r > b * 1.35)
    if color == "blue":
        return (b > 90) & (b > r * 1.35) & (b > g * 1.12)
    if color == "green":
        return (g > 95) & (g > r * 1.25) & (g > b * 1.12)
    if color == "yellow":
        return (r > 150) & (g > 120) & (b < 130) & (r > b * 1.35)
    raise ValueError(color)


def focus_score(image: np.ndarray) -> float:
    gray = image.astype(np.float32).mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float((np.mean(gx * gx) + np.mean(gy * gy)) * 0.5)


def resized_mask_metrics(mask: np.ndarray, size: int = 224) -> tuple[int, int, int]:
    """Conservative target area/bounds after the actual square VLA resize."""
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (size, size), resample=Image.Resampling.BOX
        )
    ) >= 128
    rows, cols = np.nonzero(resized)
    if not len(rows):
        return 0, 0, 0
    return int(resized.sum()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def mask_contrast(image: np.ndarray, mask: np.ndarray) -> float:
    """RGB distance from the target to a local ring around it."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    pad = 6
    y0, y1 = max(0, int(rows.min()) - pad), min(mask.shape[0], int(rows.max()) + pad + 1)
    x0, x1 = max(0, int(cols.min()) - pad), min(mask.shape[1], int(cols.max()) + pad + 1)
    ring = np.zeros_like(mask, dtype=bool)
    ring[y0:y1, x0:x1] = True
    ring &= ~mask
    if not ring.any():
        return 0.0
    target_mean = image[mask].astype(np.float32).mean(axis=0)
    ring_mean = image[ring].astype(np.float32).mean(axis=0)
    return float(np.linalg.norm(target_mean - ring_mean))


def border_margin(mask: np.ndarray) -> int:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return -1
    return int(
        min(rows.min(), cols.min(), mask.shape[0] - 1 - rows.max(), mask.shape[1] - 1 - cols.max())
    )


def add_label(image: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return image


def save_contact_sheets(dataset: Path, grouped: dict[str, list[Path]], output: Path, *, task_names=None) -> None:
    # Optional subset is an IMAGE EXPORT feature only. main() retains all
    # four-task positive-count qualification gates unchanged.
    task_names = tuple(TASKS) if task_names is None else tuple(task_names)
    if not task_names or set(task_names) - set(TASKS) or any(not grouped.get(name) for name in task_names):
        raise ValueError("Image export requires explicit known nonempty task groups")
    panels: list[Image.Image] = []
    model_panels: list[Image.Image] = []
    for task_name in task_names:
        path = grouped[task_name][len(grouped[task_name]) // 2]
        with np.load(path) as episode:
            length = len(episode["canonical_actions"])
            for phase, step in (("start", 0), ("middle", length // 2), ("end", length - 1)):
                front = Image.fromarray(episode["front_rgb"][step]).resize((256, 256))
                wrist = Image.fromarray(episode["wrist_rgb"][step]).resize((256, 256))
                pair = Image.new("RGB", (512, 256))
                pair.paste(front, (0, 0))
                pair.paste(wrist, (256, 0))
                panels.append(add_label(pair, f"{task_name} | {phase} | front / wrist"))
            front_224 = Image.fromarray(episode["front_rgb"][0]).resize((224, 224))
            wrist_224 = Image.fromarray(episode["wrist_rgb"][0]).resize((224, 224))
            pair_224 = Image.new("RGB", (448, 224))
            pair_224.paste(front_224, (0, 0))
            pair_224.paste(wrist_224, (224, 0))
            model_panels.append(add_label(pair_224, f"{task_name} | 224px model input"))

    sheet = Image.new("RGB", (512 * 3, 256 * len(task_names)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 512, (index // 3) * 256))
    sheet.save(output / "camera_contact_sheet.png")

    model_sheet = Image.new("RGB", (448, 224 * len(task_names)), "white")
    for index, panel in enumerate(model_panels):
        model_sheet.paste(panel, (0, index * 224))
    model_sheet.save(output / "model_input_224_contact_sheet.png")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.dataset / "index.json").read_text())
    manifest = json.loads((args.dataset / "dataset_manifest.json").read_text())
    from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint
    payload_fingerprint = source_payload_fingerprint(args.dataset, index)
    body_v9 = (manifest.get("observation_profile") or {}).get("gripper_controller") == "body_contact_v9"
    scene_profile = manifest.get("scene_profile", "single_target")
    if args.recovery_reset_observability == "dual_view" and (not body_v9 or scene_profile != "rl_random_env"):
        raise ValueError("Dual recovery reset contract requires measured-feedback V9 RL-density data")
    required_success_steps = int(manifest.get("required_success_steps", 10))
    joint_action_kind = manifest.get("native_joint_action_kind", "measured_next_joint_position")
    if joint_action_kind not in ("measured_next_joint_position", "requested_joint_position_command"):
        raise ValueError("Unknown native joint-action representation")
    front_role = manifest.get("front_camera_role", "robot_overview")
    if front_role not in ("robot_overview", "policy_workspace"):
        raise ValueError("Unknown camera role")
    policy_workspace_camera = front_role == "policy_workspace"
    if policy_workspace_camera and ("overview" not in manifest["cameras"] or manifest.get("overview_supplied_to_policy") is not False):
        raise ValueError("Workspace camera must declare an independent video-only overview")
    if required_success_steps < 1:
        raise ValueError("dataset manifest required_success_steps must be positive")
    reward_path = PROJECT_ROOT / "asset" / "params" / "reward.xml"
    reward_sha256 = sha256(reward_path)
    reward_root = ET.parse(reward_path).getroot()
    frozen_contracts = defaultdict(set)
    live_reward_contract_matches = True
    grouped: dict[str, list[Path]] = defaultdict(list)
    failures: list[str] = []
    task_counts = Counter(item["task"]["name"] for item in index)
    overview_margins = []
    expected_counts = (json.loads(args.expected_task_counts) if args.expected_task_counts
                       else {name: args.expected_per_task for name in TASKS})
    if set(expected_counts) != set(TASKS) or any(type(v) is not int or v < 1 for v in expected_counts.values()):
        raise ValueError("Expected counts must declare all four tasks with positive integer counts")
    if dict(task_counts) != expected_counts:
        failures.append(f"task counts {dict(task_counts)} != {expected_counts}")

    frame_count = 0
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)
    action_sum = np.zeros(7, dtype=np.float64)
    action_sq_sum = np.zeros(7, dtype=np.float64)
    transition_joint_errors: list[float] = []
    transition_gripper_errors: list[float] = []
    camera = {
        key: {
            "focus": [],
            "object_pixels": [],
            "first_object_pixels": [],
            "place_target_pixels": [],
            "first_place_target_pixels": [],
            "model_224_object_pixels": [],
            "first_model_224_object_pixels": [],
            "first_model_224_object_min_dimension": [],
            "model_224_place_target_pixels": [],
            "first_model_224_place_target_pixels": [],
            "first_object_contrast": [],
            "robot_fraction": [],
            "first_robot_fraction": [],
            "first_robot_border_margin": [],
        }
        for key in ("front", "wrist")
    }
    worst_focus = {
        key: {"score": float("inf"), "label": "", "image": None}
        for key in ("front", "wrist")
    }
    pose_ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
    place_pose_ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
    target_vocabulary: dict[str, Counter] = defaultdict(Counter)
    destination_vocabulary: dict[str, Counter] = defaultdict(Counter)
    target_destination_pairs: dict[str, Counter] = defaultdict(Counter)

    reset_samples = []
    for episode_number, item in enumerate(index, start=1):
        task = item["task"]
        name = task["name"]
        path = args.dataset / item["file"]
        grouped[name].append(path)
        if not item.get("success", False):
            failures.append(f"oracle failure: {path.name}")
        if scene_profile == "single_target":
            expected_prompt = TASKS[name].prompt
        else:
            template = INSTRUCTION_FORM[int(task["task_id"])]
            selected_place = task.get("config_selected_place")
            expected_prompt = (
                template.format(task["pick_name"], selected_place)
                if template.count("{}") == 2
                else template.format(task["pick_name"])
            )
        if task["prompt"] != expected_prompt:
            failures.append(f"prompt mismatch: {path.name}")
        if task.get("scene_profile", "single_target") != scene_profile:
            failures.append(f"{path.name}: task/manifest scene profile mismatch")
        if scene_profile == "rl_random_env":
            _audit_rl_scene(task, path.name, failures)
        pose_ranges[name].append(tuple(task["pick_position"][:2]))
        target_vocabulary[name][task["pick_name"]] += 1
        if task.get("config_selected_place") is not None:
            destination_vocabulary[name][task["config_selected_place"]] += 1
            if name in {"move", "pick_place"}:
                target_destination_pairs[name][
                    (task["pick_name"], task["config_selected_place"])
                ] += 1
        if task.get("place_position") is not None:
            place_pose_ranges[name].append(tuple(task["place_position"][:2]))
        with np.load(path) as archive:
            # NpzFile re-decompresses a member on every lookup.  Materialize
            # each array once so auditing every RGB/mask/state frame remains
            # practical for the balanced dataset.
            episode = {key: archive[key] for key in archive.files}
            reset_sample = {"file": path.name, "views": {}, "closed_empty_recovery": bool(
                item.get("recovery_start_actions_not_training_labels") and
                len(episode.get("gripper_closure", ())) and episode["gripper_closure"][0] >= .95 and
                len(episode.get("next_grasped_object", ())) and not episode["next_grasped_object"][0])}
            if joint_action_kind == "requested_joint_position_command":
                if "next_observed_joint_positions" not in episode:
                    failures.append(f"{path.name}: commanded action labels require separately recorded measured feedback")
                if item.get("native_joint_action_kind") != joint_action_kind:
                    failures.append(f"{path.name}: joint action kind differs from manifest")
            if policy_workspace_camera:
                overview_mask = episode.get("overview_reset_robot_mask")
                if overview_mask is None or overview_mask.shape != (int(manifest["image_size"]),)*2:
                    failures.append(f"{path.name}: missing/malformed video-only overview reset robot mask")
                else:
                    mask224 = np.asarray(Image.fromarray(overview_mask.astype(np.uint8)*255).resize((224, 224), Image.Resampling.BOX)) >= 128
                    overview_margins.append(border_margin(mask224))
                    if border_margin(mask224) < 3:
                        failures.append(f"{path.name}: full robot not inside video-only overview at reset with 3px margin")
            required = {
                "front_rgb",
                "wrist_rgb",
                "joint_positions",
                "gripper_closure",
                "canonical_actions",
                "next_joint_targets",
                "next_gripper_closure",
                "next_ee_object_distance",
                "next_object_goal_distance",
                "next_touching_object",
                "next_grasped_object",
                "next_raw_success",
                "next_success_streak",
                "prompt",
            }
            if scene_profile == "rl_random_env":
                required |= {
                    "front_pick_mask",
                    "wrist_pick_mask",
                    "front_place_mask",
                    "wrist_place_mask",
                    "front_robot_mask",
                    "wrist_robot_mask",
                }
            missing = required - set(episode)
            if missing:
                failures.append(f"{path.name}: missing {sorted(missing)}")
                continue
            if str(np.asarray(episode["prompt"]).item()) != task["prompt"]:
                failures.append(f"{path.name}: stored prompt differs from episode metadata")
            actions = episode["canonical_actions"].astype(np.float64)
            if body_v9:
                for field in ("next_requested_gripper_closure", "next_verified_grasp", "next_bilateral_pad_contact"):
                    if field not in episode:
                        failures.append(f"{path.name}: missing explicit V9 feedback/action/quality field {field}")
                if name != "touch" and ("next_verified_grasp" not in episode or not episode["next_verified_grasp"][-1]):
                    failures.append(f"{path.name}: terminal grasp is not physically verified")
            length = len(actions)
            if length != item["steps"]:
                failures.append(f"{path.name}: index steps {item['steps']} != arrays {length}")
            for key in required - {"prompt"}:
                if len(episode[key]) != length:
                    failures.append(f"{path.name}: {key} length mismatch")
            if actions.shape != (length, 7) or not np.isfinite(actions).all():
                failures.append(f"{path.name}: invalid actions {actions.shape}")
                continue
            if np.any(np.abs(actions[:, :3]) > 0.050001):
                failures.append(f"{path.name}: translation exceeds 5 cm contract")
            if np.any(np.abs(actions[:, 3:6]) > 0.250001):
                failures.append(f"{path.name}: rotation exceeds 0.25 rad contract")
            if np.any(np.abs(actions[:, 6]) > 1.000001):
                failures.append(f"{path.name}: gripper exceeds [-1,1]")

            reward_node = reward_root.find(f"r{task['task_id']}")
            if reward_node is None:
                failures.append(f"{path.name}: missing reward.xml r{task['task_id']}")
                continue
            live_thresholds = (float(reward_node.attrib["threshold0"]), float(reward_node.attrib["threshold1"]))
            from VLA.common.success_contract_snapshot import resolve_audit_thresholds
            try:
                dist_threshold, obj_threshold = resolve_audit_thresholds(
                    item, required_success_steps, live_thresholds, args.allow_stricter_frozen_reward_contract)
            except ValueError as exc:
                failures.append(f"{path.name}: {exc}")
                # Keep reporting independent image/feedback errors even when
                # the default live reward contract fails qualification.
                dist_threshold, obj_threshold = live_thresholds
            frozen_contracts[name].add((dist_threshold, obj_threshold))
            live_reward_contract_matches &= (dist_threshold, obj_threshold) == live_thresholds
            ee_distance = episode["next_ee_object_distance"].astype(np.float64)
            goal_distance = episode["next_object_goal_distance"].astype(np.float64)
            touching = episode["next_touching_object"].astype(bool)
            grasped = episode["next_grasped_object"].astype(bool)
            recorded_raw = episode["next_raw_success"].astype(bool)
            if name == "touch":
                expected_raw = (ee_distance < dist_threshold) | touching
            else:
                expected_raw = grasped & (goal_distance < obj_threshold)
            if not np.array_equal(recorded_raw, expected_raw):
                failures.append(f"{path.name}: per-frame success labels disagree with reward.xml")
            expected_streak = np.zeros(length, dtype=np.int64)
            running_streak = 0
            for step, raw_success in enumerate(expected_raw):
                running_streak = running_streak + 1 if raw_success else 0
                expected_streak[step] = running_streak
            if not np.array_equal(episode["next_success_streak"], expected_streak):
                failures.append(f"{path.name}: consecutive-success counter is misaligned")
            expected_episode_success = bool(
                length and expected_streak[-1] >= required_success_steps
            )
            if bool(item.get("success", False)) != expected_episode_success:
                failures.append(f"{path.name}: episode success label disagrees with final streak")
            action_min = np.minimum(action_min, actions.min(axis=0))
            action_max = np.maximum(action_max, actions.max(axis=0))
            action_sum += actions.sum(axis=0)
            action_sq_sum += (actions * actions).sum(axis=0)
            frame_count += length

            if length > 1:
                joint_error = np.max(
                    np.abs(
                        episode.get("next_observed_joint_positions", episode["next_joint_targets"])[:-1]
                        - episode["joint_positions"][1:]
                    ),
                    axis=1,
                )
                gripper_error = np.abs(
                    episode["next_gripper_closure"][:-1]
                    - episode["gripper_closure"][1:]
                )
                transition_joint_errors.extend(joint_error.tolist())
                transition_gripper_errors.extend(gripper_error.tolist())

            color = task["pick_name"].split()[0]
            for camera_name, array_key in (("front", "front_rgb"), ("wrist", "wrist_rgb")):
                images = episode[array_key]
                expected_image_shape = (int(manifest["image_size"]),) * 2 + (3,)
                if images.shape[1:] != expected_image_shape or images.dtype != np.uint8:
                    failures.append(f"{path.name}: invalid {array_key} {images.shape}/{images.dtype}")
                    continue
                pick_mask_key = f"{camera_name}_pick_mask"
                if pick_mask_key in episode:
                    masks = episode[pick_mask_key]
                    if masks.shape != images.shape[:3] or masks.dtype != np.bool_:
                        failures.append(
                            f"{path.name}: invalid {pick_mask_key} {masks.shape}/{masks.dtype}"
                        )
                    counts = masks.reshape(length, -1).sum(axis=1).astype(int).tolist()
                    model_metrics = [resized_mask_metrics(mask) for mask in masks]
                    camera[camera_name]["model_224_object_pixels"].extend(
                        metric[0] for metric in model_metrics
                    )
                    camera[camera_name]["first_model_224_object_pixels"].append(
                        model_metrics[0][0]
                    )
                    camera[camera_name]["first_model_224_object_min_dimension"].append(
                        min(model_metrics[0][1:])
                    )
                    camera[camera_name]["first_object_contrast"].append(
                        mask_contrast(images[0], masks[0])
                    )
                    reset_sample["views"][camera_name] = {
                        "raw_pixels": counts[0], "pixels_224": model_metrics[0][0],
                        "minimum_dimension_224": min(model_metrics[0][1:]),
                        "contrast": camera[camera_name]["first_object_contrast"][-1]}
                else:
                    counts = [int(color_mask(image, color).sum()) for image in images]
                camera[camera_name]["object_pixels"].extend(counts)
                camera[camera_name]["first_object_pixels"].append(counts[0])
                focus_scores = [focus_score(image) for image in images]
                camera[camera_name]["focus"].extend(focus_scores)
                worst_step = int(np.argmin(focus_scores))
                if focus_scores[worst_step] < worst_focus[camera_name]["score"]:
                    worst_focus[camera_name] = {
                        "score": focus_scores[worst_step],
                        "label": f"{path.name} step {worst_step}",
                        "image": images[worst_step].copy(),
                    }
                if (
                    task.get("place_name") is not None
                    and task["place_name"] not in FIXED_DESTINATION
                ):
                    place_mask_key = f"{camera_name}_place_mask"
                    if place_mask_key in episode:
                        place_masks = episode[place_mask_key]
                        if place_masks.shape != images.shape[:3] or place_masks.dtype != np.bool_:
                            failures.append(
                                f"{path.name}: invalid {place_mask_key} "
                                f"{place_masks.shape}/{place_masks.dtype}"
                            )
                        target_counts = (
                            place_masks.reshape(length, -1).sum(axis=1).astype(int).tolist()
                        )
                        place_model_counts = [
                            resized_mask_metrics(mask)[0] for mask in place_masks
                        ]
                        camera[camera_name]["model_224_place_target_pixels"].extend(
                            place_model_counts
                        )
                        camera[camera_name]["first_model_224_place_target_pixels"].append(
                            place_model_counts[0]
                        )
                    else:
                        target_counts = [int(color_mask(image, "blue").sum()) for image in images]
                    camera[camera_name]["place_target_pixels"].extend(target_counts)
                    camera[camera_name]["first_place_target_pixels"].append(target_counts[0])

                robot_mask_key = f"{camera_name}_robot_mask"
                if robot_mask_key in episode:
                    robot_masks = episode[robot_mask_key]
                    if robot_masks.shape != images.shape[:3] or robot_masks.dtype != np.bool_:
                        failures.append(
                            f"{path.name}: invalid {robot_mask_key} "
                            f"{robot_masks.shape}/{robot_masks.dtype}"
                        )
                    fractions = robot_masks.reshape(length, -1).mean(axis=1)
                    camera[camera_name]["robot_fraction"].extend(fractions.tolist())
                    camera[camera_name]["first_robot_fraction"].append(float(fractions[0]))
                    camera[camera_name]["first_robot_border_margin"].append(
                        border_margin(robot_masks[0])
                    )

        reset_samples.append(reset_sample)
        if episode_number % 100 == 0 or episode_number == len(index):
            print(f"audited {episode_number}/{len(index)} episodes; {frame_count} frames", flush=True)

    grouped = {name: sorted(paths) for name, paths in grouped.items()}
    save_contact_sheets(args.dataset, grouped, args.output)
    worst_sheet = Image.new("RGB", (512, 256), "black")
    for camera_index, camera_name in enumerate(("front", "wrist")):
        worst = worst_focus[camera_name]
        if worst["image"] is not None:
            panel = Image.fromarray(worst["image"]).resize((256, 256))
            add_label(panel, f"{camera_name} min focus {worst['score']:.2f} | {worst['label']}")
            worst_sheet.paste(panel, (camera_index * 256, 0))
    worst_sheet.save(args.output / "worst_focus_frames.png")
    transition_joint_max = max(transition_joint_errors, default=float("inf"))
    transition_gripper_max = max(transition_gripper_errors, default=float("inf"))
    if transition_joint_max > 1e-5:
        failures.append(f"observation/action joint transition max error {transition_joint_max}")
    if transition_gripper_max > 1e-6:
        failures.append(f"observation/action gripper transition max error {transition_gripper_max}")
    if scene_profile == "rl_random_env" and min(expected_counts.values()) >= 20:
        for task_name, points in pose_ranges.items():
            values = np.asarray(points)
            spans = np.ptp(values, axis=0)
            if np.any(spans < 0.40):
                failures.append(
                    f"{task_name}: commanded target XY span {spans.tolist()} does not cover workspace"
                )
    if manifest.get("stratified_target_vocabulary"):
        for task_name in TASKS:
            missing_pick = sorted(set(PICK_TARGETS) - set(target_vocabulary[task_name]))
            if missing_pick:
                failures.append(f"{task_name}: stratified pick vocabulary missing {missing_pick}")
            minimum_pick_repetitions = expected_counts[task_name] // len(PICK_TARGETS)
            if min(target_vocabulary[task_name].values(), default=0) < minimum_pick_repetitions:
                failures.append(
                    f"{task_name}: pick vocabulary is not balanced to at least "
                    f"{minimum_pick_repetitions} scenes per target"
                )
        for task_name in ("move", "pick_place"):
            missing_place = sorted(
                set(PLACE_TARGETS) - set(destination_vocabulary[task_name])
            )
            if missing_place:
                failures.append(
                    f"{task_name}: stratified destination vocabulary missing {missing_place}"
                )
            # Self-placement labels are skipped and advanced to the next
            # destination, so one count of deviation from the ideal quotient
            # is expected while still retaining near-uniform coverage.
            minimum_place_repetitions = max(
                expected_counts[task_name] // len(PLACE_TARGETS) - 1, 1
            )
            if (
                min(destination_vocabulary[task_name].values(), default=0)
                < minimum_place_repetitions
            ):
                failures.append(
                    f"{task_name}: destination vocabulary is not balanced to at least "
                    f"{minimum_place_repetitions} scenes per target"
                )
        valid_pairs = {
            (pick_name, place_name)
            for pick_name in PICK_TARGETS
            for place_name in PLACE_TARGETS
            if pick_name != place_name
        }
        # A complete compositional cycle needs 559 valid object/destination
        # pairs (self-placement is excluded by random_env).  Smaller pilot
        # datasets still report coverage without being rejected for not yet
        # containing a full cycle.
        for task_name in ("move", "pick_place"):
            if expected_counts[task_name] >= len(valid_pairs):
                missing_pairs = valid_pairs - set(target_destination_pairs[task_name])
                if missing_pairs:
                    failures.append(
                        f"{task_name}: stratified target/destination coverage missing "
                        f"{len(missing_pairs)} of {len(valid_pairs)} valid pairs"
                    )
    # TinyRenderer is deterministic and has no optical blur.  Global gradient
    # energy is still a useful overview-camera sanity check, but is invalid for
    # a wrist frame that correctly contains a large, textureless tabletop (the
    # exported worst-frame sheet makes such cases auditable).  Wrist quality is
    # instead gated on segmented target size/contrast and robot occupancy below.
    if min(camera["front"]["focus"], default=0.0) < 20.0:
        failures.append("front: focus score below 20 in at least one frame")

    # The overview camera must locate the commanded object at reset.  During
    # approach/grasp, either synchronized view may carry it; requiring the
    # wrist camera to see the whole table from the home pose would reward an
    # unrealistically wide and low-resolution hand-eye lens.
    front_object = np.asarray(camera["front"]["object_pixels"])
    wrist_object = np.asarray(camera["wrist"]["object_pixels"])
    front_object_first = np.asarray(camera["front"]["first_object_pixels"])
    wrist_object_first = np.asarray(camera["wrist"]["first_object_pixels"])
    front_object_reset_min = int(np.min(front_object_first))
    combined_object_reset_min = int(np.min(np.maximum(front_object_first, wrist_object_first)))
    combined_object_visibility = float(
        np.mean((front_object >= 25) | (wrist_object >= 25))
    )
    if front_object_reset_min < 25 and args.recovery_reset_observability == "front":
        failures.append("front: commanded object has <25 pixels in a reset frame")
    if combined_object_visibility < 0.95:
        failures.append("commanded-object visibility across both cameras is below 95%")

    model_input_camera: dict[str, object] = {"resolution": [224, 224]}
    if scene_profile == "rl_random_env":
        front_model_object = np.asarray(camera["front"]["model_224_object_pixels"])
        wrist_model_object = np.asarray(camera["wrist"]["model_224_object_pixels"])
        front_model_first = np.asarray(camera["front"]["first_model_224_object_pixels"])
        wrist_model_first = np.asarray(camera["wrist"]["first_model_224_object_pixels"])
        if int(front_model_first.min()) < MIN_FRONT_RESET_TARGET_PIXELS_224 and args.recovery_reset_observability == "front":
            failures.append(
                "224px front input: target has fewer than "
                f"{MIN_FRONT_RESET_TARGET_PIXELS_224} pixels in a reset frame"
            )
        if min(camera["front"]["first_model_224_object_min_dimension"]) < 7 and args.recovery_reset_observability == "front":
            failures.append("224px front input: target is thinner than 7 pixels at reset")
        combined_model_visibility = float(
            np.mean((front_model_object >= 49) | (wrist_model_object >= 49))
        )
        if combined_model_visibility < 0.95:
            failures.append("224px input: target visibility across both cameras is below 95%")
        if min(camera["front"]["first_object_contrast"]) < 25.0 and args.recovery_reset_observability == "front":
            failures.append("front input: target/local-background RGB contrast is below 25")
        # A white object can have near-zero wrist/background contrast when the
        # hand camera looks beyond the dark tabletop into the white renderer
        # background.  The synchronized front view is therefore the required
        # color-contrast source; wrist contrast remains reported as a warning
        # metric while its segmented size/visibility is still fail-closed.

        front_robot_reset = np.asarray(camera["front"]["first_robot_fraction"])
        wrist_robot_reset = np.asarray(camera["wrist"]["first_robot_fraction"])
        wrist_robot_all = np.asarray(camera["wrist"]["robot_fraction"])
        front_robot_margin = np.asarray(camera["front"]["first_robot_border_margin"])
        if not policy_workspace_camera and int(front_robot_margin.min()) < 3:
            failures.append("front input: complete robot is not inside the reset frame with 3px margin")
        if float(front_robot_reset.max()) > 0.35:
            failures.append("front input: robot occupies >35% of a reset frame")
        if float(wrist_robot_reset.max()) > 0.40:
            failures.append("wrist input: robot/gripper occupies >40% of a reset frame")
        if float(np.quantile(wrist_robot_all, 0.99)) > 0.65:
            failures.append("wrist input: robot/gripper occupies >65% at trajectory p99")

        front_model_place = np.asarray(camera["front"]["model_224_place_target_pixels"])
        wrist_model_place = np.asarray(camera["wrist"]["model_224_place_target_pixels"])
        model_place_visibility = (
            float(np.mean((front_model_place >= 49) | (wrist_model_place >= 49)))
            if front_model_place.size
            else 1.0
        )
        if model_place_visibility < 0.95:
            failures.append("224px input: physical place-target visibility is below 95%")
        model_input_camera.update(
            {
                "front_reset_target_pixels_min": int(front_model_first.min()),
                "front_reset_target_pixel_gate": MIN_FRONT_RESET_TARGET_PIXELS_224,
                "front_reset_target_min_dimension_px": int(
                    min(camera["front"]["first_model_224_object_min_dimension"])
                ),
                "combined_target_visible_fraction_at_49px": combined_model_visibility,
                "combined_place_visible_fraction_at_49px": model_place_visibility,
                "front_reset_target_contrast_min": float(
                    min(camera["front"]["first_object_contrast"])
                ),
                "wrist_reset_target_contrast_min": float(
                    min(camera["wrist"]["first_object_contrast"])
                ),
                "front_reset_robot_fraction_max": float(front_robot_reset.max()),
                "front_reset_robot_border_margin_min_px": int(front_robot_margin.min()),
                "wrist_reset_robot_fraction_max": float(wrist_robot_reset.max()),
                "wrist_trajectory_robot_fraction_p99": float(
                    np.quantile(wrist_robot_all, 0.99)
                ),
            }
        )
        if args.recovery_reset_observability == "dual_view":
            if len(reset_samples) != len(index):
                failures.append("Incomplete per-episode recovery reset camera evidence")
            failures.extend(reset_observability_failures(reset_samples, dual_recovery=True))

    # The front camera provides global goal context while the wrist camera is
    # intentionally close-range.  Require every physical destination to be
    # visible in a synchronized view.  Named workspace regions have no body,
    # matching the original RL environment, and are therefore excluded.
    front_target = np.asarray(camera["front"]["place_target_pixels"])
    wrist_target = np.asarray(camera["wrist"]["place_target_pixels"])
    front_target_first = np.asarray(camera["front"]["first_place_target_pixels"])
    wrist_target_first = np.asarray(camera["wrist"]["first_place_target_pixels"])
    if front_target.size:
        combined_target_visibility = float(
            np.mean((front_target >= 25) | (wrist_target >= 25))
        )
        combined_target_reset_min = int(
            np.min(np.maximum(front_target_first, wrist_target_first))
        )
        if combined_target_reset_min < 25:
            failures.append("physical place target has <25 pixels across both cameras at reset")
        if combined_target_visibility < 0.95:
            failures.append("physical place-target visibility across both cameras is below 95%")
    else:
        combined_target_visibility = 1.0
        combined_target_reset_min = 0

    action_mean = action_sum / max(frame_count, 1)
    action_std = np.sqrt(np.maximum(action_sq_sum / max(frame_count, 1) - action_mean**2, 0))
    verify_source_payload_fingerprint(payload_fingerprint, args.dataset, index)
    for name, contracts in frozen_contracts.items():
        if len(contracts) != 1:
            failures.append(f"{name}: mixed frozen distance contracts in one source")
    if sha256(reward_path) != reward_sha256:
        failures.append("Live reward.xml changed during source audit")
    report = {
        "dataset": str(args.dataset.resolve()),
        "index_sha256": sha256(args.dataset / "index.json"),
        "manifest_sha256": sha256(args.dataset / "dataset_manifest.json"),
        "source_payload_fingerprint": payload_fingerprint,
        "passed": not failures,
        "failures": failures,
        "episodes": len(index),
        "frames": frame_count,
        "required_success_steps": required_success_steps,
        "success_contract_source": ("recorded_no_looser_than_live_snapshot" if args.allow_stricter_frozen_reward_contract
                                    else "live_reward_xml"),
        "recovery_reset_observability": args.recovery_reset_observability,
        "reset_camera_samples": reset_samples,
        "live_reward_xml_sha256": reward_sha256,
        "live_reward_contract_matches": live_reward_contract_matches,
        "audited_distance_thresholds_by_task": {name: [list(pair) for pair in sorted(contracts)]
                                                for name, contracts in frozen_contracts.items()},
        "live_distance_thresholds_by_task": {
            name: [float(reward_root.find(f"r{spec.task_id}").attrib[key]) for key in ("threshold0", "threshold1")]
            for name, spec in TASKS.items()},
        "task_counts": dict(task_counts),
        "native_joint_action_kind": joint_action_kind,
        "front_camera_role": front_role,
        "full_robot_overview_gate": "independent audit-only overview masks at 224px" if policy_workspace_camera else "legacy front camera",
        "video_only_overview": {"reset_masks_checked": len(overview_margins),
                                "min_robot_border_margin_224": min(overview_margins) if overview_margins else None,
                                "supplied_to_policy": False},
        "expected_task_counts": expected_counts,
        "target_vocabulary_counts": {
            name: dict(counts) for name, counts in target_vocabulary.items()
        },
        "destination_vocabulary_counts": {
            name: dict(counts) for name, counts in destination_vocabulary.items()
        },
        "target_destination_pair_coverage": {
            name: {
                "unique_pairs": len(target_destination_pairs[name]),
                "valid_pairs": len(PICK_TARGETS) * len(PLACE_TARGETS)
                - len(set(PICK_TARGETS) & set(PLACE_TARGETS)),
                "minimum_repetitions": min(
                    target_destination_pairs[name].values(), default=0
                ),
                "maximum_repetitions": max(
                    target_destination_pairs[name].values(), default=0
                ),
            }
            for name in ("move", "pick_place")
        },
        "all_oracle_success": all(item.get("success", False) for item in index),
        "temporal_alignment": {
            "next_joint_to_next_observation_max_abs_error": transition_joint_max,
            "next_gripper_to_next_observation_max_abs_error": transition_gripper_max,
        },
        "action": {
            "min": action_min.tolist(),
            "max": action_max.tolist(),
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
        },
        "camera": {
            name: {
                "reset_object_pixels_min": min(values["first_object_pixels"]),
                "reset_object_pixels_median": float(np.median(values["first_object_pixels"])),
                "trajectory_object_visible_fraction_at_25px": float(
                    np.mean(np.asarray(values["object_pixels"]) >= 25)
                ),
                "focus_score_min": float(min(values["focus"])),
                "focus_score_median": float(np.median(values["focus"])),
                "reset_place_target_pixels_min": min(
                    values["first_place_target_pixels"], default=0
                ),
                "trajectory_place_target_visible_fraction_at_25px": float(
                    np.mean(np.asarray(values["place_target_pixels"]) >= 25)
                ) if values["place_target_pixels"] else 1.0,
            }
            for name, values in camera.items()
        },
        "worst_focus_frames": {
            name: {"score": values["score"], "source": values["label"]}
            for name, values in worst_focus.items()
        },
        "combined_camera": {
            "front_reset_object_pixels_min": front_object_reset_min,
            "reset_object_pixels_min": combined_object_reset_min,
            "trajectory_object_visible_fraction_at_25px": combined_object_visibility,
            "reset_place_target_pixels_min": combined_target_reset_min,
            "trajectory_place_target_visible_fraction_at_25px": combined_target_visibility,
        },
        "model_input_camera": model_input_camera,
        "pick_xy_range_m": {
            name: {
                "min": np.min(np.asarray(points), axis=0).tolist(),
                "max": np.max(np.asarray(points), axis=0).tolist(),
            }
            for name, points in pose_ranges.items()
        },
        "place_xy_range_m": {
            name: {
                "min": np.min(np.asarray(points), axis=0).tolist(),
                "max": np.max(np.asarray(points), axis=0).tolist(),
            }
            for name, points in place_pose_ranges.items()
        },
        "artifacts": [
            "camera_contact_sheet.png",
            "model_input_224_contact_sheet.png",
            "worst_focus_frames.png",
        ],
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def _audit_rl_scene(task: dict, filename: str, failures: list[str]) -> None:
    """Validate the exact structural invariants inherited from random_env.py."""

    picks = list(task.get("config_pick_items", ()))
    places = list(task.get("config_place_items", ()))
    if len(picks) not in (3, 4):
        failures.append(f"{filename}: expected 3-4 configured pick items, got {len(picks)}")
    if len(places) not in (2, 3):
        failures.append(f"{filename}: expected 2-3 configured place items, got {len(places)}")
    if len(set(picks)) != len(picks) or len(set(places)) != len(places):
        failures.append(f"{filename}: duplicate configured scene item")
    if set(picks) & set(places):
        failures.append(f"{filename}: pick/place pools overlap")
    if task["pick_name"] not in picks:
        failures.append(f"{filename}: commanded pick target absent from pick config")
    if task.get("place_name") is not None and task["place_name"] not in places:
        failures.append(f"{filename}: commanded destination absent from place config")
    if task.get("config_selected_place") not in places:
        failures.append(f"{filename}: RobotEnv-selected place item absent from place config")
    if any(name not in PICK_TARGETS for name in picks):
        failures.append(f"{filename}: pick item outside config.PICK_TARGETS")
    if any(name not in PLACE_TARGETS for name in places):
        failures.append(f"{filename}: place item outside config.PLACE_TARGETS")

    expected_physical = {
        name for name in (*picks, *places) if name not in FIXED_DESTINATION
    }
    represented = {task["pick_name"]}
    if task.get("place_name") not in (None, *FIXED_DESTINATION):
        represented.add(task["place_name"])
    represented.update(item["name"] for item in task.get("scene_objects", ()))
    if represented != expected_physical:
        failures.append(
            f"{filename}: physical scene objects {sorted(represented)} != config {sorted(expected_physical)}"
        )

    positions: list[np.ndarray] = [np.asarray(task["pick_position"], dtype=np.float64)]
    if task.get("place_name") not in (None, *FIXED_DESTINATION):
        positions.append(np.asarray(task["place_position"], dtype=np.float64))
    positions.extend(
        np.asarray(item["position"], dtype=np.float64)
        for item in task.get("scene_objects", ())
    )
    for position in positions:
        if not (
            BOUNDS[0, 0] <= position[0] <= BOUNDS[0, 1]
            and BOUNDS[1, 0] <= position[1] <= BOUNDS[1, 1]
        ):
            failures.append(f"{filename}: physical object outside config.BOUNDS")
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            if np.linalg.norm(first[:2] - second[:2]) <= 0.13:
                failures.append(f"{filename}: physical objects violate 13 cm spacing")
                return
    if task.get("place_name") in FIXED_DESTINATION:
        destination = np.asarray(FIXED_DESTINATION[task["place_name"]], dtype=np.float64)
        if any(np.linalg.norm(position[:2] - destination[:2]) <= 0.13 for position in positions):
            failures.append(f"{filename}: physical object obstructs commanded fixed destination")


if __name__ == "__main__":
    raise SystemExit(main())
