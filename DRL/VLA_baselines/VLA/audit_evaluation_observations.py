#!/usr/bin/env python3
"""Fail-closed camera and language audit for every locked evaluation scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.spatial import FIXED_DESTINATION
from VLA.audit_finetune_data import (
    MIN_FRONT_RESET_TARGET_PIXELS_224,
    border_margin,
    focus_score,
    mask_contrast,
    resized_mask_metrics,
)
from VLA.common.environment import CAMERA_CONFIG, ManipulationEnv
from VLA.common.tasks import randomize_rl_scene, select_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "VLA/evaluation_protocol.json"
DEFAULT_TRAIN = (
    PROJECT_ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "VLA/audits/evaluation_observations_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-exact-prompts",
        action="store_true",
        help="Fail unless every locked evaluation instruction occurs in training",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_pair(front: np.ndarray, wrist: np.ndarray, label: str) -> Image.Image:
    panel = Image.new("RGB", (448, 224), "black")
    panel.paste(Image.fromarray(front).resize((224, 224)), (0, 0))
    panel.paste(Image.fromarray(wrist).resize((224, 224)), (224, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 448, 24), fill="black")
    draw.text((5, 6), label, fill="white")
    return panel


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text())
    train_index = json.loads((args.train_dataset / "index.json").read_text())
    train_picks = {
        task_name: {
            row["task"]["pick_name"]
            for row in train_index
            if row["task"]["name"] == task_name
        }
        for task_name in ("touch", "pick", "move", "pick_place")
    }
    train_places = {
        task_name: {
            row["task"]["place_name"]
            for row in train_index
            if row["task"]["name"] == task_name
            and row["task"].get("place_name") is not None
        }
        for task_name in ("move", "pick_place")
    }
    train_prompts = {row["task"]["prompt"] for row in train_index}

    rows: list[dict] = []
    panels: list[tuple[int, Image.Image]] = []
    seeds = [
        *(('development', int(seed)) for seed in protocol["development_seeds"]),
        *(("final_test", int(seed)) for seed in protocol["final_test_seeds"]),
    ]
    with ManipulationEnv(
        robot=protocol["robot"],
        image_size=384,
        control_hz=int(protocol["control_hz"]),
        seed=0,
        required_success_steps=1,
        use_semantic_success=True,
    ) as env:
        for split, seed in seeds:
            rng = np.random.default_rng(seed)
            for base_task in select_tasks("all"):
                task = randomize_rl_scene(base_task, rng)
                observation = env.reset(task)
                front_pick = env.target_mask("front", "pick")
                wrist_pick = env.target_mask("wrist", "pick")
                front_pick_224 = resized_mask_metrics(front_pick)
                wrist_pick_224 = resized_mask_metrics(wrist_pick)
                front_robot = env.robot_mask("front")
                wrist_robot = env.robot_mask("wrist")
                physical_place = (
                    task.place_name is not None
                    and task.place_name not in FIXED_DESTINATION
                )
                front_place_224 = (
                    resized_mask_metrics(env.target_mask("front", "place"))
                    if physical_place
                    else (0, 0, 0)
                )
                wrist_place_224 = (
                    resized_mask_metrics(env.target_mask("wrist", "place"))
                    if physical_place
                    else (0, 0, 0)
                )
                row = {
                    "split": split,
                    "seed": seed,
                    "task": task.name,
                    "prompt": task.prompt,
                    "pick_name": task.pick_name,
                    "place_name": task.place_name,
                    "pick_label_seen_for_task_in_train": task.pick_name in train_picks[task.name],
                    "place_label_seen_for_task_in_train": (
                        True
                        if task.name not in train_places
                        else task.place_name in train_places[task.name]
                    ),
                    "exact_prompt_seen_in_train": task.prompt in train_prompts,
                    "front_pick_pixels_224": front_pick_224[0],
                    "front_pick_width_224": front_pick_224[1],
                    "front_pick_height_224": front_pick_224[2],
                    "wrist_pick_pixels_224": wrist_pick_224[0],
                    "front_pick_contrast": mask_contrast(observation.front_rgb, front_pick),
                    "front_focus": focus_score(observation.front_rgb),
                    "wrist_focus": focus_score(observation.wrist_rgb),
                    "front_robot_fraction": float(front_robot.mean()),
                    "wrist_robot_fraction": float(wrist_robot.mean()),
                    "front_robot_border_margin_px": border_margin(front_robot),
                    "physical_place_target": physical_place,
                    "front_place_pixels_224": front_place_224[0],
                    "wrist_place_pixels_224": wrist_place_224[0],
                }
                rows.append(row)
                # Rank small/low-contrast targets first for a compact visual audit.
                rank = int(
                    min(front_pick_224[0], 1000)
                    + 2 * min(row["front_pick_contrast"], 255.0)
                )
                panels.append(
                    (
                        rank,
                        labelled_pair(
                            observation.front_rgb,
                            observation.wrist_rgb,
                            f"{split} {seed} {task.name}: {task.pick_name} | "
                            f"front/wrist={front_pick_224[0]}/{wrist_pick_224[0]} px",
                        ),
                    )
                )

    failures: list[str] = []
    for row in rows:
        label = f"{row['split']}:{row['seed']}:{row['task']}"
        if not row["pick_label_seen_for_task_in_train"]:
            failures.append(f"{label}: unseen task-conditioned pick label")
        if not row["place_label_seen_for_task_in_train"]:
            failures.append(f"{label}: unseen task-conditioned place label")
        if args.require_exact_prompts and not row["exact_prompt_seen_in_train"]:
            failures.append(f"{label}: exact evaluation instruction absent from training")
        if row["front_pick_pixels_224"] < MIN_FRONT_RESET_TARGET_PIXELS_224:
            failures.append(
                f"{label}: front target smaller than "
                f"{MIN_FRONT_RESET_TARGET_PIXELS_224} pixels at 224px"
            )
        if min(row["front_pick_width_224"], row["front_pick_height_224"]) < 7:
            failures.append(f"{label}: front target thinner than 7 pixels at 224px")
        if row["front_pick_contrast"] < 25.0:
            failures.append(f"{label}: front target contrast below 25")
        if row["front_focus"] < 20.0:
            failures.append(f"{label}: front focus score below 20")
        if row["front_robot_fraction"] > 0.35:
            failures.append(f"{label}: robot occupies over 35% of front frame")
        if row["wrist_robot_fraction"] > 0.40:
            failures.append(f"{label}: robot occupies over 40% of wrist frame")
        if row["front_robot_border_margin_px"] < 3:
            failures.append(f"{label}: complete robot lacks 3px front-frame margin")
        if row["physical_place_target"] and max(
            row["front_place_pixels_224"], row["wrist_place_pixels_224"]
        ) < 49:
            failures.append(f"{label}: physical destination smaller than 49 pixels")

    args.output.mkdir(parents=True, exist_ok=True)
    selected_panels = [panel for _rank, panel in sorted(panels, key=lambda item: item[0])[:16]]
    sheet = Image.new("RGB", (448 * 2, 224 * 8), "black")
    for index, panel in enumerate(selected_panels):
        sheet.paste(panel, ((index % 2) * 448, (index // 2) * 224))
    sheet_path = args.output / "worst_locked_scenes_224.png"
    sheet.save(sheet_path)

    physical_rows = [row for row in rows if row["physical_place_target"]]
    report = {
        "passed": not failures,
        "failures": failures,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "training_dataset": str(args.train_dataset.resolve()),
        "training_index_sha256": sha256(args.train_dataset / "index.json"),
        "environment_sha256": sha256(PROJECT_ROOT / "VLA/common/environment.py"),
        "camera": CAMERA_CONFIG,
        "trials": len(rows),
        "criteria": {
            "require_exact_prompts": args.require_exact_prompts,
            "front_pick_pixels_224_min": MIN_FRONT_RESET_TARGET_PIXELS_224,
            "front_pick_min_dimension_224_min": 7,
            "front_pick_contrast_min": 25.0,
            "front_focus_min": 20.0,
            "front_robot_fraction_max": 0.35,
            "wrist_robot_fraction_max": 0.40,
            "front_robot_border_margin_px_min": 3,
            "physical_place_pixels_224_combined_min": 49,
        },
        "summary": {
            "front_pick_pixels_224_min": min(row["front_pick_pixels_224"] for row in rows),
            "front_pick_min_dimension_224_min": min(
                min(row["front_pick_width_224"], row["front_pick_height_224"])
                for row in rows
            ),
            "front_pick_contrast_min": min(row["front_pick_contrast"] for row in rows),
            "front_focus_min": min(row["front_focus"] for row in rows),
            "wrist_focus_min_report_only": min(row["wrist_focus"] for row in rows),
            "front_robot_fraction_max": max(row["front_robot_fraction"] for row in rows),
            "wrist_robot_fraction_max": max(row["wrist_robot_fraction"] for row in rows),
            "front_robot_border_margin_px_min": min(
                row["front_robot_border_margin_px"] for row in rows
            ),
            "physical_place_pixels_224_combined_min": (
                min(
                    max(row["front_place_pixels_224"], row["wrist_place_pixels_224"])
                    for row in physical_rows
                )
                if physical_rows
                else None
            ),
            "pick_label_coverage": sum(
                int(row["pick_label_seen_for_task_in_train"]) for row in rows
            ) / len(rows),
            "place_label_coverage": sum(
                int(row["place_label_seen_for_task_in_train"]) for row in rows
            ) / len(rows),
            "exact_prompt_seen_fraction": sum(
                int(row["exact_prompt_seen_in_train"]) for row in rows
            ) / len(rows),
        },
        "contact_sheet": str(sheet_path.resolve()),
        "rows": rows,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if failures:
        raise RuntimeError("Locked evaluation observation audit failed")


if __name__ == "__main__":
    main()
