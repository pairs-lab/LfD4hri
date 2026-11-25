#!/usr/bin/env python3
"""Stress-test VLA camera geometry at the full random_env workspace boundary."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from VLA.audit_finetune_data import border_margin, mask_contrast, resized_mask_metrics
from VLA.common.environment import CAMERA_CONFIG, ManipulationEnv
from config import PICK_TARGETS
from VLA.common.tasks import TASKS


POSITIONS = {
    # Stay 1 mm inside config.BOUNDS to stress the actual sampling support
    # without relying on floating-point equality at the exact edge.
    "top_left": (-0.249, -0.251),
    "top_center": (0.00, -0.251),
    "top_right": (0.249, -0.251),
    "middle_left": (-0.249, -0.50),
    "middle": (0.00, -0.50),
    "middle_right": (0.249, -0.50),
    "bottom_left": (-0.249, -0.749),
    "bottom_center": (0.00, -0.749),
    "bottom_right": (0.249, -0.749),
}


def triple(value: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return values  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eye", type=triple, required=True)
    parser.add_argument("--target", type=triple, required=True)
    parser.add_argument("--fov", type=float, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--objects",
        default="all",
        help="'all' for PICK_TARGETS or a comma-separated subset for camera search",
    )
    return parser.parse_args()


def label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill="black")
    draw.text((5, 5), text, fill="white")
    return image


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    CAMERA_CONFIG["front"]["eye_world"] = args.eye
    CAMERA_CONFIG["front"]["target_world"] = args.target
    CAMERA_CONFIG["front"]["fov_degrees"] = args.fov
    object_names = (
        list(PICK_TARGETS)
        if args.objects == "all"
        else [name.strip() for name in args.objects.split(",") if name.strip()]
    )
    unknown = [name for name in object_names if name not in PICK_TARGETS]
    if unknown:
        raise ValueError(f"Unknown pick object(s): {unknown}")

    rows: list[dict] = []
    worst_by_position: dict[str, tuple[int, Image.Image]] = {}
    with ManipulationEnv(image_size=args.image_size, seed=0) as env:
        for position_name, xy in POSITIONS.items():
            for pick_name in object_names:
                z = 0.03 if pick_name.endswith((" block", " egg")) else 0.0
                position = (*xy, z)
                task = replace(
                    TASKS["touch"],
                    pick_name=pick_name,
                    pick_position=position,
                )
                observation = env.reset(task)
                front_target = env.target_mask("front", "pick")
                wrist_target = env.target_mask("wrist", "pick")
                front_robot = env.robot_mask("front")
                wrist_robot = env.robot_mask("wrist")
                front_224 = resized_mask_metrics(front_target)
                wrist_224 = resized_mask_metrics(wrist_target)
                rows.append(
                    {
                        "position": position_name,
                        "pick_name": pick_name,
                        "xyz": list(position),
                        "front_target_pixels_224": front_224[0],
                        "front_target_bounds_224": list(front_224[1:]),
                        "wrist_target_pixels_224": wrist_224[0],
                        "wrist_target_bounds_224": list(wrist_224[1:]),
                        "front_target_contrast": mask_contrast(observation.front_rgb, front_target),
                        "wrist_target_contrast": mask_contrast(observation.wrist_rgb, wrist_target),
                        "front_target_border_margin_px": border_margin(front_target),
                        "wrist_target_border_margin_px": border_margin(wrist_target),
                        "front_robot_fraction": float(front_robot.mean()),
                        "wrist_robot_fraction": float(wrist_robot.mean()),
                        "front_robot_border_margin_px": border_margin(front_robot),
                    }
                )
                if (
                    position_name not in worst_by_position
                    or front_224[0] < worst_by_position[position_name][0]
                ):
                    front = Image.fromarray(observation.front_rgb).resize((224, 224))
                    wrist = Image.fromarray(observation.wrist_rgb).resize((224, 224))
                    panel = Image.new("RGB", (448, 224), "black")
                    panel.paste(front, (0, 0))
                    panel.paste(wrist, (224, 0))
                    worst_by_position[position_name] = (
                        front_224[0],
                        label(
                            panel,
                            f"{position_name} | {pick_name} | front / wrist",
                        ),
                    )

    sheet = Image.new("RGB", (448 * 3, 224 * 3), "black")
    for index, position_name in enumerate(POSITIONS):
        panel = worst_by_position[position_name][1]
        sheet.paste(panel, ((index % 3) * 448, (index // 3) * 224))
    sheet_path = args.output / "workspace_grid_224.png"
    sheet.save(sheet_path)

    minimum_front_pixels = min(row["front_target_pixels_224"] for row in rows)
    minimum_front_dimension = min(min(row["front_target_bounds_224"]) for row in rows)
    minimum_combined_pixels = min(
        max(row["front_target_pixels_224"], row["wrist_target_pixels_224"])
        for row in rows
    )
    minimum_contrast = min(row["front_target_contrast"] for row in rows)
    minimum_target_margin = min(row["front_target_border_margin_px"] for row in rows)
    worst_target = min(rows, key=lambda row: row["front_target_pixels_224"])
    report = {
        "passed": (
            minimum_front_pixels >= 49
            and minimum_front_dimension >= 7
            and minimum_combined_pixels >= 49
            and minimum_contrast >= 25.0
            and minimum_target_margin > 0
        ),
        "camera": {"eye_world": args.eye, "target_world": args.target, "fov_degrees": args.fov},
        "criteria": {
            "front_target_pixels_224_min": 49,
            "front_target_min_dimension_224_min": 7,
            "combined_target_pixels_224_min": 49,
            "front_target_contrast_min": 25.0,
            "front_target_border_margin_px_min": 1,
        },
        "summary": {
            "front_target_pixels_224_min": minimum_front_pixels,
            "front_target_min_dimension_224_min": minimum_front_dimension,
            "combined_target_pixels_224_min": minimum_combined_pixels,
            "front_target_contrast_min": minimum_contrast,
            "front_target_border_margin_px_min": minimum_target_margin,
            "front_robot_border_margin_px_min": min(
                row["front_robot_border_margin_px"] for row in rows
            ),
            "worst_target": worst_target,
        },
        "positions": rows,
        "contact_sheet": str(sheet_path.resolve()),
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "camera", "criteria", "summary")}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
