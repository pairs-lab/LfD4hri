#!/usr/bin/env python3
"""Export exact pi0, SmolVLA and RDT image transforms before fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TASK_NAMES = ("touch", "pick", "move", "pick_place")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "smolvla", "rdt"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-prefix", default=None,
                        help="Optional explicit source subset for an additional visual review (not a full-dataset gate)")
    parser.add_argument(
        "--rdt-processor",
        default="VLA/checkpoints/siglip-so400m-patch14-384",
    )
    return parser.parse_args()


def select_frames(dataset: Path, file_prefix=None) -> list[tuple[str, str, str, str, np.ndarray, np.ndarray]]:
    """Select the smallest-reset-target episode per task, then its trajectory phases.

    A first-episode contact sheet systematically misses the difficult small
    fruit/corner combinations.  Use the simulator-only mask solely to choose
    audit examples; it is never part of a policy input or converted dataset.
    """
    index = json.loads((dataset / "index.json").read_text())
    if file_prefix is not None:
        index = [item for item in index if item["file"].startswith(file_prefix)]
    selected: list[tuple[str, str, str, str, np.ndarray, np.ndarray]] = []
    for task_name in TASK_NAMES:
        candidates = [entry for entry in index if entry["task"]["name"] == task_name]
        scored: list[tuple[int, dict]] = []
        for item in candidates:
            with np.load(dataset / item["file"]) as episode:
                scored.append((int(episode["front_pick_mask"][0].sum()), item))
        _, item = min(scored, key=lambda pair: pair[0])
        with np.load(dataset / item["file"]) as episode:
            length = len(episode["canonical_actions"])
            for phase, step in (("start", 0), ("middle", length // 2), ("success", length - 1)):
                selected.append(
                    (
                        task_name,
                        phase,
                        item["file"],
                        item["task"]["pick_name"],
                        episode["front_rgb"][step].copy(),
                        episode["wrist_rgb"][step].copy(),
                    )
                )
    return selected


def pi0_transform(image: np.ndarray) -> tuple[np.ndarray, list[int]]:
    from openpi_client import image_tools

    result = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    return np.asarray(result), list(result.shape)


def smolvla_transform(image: np.ndarray) -> tuple[np.ndarray, list[int]]:
    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    result = resize_with_pad(tensor, 512, 512, pad_value=0)[0]
    rgb = np.rint(result.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return rgb, list(result.shape)


def make_rdt_transform(processor_path: str):
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(processor_path)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)

    def transform(image: np.ndarray) -> tuple[np.ndarray, list[int]]:
        tensor = processor.preprocess(Image.fromarray(image), return_tensors="np")["pixel_values"][0]
        rgb = np.clip(tensor.transpose(1, 2, 0) * std + mean, 0.0, 1.0)
        return np.rint(rgb * 255.0).astype(np.uint8), list(tensor.shape)

    def json_size(value):
        if value is None:
            return None
        if hasattr(value, "items"):
            return {str(key): int(item) for key, item in value.items()}
        if hasattr(value, "__dict__"):
            return {
                str(key): int(item) if isinstance(item, (int, np.integer)) else item
                for key, item in vars(value).items()
                if item is not None
            }
        if isinstance(value, (tuple, list)):
            return [int(item) for item in value]
        return value

    return transform, {
        "processor": processor_path,
        "size": json_size(processor.size),
        "crop_size": json_size(getattr(processor, "crop_size", None)),
        "do_center_crop": getattr(processor, "do_center_crop", None),
    }


def main() -> int:
    args = parse_args()
    frames = select_frames(args.dataset, args.file_prefix)
    details: dict = {}
    if args.model == "pi0":
        transform = pi0_transform
        details = {"method": "openpi_client.image_tools.resize_with_pad", "size": [224, 224]}
    elif args.model == "smolvla":
        transform = smolvla_transform
        details = {"method": "lerobot resize_with_pad", "size": [512, 512]}
    else:
        transform, details = make_rdt_transform(args.rdt_processor)

    cells: list[Image.Image] = []
    samples: list[dict] = []
    for task_name, phase, filename, pick_name, front, wrist in frames:
        front_out, front_shape = transform(front)
        wrist_out, wrist_shape = transform(wrist)
        finite = bool(np.isfinite(front_out).all() and np.isfinite(wrist_out).all())
        pair = Image.new("RGB", (448, 248), "black")
        pair.paste(Image.fromarray(front_out).resize((224, 224)), (0, 24))
        pair.paste(Image.fromarray(wrist_out).resize((224, 224)), (224, 24))
        ImageDraw.Draw(pair).text(
            (4, 5), f"{task_name} | {phase} | {pick_name} | front / wrist", fill="white"
        )
        cells.append(pair)
        samples.append(
            {
                "task": task_name,
                "phase": phase,
                "file": filename,
                "pick_name": pick_name,
                "front_tensor_shape": front_shape,
                "wrist_tensor_shape": wrist_shape,
                "finite": finite,
            }
        )

    sheet = Image.new("RGB", (448 * 3, 248 * 4), "black")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 3) * 448, (index // 3) * 248))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    report = {
        "passed": all(sample["finite"] for sample in samples),
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "source_file_prefix": args.file_prefix,
        "scope": "Finite image-transform export; full visibility/label qualification requires the separate source audit",
        "transform": details,
        "samples": samples,
        "contact_sheet": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
