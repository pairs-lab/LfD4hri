#!/usr/bin/env python3
"""Export the exact two-backbone OpenVLA image tensors for visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="Original audited NPZ dataset; selects the smallest target episode per task",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openvla/openvla-7b")
    return parser.parse_args()


def unnormalize(tensor, mean, std) -> Image.Image:
    array = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    array = np.clip(array * np.asarray(std) + np.asarray(mean), 0.0, 1.0)
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def main() -> None:
    from transformers import AutoProcessor

    args = parse_args()
    samples = json.loads((args.dataset / "samples.json").read_text())
    by_episode: dict[int, list[dict]] = {}
    for sample in samples:
        by_episode.setdefault(sample["episode_index"], []).append(sample)
    selected = []
    source_labels: dict[int, dict] = {}
    if args.source is not None:
        source_index = json.loads((args.source / "index.json").read_text())
        selected_episode_indices = []
        for task_name in ("touch", "pick", "move", "pick_place"):
            scored = []
            for episode_index, item in enumerate(source_index):
                if item["task"]["name"] != task_name:
                    continue
                with np.load(args.source / item["file"]) as episode:
                    scored.append((int(episode["front_pick_mask"][0].sum()), episode_index, item))
            _, episode_index, item = min(scored, key=lambda row: row[0])
            selected_episode_indices.append(episode_index)
            source_labels[episode_index] = {
                "file": item["file"],
                "task": task_name,
                "pick_name": item["task"]["pick_name"],
            }
        episode_groups = [by_episode[index] for index in selected_episode_indices]
    else:
        # Single-target legacy datasets have one prompt per task.
        episode_groups = []
        seen_prompts = set()
        for episode_samples in by_episode.values():
            prompt = episode_samples[0]["prompt"]
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            episode_groups.append(episode_samples)

    for episode_samples in episode_groups:
        prompt = episode_samples[0]["prompt"]
        episode_index = int(episode_samples[0]["episode_index"])
        for phase, offset in (
            ("start", 0),
            ("middle", len(episode_samples) // 2),
            ("success", len(episode_samples) - 1),
        ):
            selected.append((prompt, phase, episode_samples[offset], source_labels.get(episode_index)))

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image_processor = processor.image_processor
    cells = []
    report_samples = []
    for prompt, phase, sample, source_label in selected:
        source = Image.open(args.dataset / sample["image"]).convert("RGB")
        transformed = image_processor.apply_transform(source)
        if tuple(transformed.shape) != (6, 224, 224):
            raise ValueError(f"Unexpected OpenVLA image tensor {tuple(transformed.shape)}")
        first = unnormalize(transformed[:3], image_processor.means[0], image_processor.stds[0])
        second = unnormalize(transformed[3:], image_processor.means[1], image_processor.stds[1])
        cell = Image.new("RGB", (448, 248), "black")
        cell.paste(first, (0, 24))
        cell.paste(second, (224, 24))
        draw = ImageDraw.Draw(cell)
        label = prompt if source_label is None else (
            f"{source_label['task']} | {source_label['pick_name']}"
        )
        draw.text((4, 4), f"{label} | {phase} | DINO(left), SigLIP(right)", fill="white")
        cells.append(cell)
        report_samples.append(
            {
                "index": sample["index"],
                "prompt": prompt,
                "phase": phase,
                "tensor_shape": list(transformed.shape),
                "finite": bool(np.isfinite(transformed.numpy()).all()),
                "source": source_label,
            }
        )

    columns = 3
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (448 * columns, 248 * rows), "black")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 448, (index // columns) * 248))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    report = {
        "passed": all(item["finite"] for item in report_samples),
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "source": str(args.source.resolve()) if args.source is not None else None,
        "resize_strategy": image_processor.image_resize_strategy,
        "samples": report_samples,
        "contact_sheet": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
