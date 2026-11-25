#!/usr/bin/env python3
"""Inspect the actual openpi train-time augmentation on difficult trajectories.

Segmentation is used only in this audit.  The exact augmentation is applied to
an image/mask pair with identical randomness, and the image is compared against
``preprocess_observation(train=True)`` to prove it is the tensor seen by loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--color-safe-augmentation", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import augmax
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi_client import image_tools

    from VLA.audit_finetune_data import mask_contrast

    args = parse_args()
    augmentation_profile = {"profile": "upstream_default", "hue_jitter": .1}
    if args.color_safe_augmentation:
        from VLA.pi0.augmentation import install_colour_safe_augmentation
        augmentation_profile = install_colour_safe_augmentation()
        augmentation_profile["patch_source_sha256"] = sha256(Path(__file__).with_name("augmentation.py"))
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    selected = json.loads(args.selection_report.read_text())
    if isinstance(selected, dict):
        selected = selected["selected_difficult_frames"]
    unique = {}
    for row in selected:
        unique.setdefault(row["file"], {**row, "pick": row.get("pick", row.get("pick_name"))})
    selected = list(unique.values())
    rows = []
    cells = []
    max_image_error = 0.0
    failures = []
    keys = ("base_0_rgb", "left_wrist_0_rgb")
    for selection in selected:
        with np.load(args.dataset / selection["file"]) as archive:
            fronts = archive["front_rgb"]
            wrists = archive["wrist_rgb"]
            front_masks = archive["front_pick_mask"]
            wrist_masks = archive["wrist_pick_mask"]
        for phase, step in (("reset", 0), ("middle", len(fronts) // 2), ("success", len(fronts) - 1)):
            images = [
                np.asarray(image_tools.resize_with_pad(array[step], 224, 224))
                for array in (fronts, wrists)
            ]
            masks = [
                np.asarray(Image.fromarray(array[step].astype(np.uint8) * 255).resize(
                    (224, 224), Image.Resampling.BOX
                )) >= 128
                for array in (front_masks, wrist_masks)
            ]
            observation = model_api.Observation(
                images={key: jnp.asarray(image[None], dtype=jnp.float32) / 127.5 - 1.0
                        for key, image in zip(keys, images, strict=True)},
                image_masks={key: jnp.array([True]) for key in keys},
                state=jnp.zeros((1, 32), dtype=jnp.float32),
            )
            phase_rows = []
            phase_views = []
            for seed in range(args.seeds):
                rng = jax.random.key(seed)
                actual = model_api.preprocess_observation(
                    rng, observation, train=True, image_keys=keys
                )
                sample = {"task": selection["task"], "file": selection["file"],
                          "pick": selection["pick"], "phase": phase, "step": step, "seed": seed}
                views = []
                for key, image, mask in zip(keys, images, masks, strict=True):
                    transforms = []
                    if "wrist" not in key:
                        transforms.extend([
                            augmax.RandomCrop(int(224 * 0.95), int(224 * 0.95)),
                            augmax.Resize(224, 224), augmax.Rotate((-5, 5)),
                        ])
                    transforms.append(augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5))
                    chain = augmax.Chain(*transforms, input_types=(augmax.InputType.IMAGE, augmax.InputType.MASK))
                    image_out, mask_out = chain(
                        jax.random.split(rng, 1)[0],
                        (jnp.asarray(image, dtype=jnp.float32) / 255.0,
                         jnp.asarray(mask[..., None], dtype=jnp.float32)),
                    )
                    expected = np.asarray(image_out) * 2.0 - 1.0
                    max_image_error = max(max_image_error, float(np.max(np.abs(
                        np.asarray(actual.images[key])[0] - expected
                    ))))
                    rgb = np.rint(np.clip(np.asarray(image_out), 0, 1) * 255).astype(np.uint8)
                    target_mask = np.asarray(mask_out)[..., 0] >= 0.5
                    yy, xx = np.nonzero(target_mask)
                    sample[key] = {
                        "target_pixels": int(target_mask.sum()),
                        "minimum_dimension_px": (
                            0 if not len(xx) else int(min(np.ptp(xx) + 1, np.ptp(yy) + 1))
                        ),
                        "contrast": mask_contrast(rgb, target_mask),
                        "finite": bool(np.isfinite(np.asarray(actual.images[key])).all()),
                    }
                    views.append(rgb)
                if not all(sample[key]["finite"] for key in keys):
                    failures.append(f"{selection['file']}:{step}:{seed} non-finite augmentation")
                if max(sample[key]["target_pixels"] for key in keys) < 35:
                    failures.append(f"{selection['file']}:{step}:{seed} target erased by augmentation")
                phase_rows.append(sample)
                phase_views.append(views)
            # Show the least contrasting front augmentation, not a pleasant
            # random sample.  All other realizations remain in the JSON report.
            worst = min(range(args.seeds), key=lambda i: phase_rows[i][keys[0]]["contrast"])
            row = phase_rows[worst]
            cell = Image.new("RGB", (448, 248), "black")
            for camera, rgb in enumerate(phase_views[worst]):
                cell.paste(Image.fromarray(rgb), (camera * 224, 24))
            ImageDraw.Draw(cell).text(
                (4, 5), f"{selection['task']} {phase}: {selection['pick']} | seed {worst}", fill="white"
            )
            cells.append(cell)
            rows.extend(phase_rows)
    if max_image_error > 1e-5:
        failures.append(f"Image differs from actual train-time preprocessing: {max_image_error}")
    sheet = Image.new("RGB", (448 * 3, 248 * len(selected)), "black")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % 3) * 448, (i // 3) * 248))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    report = {
        "passed": not failures, "failures": failures,
        "dataset": str(args.dataset.resolve()),
        "augmentation_profile": augmentation_profile,
        "source_index_sha256": sha256(args.dataset / "index.json"),
        "source_manifest_sha256": sha256(args.dataset / "dataset_manifest.json"),
        "actual_preprocessor_source": str(Path(model_api.__file__).resolve()),
        "actual_preprocessor_sha256": sha256(Path(model_api.__file__)),
        "seeds_per_frame": args.seeds, "frames": len(rows),
        "max_actual_image_error": max_image_error,
        "summary": {
            key: {
                "target_pixels_min": min(row[key]["target_pixels"] for row in rows),
                "contrast_min": min(row[key]["contrast"] for row in rows),
                "nonzero_target_fraction": float(np.mean([row[key]["target_pixels"] > 0 for row in rows])),
            } for key in keys
        },
        "contact_sheet": str(args.output.resolve()), "rows": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
