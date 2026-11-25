"""Audit-only object crops from actual full sensor images, never VLA inputs.

First ordinary TRAIN touch row per target, at its first XY handoff. Selection
does not use learned outcomes. Texture/shadow colors need not equal rgba tint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import COLORS, PICK_TARGETS
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Keep prior appearances")
    paths = [source / "index.json", source / "dataset_manifest.json", Path(__file__).resolve()]
    hashes = {str(path): sha256_file(path) for path in paths}
    selected = {}
    for row in json.loads((source / "index.json").read_text()):
        if row["task"]["name"] == "touch" and not row.get("paired_scene_source") and not row.get("recovery_start_actions_not_training_labels"):
            selected.setdefault(row["task"]["pick_name"], row)
    if set(selected) != set(PICK_TARGETS):
        raise ValueError("Cannot silently omit vocabulary from the catalog")
    output.mkdir(parents=True)
    sheet = Image.new("RGB", (5*192, 4*188), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    for number, (name, row) in enumerate(selected.items()):
        path = source / row["file"]
        hashes[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            motion = np.flatnonzero(np.linalg.norm(archive["canonical_actions"][:, :2], axis=1) > .003)
            frame = int(motion[0]) if len(motion) else 0
            masks = {c: archive[c+"_pick_mask"][frame] for c in ("front", "wrist")}
            camera = max(masks, key=lambda c: int(masks[c].sum()))
            mask = masks[camera]
            image = archive[camera+"_rgb"][frame]
            y, x = np.nonzero(mask)
            if not len(x):
                raise ValueError("Target not visible in either selected view")
            xmin, xmax = max(0, int(x.min())-8), min(image.shape[1], int(x.max())+9)
            ymin, ymax = max(0, int(y.min())-8), min(image.shape[0], int(y.max())+9)
            crop = Image.fromarray(image[ymin:ymax, xmin:xmax])
            crop.thumbnail((176, 144))
            left, top = number % 5 * 192, number // 5 * 188
            draw.text((left+4, top+3), name, fill="black")
            draw.text((left+4, top+18), f"{camera} t={frame} AUDIT CROP", fill="black")
            sheet.paste(crop, (left+8, top+40))
            Image.fromarray(image).save(output / f"target{number:02}_{camera}_full384.png")
            rgba = np.asarray(COLORS[name.split(" ", 1)[0]], float)
            records.append({"name": name, "file": row["file"], "frame": frame, "camera": camera,
                "target_pixels": int(mask.sum()), "target_masked_rgb_median": np.median(image[mask], axis=0).tolist(),
                "nominal_config_rgba": rgba.tolist(), "full_sensor_image": f"target{number:02}_{camera}_full384.png"})
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Source or exporter changed")
    sheet.save(output / "audit_only_target_catalog.png")
    report = {"scope": "Audit-only appearance catalog from real TRAIN observations; cropped tiles NOT model inputs",
        "source": str(source), "source_hashes": hashes, "targets": len(records), "rows": records,
        "model_inputs_changed": False, "neural_outcomes_used_for_selection": False,
        "rgba_tint_NOT_expected_pixel_equality_with_texture_or_shadows": True}
    (output / "catalog.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"targets": len(records), "model_inputs_changed": False, "catalog": str(output)}), flush=True)


if __name__ == "__main__":
    main()
