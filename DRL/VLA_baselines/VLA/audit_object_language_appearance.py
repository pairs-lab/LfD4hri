"""Render every named object and flag possible color-word/texture conflicts.

The ROI is privileged audit evidence only, never a model input. Palette matching
is a heuristic requiring visual review, not an automatic rename/recolor rule.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import COLORS, PICK_TARGETS, PLACE_TARGETS
from VLA.common.environment import ManipulationEnv
from VLA.common.spatial import FIXED_DESTINATION
from VLA.common.tasks import TASKS


def normalized_color(name):
    color = np.asarray(COLORS[name][:3], dtype=np.float64)
    if color.max() > 1:
        color /= 255
    return color / max(float(color.sum()), 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    names = list(dict.fromkeys([*PICK_TARGETS, *(name for name in PLACE_TARGETS if name not in FIXED_DESTINATION)]))
    sheet = Image.new("RGB", (256 * 4, 272 * ((len(names) + 3) // 4)), "black")
    palette = {name: normalized_color(name) for name in COLORS if name not in ("white", "black", "gray")}
    rows = []
    with ManipulationEnv(robot="ur5e", image_size=384, seed=0) as env:
        for number, name in enumerate(names):
            color, kind = name.split(" ", 1)
            z = .03 if kind in ("block", "egg") else 0.
            task = replace(TASKS["touch"], prompt=f"Touch the {name}", pick_name=name, pick_position=(0., -.55, z))
            observation = env.reset(task)
            mask = env.target_mask("front", "pick")
            pixels = observation.front_rgb
            y, x = np.nonzero(mask)
            if not len(y):
                raise RuntimeError("Audit target not visible: " + name)
            mean = pixels[mask].astype(np.float64).mean(axis=0)
            chroma = mean / mean.sum()
            nearest = min(palette, key=lambda candidate: np.linalg.norm(chroma - palette[candidate]))
            expected_distance = float(np.linalg.norm(chroma - normalized_color(color)))
            closest_distance = float(np.linalg.norm(chroma - palette[nearest]))
            flagged = color in palette and nearest != color and expected_distance - closest_distance > .06
            Image.fromarray(pixels).save(args.output / f"{name.replace(' ', '_')}_front.png")
            # Zoom for human semantic review only; this crop must never be used
            # as a policy observation, since its location comes from segmentation.
            box = (max(0, int(x.min()) - 4), max(0, int(y.min()) - 4),
                   min(384, int(x.max()) + 5), min(384, int(y.max()) + 5))
            roi = Image.fromarray(pixels).crop(box)
            scale = min(240 / roi.width, 240 / roi.height)
            roi = roi.resize((round(roi.width * scale), round(roi.height * scale)), Image.Resampling.NEAREST)
            cell = Image.new("RGB", (256, 256), "#303030")
            cell.paste(roi, ((256 - roi.width) // 2, (256 - roi.height) // 2))
            sheet.paste(cell, ((number % 4) * 256, (number // 4) * 272 + 16))
            ImageDraw.Draw(sheet).text(((number % 4) * 256 + 4, (number // 4) * 272),
                                       f"{name} | nearest={nearest}" + (" REVIEW" if flagged else ""), fill="white")
            rows.append({"name": name, "mean_target_rgb": mean.tolist(), "target_chromaticity": chroma.tolist(),
                         "nominal_color": color, "closest_chromatic_palette": nearest,
                         "nominal_chroma_distance": expected_distance, "closest_chroma_distance": closest_distance,
                         "requires_color_word_review": flagged, "visible_pixels": int(mask.sum())})
    sheet.save(args.output / "object_appearance_contact_sheet.png")
    report = {"measurement": "object_language_appearance_audit_not_model_accuracy", "rows": rows,
              "flagged": [row["name"] for row in rows if row["requires_color_word_review"]],
              "scene_and_object_assets_modified": False, "policy_inputs_modified": False,
              "roi_is_audit_only_and_never_a_policy_input": True,
              "caveat": "Texture and lighting can alter color. Flags require human semantic review; do not automatically change object names or assets."}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
