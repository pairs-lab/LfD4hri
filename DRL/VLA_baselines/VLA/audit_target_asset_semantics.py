"""Export actual native policy pixels for each named object, audit-only crops.

No generated images, recoloring, scene changes or segmentation policy inputs.
Crops are explicitly magnified for human inspection; full native sensor PNGs
are exported beside them to expose actual policy resolution and context.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from config import PICK_TARGETS
from VLA.audit_native_camera_tensors import transformer
from VLA.common.provenance import sha256_file
from VLA.smolvla.audit_conditioning import verify_explicit_source_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "checkpoint", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_explicit_source_audit(args.source, args.source_audit)
    rows = json.loads((args.source/"index.json").read_text())
    chosen = {}
    for row in rows:
        if row["task"]["name"] == "pick":
            chosen.setdefault(row["task"]["pick_name"], row)
    if set(chosen) != set(PICK_TARGETS):
        raise ValueError("Asset semantic audit must cover all nineteen requested names")
    convert, method = transformer(SimpleNamespace(model="smolvla", checkpoint=str(args.checkpoint)))
    args.output.mkdir(parents=True)
    entries, cells = [], []
    for name, row in sorted(chosen.items()):
        with np.load(args.source/row["file"], allow_pickle=False) as episode:
            actions = episode["canonical_actions"]
            closing = np.flatnonzero(actions[:, 6] > 0)
            frames = [("reset", 0), ("first_close_pre_action", int(closing[0]) if len(closing) else len(actions)-1)]
            cell = Image.new("RGB", (800, 154), "#eeeeee")
            draw = ImageDraw.Draw(cell)
            draw.text((8, 3), name+" | audit crops enlarged; native target sizes below", fill="black")
            for column, (phase, frame, camera) in enumerate((phase, frame, camera)
                    for phase, frame in frames for camera in ("front", "wrist")):
                native = convert(episode[camera+"_rgb"][frame], episode[camera+"_pick_mask"][frame])[0]
                backbone, rgb, mask = native
                yy, xx = np.where(mask)
                filename = f"{name.replace(' ', '_')}_{phase}_{camera}_native.png"
                Image.fromarray(rgb).save(args.output/filename)
                width, height = 0, 0
                if len(xx):
                    x0, x1, y0, y1 = int(xx.min()), int(xx.max()+1), int(yy.min()), int(yy.max()+1)
                    width, height = x1-x0, y1-y0
                    # Include true neighboring pixels, never replace background.
                    pad = 4
                    crop = Image.fromarray(rgb[max(0,y0-pad):min(len(rgb),y1+pad), max(0,x0-pad):min(rgb.shape[1],x1+pad)])
                    crop.thumbnail((186, 102), Image.Resampling.NEAREST)
                    # thumbnail cannot enlarge, so explicitly fit with unchanged RGB.
                    scale = min(186/crop.width, 102/crop.height)
                    crop = crop.resize((max(1,int(crop.width*scale)), max(1,int(crop.height*scale))), Image.Resampling.NEAREST)
                    cell.paste(crop, (column*200+(200-crop.width)//2, 33))
                else:
                    draw.text((column*200+8, 60), "TARGET NOT VISIBLE", fill="red")
                draw.text((column*200+8, 20), phase+" / "+camera, fill="black")
                draw.text((column*200+8, 138), f"native {width}x{height}, {int(mask.sum())}px", fill="black")
                entries.append({"target_name": name, "prompt": row["task"]["prompt"], "episode": row["file"],
                    "phase": phase, "frame": frame, "camera": camera, "native_target_width": width,
                    "native_target_height": height, "native_target_pixels": int(mask.sum()),
                    "native_png": filename, "native_png_sha256": sha256_file(args.output/filename), "backbone": backbone})
            cells.append(cell)
    sheets = []
    for first in range(0, len(cells), 7):
        subset = cells[first:first+7]
        sheet = Image.new("RGB", (800, len(subset)*154), "white")
        for number, cell in enumerate(subset):
            sheet.paste(cell, (0, number*154))
        filename = f"asset_semantics_{first//7+1}.png"
        sheet.save(args.output/filename)
        sheets.append({"file": filename, "sha256": sha256_file(args.output/filename)})
    verify_explicit_source_audit(args.source, args.source_audit)
    report = {"scope": "actual_native_asset_semantic_inspection_NOT_machine_recognition_certificate",
        "source": str(args.source.resolve()), "source_index_sha256": sha256_file(args.source/"index.json"),
        "source_audit_sha256": sha256_file(args.source_audit), "method": method,
        "targets": sorted(chosen), "native_images_exported": len(entries), "samples": entries, "sheets": sheets,
        "segmentation_supplied_to_policy": False, "training_performed": False, "final_test_used": False,
        "notes": "Reset/first-close crops are magnified audit references; full PNGs are actual native model pixels."}
    (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
