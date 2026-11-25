"""Actual model-native inputs of offline TRAIN teacher repairs, no certification."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.audit_native_camera_tensors import transformer
from VLA.audit_staged_localization_frames import localization_frames
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    transform, method = transformer(SimpleNamespace(model="smolvla", checkpoint=str(args.checkpoint)))
    args.output.mkdir(parents=True)
    cells, rows = [], []
    for repair in args.repair:
        report = json.loads((repair/"results.json").read_text())
        if not report["passed"] or report["training_performed"] or report["final_test_used"]:
            raise ValueError("Passed offline development repair only")
        payload = repair/report["episode"]
        digest = sha256_file(payload)
        with np.load(payload, allow_pickle=False) as arrays:
            frames = {**localization_frames(arrays["canonical_actions"]), "last_pre_action": len(arrays["canonical_actions"])-1}
            yaw = np.flatnonzero(np.abs(arrays["canonical_actions"][:, 5]) > .025)
            if len(yaw):
                frames["first_yaw_at_hover"] = int(yaw[0])
            for stage, frame in sorted(frames.items(), key=lambda row: row[1]):
                views = []
                for camera in ("front", "wrist"):
                    native = transform(arrays[camera+"_rgb"][frame], arrays[camera+"_pick_mask"][frame])
                    if len(native) != 1:
                        raise ValueError("Expected one actual SmolVLA stream per camera")
                    _, rgb, mask = native[0]
                    path = args.output/f"{repair.name}_{stage}_{camera}.png"
                    Image.fromarray(rgb).save(path)
                    ys, xs = np.nonzero(mask)
                    rows.append({"repair": str(repair.resolve()), "source_payload_sha256": digest,
                        "stage": stage, "frame": frame, "camera": camera, "path": str(path.resolve()),
                        "sha256": sha256_file(path), "target_pixels": int(mask.sum()),
                        "target_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None})
                    views.append(Image.fromarray(rgb).resize((256, 256)))
                cell = Image.new("RGB", (512, 282), "white")
                cell.paste(views[0], (0, 26)); cell.paste(views[1], (256, 26))
                ImageDraw.Draw(cell).text((3, 3), f"{repair.name} / {stage} / {frame}", fill="black")
                cells.append(cell)
        if sha256_file(payload) != digest:
            raise RuntimeError("Stored repair changed during extraction")
    sheet = Image.new("RGB", (1024, 282*((len(cells)+1)//2)), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, (512*(i % 2), 282*(i//2)))
    sheet.save(args.output/"contact_sheet.png")
    (args.output/"audit.json").write_text(json.dumps({"scope": "actual offline repair inputs_NOT_full_candidate_qualification",
        "model": "smolvla", "method": method, "rows": rows,
        "contact_sheet_sha256": sha256_file(args.output/"contact_sheet.png"),
        "weights_loaded": False, "manual_review_completed_by_this_tool": False,
        "training_performed": False, "final_test_used": False}, indent=2)+"\n")
    print(json.dumps({"images": len(rows), "manual_review_required": True}))


if __name__ == "__main__":
    main()
