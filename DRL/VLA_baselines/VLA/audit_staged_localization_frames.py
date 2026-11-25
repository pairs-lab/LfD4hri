"""Export real stored teacher inputs at the NEW post-hover decision stage.

TRAIN-only physics pilot evidence, not candidate qualification, learned model
performance or universal visibility. Masks are audit-only, never policy inputs.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.audit_native_camera_tensors import transformer
from VLA.common.provenance import sha256_file


def localization_frames(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Malformed stored actions")
    horizontal = np.flatnonzero(np.linalg.norm(actions[:, :2], axis=1) > 1e-5)
    if not len(horizontal):
        raise ValueError("No post-hover horizontal decision in this trajectory")
    first_xy = int(horizontal[0])
    down = np.flatnonzero(actions[first_xy:, 2] < -.001)
    if not len(down):
        raise ValueError("No later descent: preserve this failed phase audit")
    return {"reset": 0, "first_xy_after_hover": first_xy,
            "first_down_after_xy": first_xy+int(down[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results_path = args.pilot/"results.json"
    results = json.loads(results_path.read_text())
    if (results.get("passed") is not True or results.get("training_performed") is not False
            or results.get("validation_used") is not False or results.get("final_test_used") is not False):
        raise ValueError("Only the completed TRAIN-only teacher pilot may be previewed")
    transform, method = transformer(SimpleNamespace(model="smolvla", checkpoint=str(args.checkpoint)))
    args.output.mkdir(parents=True)
    rows, panels = [], []
    for item in results["rows"]:
        filename = item["original_file"]
        path = args.pilot/Path(filename).stem/"teacher_candidate.npz"
        digest = sha256_file(path)
        folder = args.output/Path(filename).stem
        folder.mkdir()
        with np.load(path, allow_pickle=False) as arrays:
            frames = localization_frames(arrays["canonical_actions"])
            metrics = []
            for label, frame in frames.items():
                images = []
                for camera in ("front", "wrist"):
                    rgb, mask = arrays[camera+"_rgb"][frame], arrays[camera+"_pick_mask"][frame]
                    native = transform(rgb, mask)
                    if len(native) != 1:
                        raise ValueError("Expected one actual SmolVLA image geometry")
                    _name, image, native_mask = native[0]
                    Image.fromarray(rgb).save(folder/f"{label}_{camera}_raw.png")
                    Image.fromarray(image).save(folder/f"{label}_{camera}_native.png")
                    ys, xs = np.nonzero(native_mask)
                    metrics.append({"stage": label, "frame": frame, "camera": camera,
                        "raw_target_pixels": int(mask.sum()), "native_target_pixels": int(native_mask.sum()),
                        "native_target_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                        "native_rgb_sha256": sha256_file(folder/f"{label}_{camera}_native.png")})
                    images.append(Image.fromarray(image).resize((256, 256)))
                if label != "reset":
                    panel = Image.new("RGB", (512, 282), "white")
                    panel.paste(images[0], (0, 26)); panel.paste(images[1], (256, 26))
                    ImageDraw.Draw(panel).text((3, 3), f"{item['task']['name']} / {item['task']['pick_name']} / {label} / {frame}", fill="black")
                    panels.append(panel)
            rows.append({"file": filename, "teacher_npz_sha256": digest, "frames": frames, "metrics": metrics})
        if sha256_file(path) != digest:
            raise ValueError("Stored pilot payload changed during preview")
    sheet = Image.new("RGB", (1024, 282*((len(panels)+1)//2)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (512*(index % 2), 282*(index//2)))
    sheet_path = args.output/"post_hover_contact_sheet.png"
    sheet.save(sheet_path)
    report = {"scope": "actual_stored_TRAIN_teacher_post_hover_images_NOT_qualification_or_manual_review",
        "pilot_results_sha256": sha256_file(results_path), "image_transform": method,
        "checkpoint_config_sha256": sha256_file(args.checkpoint/"config.json"),
        "all_predeclared_pilot_trials_included": True, "training_performed": False,
        "mask_is_model_input": False, "model_success_established": False,
        "contact_sheet_sha256": sha256_file(sheet_path), "rows": rows}
    (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"trials": len(rows), "scope": report["scope"]}))


if __name__ == "__main__":
    main()
